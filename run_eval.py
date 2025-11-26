import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from utils import load_db, query_db, get_schema

# Load environment variables from .env file
load_dotenv()

# Initialize Fireworks client (uses OpenAI-compatible API)
client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)

# Model pricing per 1M tokens (input/output combined, approximate)
# Note: These are approximate prices - check Fireworks pricing page for exact rates
# Update these values based on actual Fireworks pricing
MODEL_PRICING = {
    "accounts/fireworks/models/llama-v3p1-8b-instruct": {
        "input_price_per_1M": 0.20,  # $0.20 per 1M input tokens
        "output_price_per_1M": 0.20   # $0.20 per 1M output tokens
    },
    "accounts/fireworks/models/gpt-oss-20b": {
        "input_price_per_1M": 0.07,  # Updated based on search: $0.07 per 1M tokens
        "output_price_per_1M": 0.07
    },
    "accounts/fireworks/models/gpt-oss-120b": {
        "input_price_per_1M": 0.15,  # Based on search: $0.15 per 1M tokens
        "output_price_per_1M": 0.15
    },
    "accounts/fireworks/models/gpt-oss-safeguard-20b": {
        "input_price_per_1M": 0.07,  # Fallback pricing
        "output_price_per_1M": 0.07
    },
    "accounts/fireworks/models/kimi-k2": {
        "input_price_per_1M": 0.60,
        "output_price_per_1M": 0.60
    },
    "accounts/fireworks/models/llama4-maverick-instruct-basic": {
        "input_price_per_1M": 0.20,  # Approximate pricing - update based on actual Fireworks pricing
        "output_price_per_1M": 0.20
    },
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct": {
        "input_price_per_1M": 0.90,  # Updated based on search: $0.90 per 1M tokens
        "output_price_per_1M": 0.90
    },
    "accounts/fireworks/models/qwen-2.5-vl-32b": {
        "input_price_per_1M": 0.90,  # Fallback pricing
        "output_price_per_1M": 0.90
    },
    "accounts/fireworks/models/qwen-2.5-32b": {
        "input_price_per_1M": 0.80,
        "output_price_per_1M": 0.80
    },
    # Alternative model name formats (try these if above don't work)
    "accounts/fireworks/models/gpt-oss-20b-instruct": {
        "input_price_per_1M": 0.50,
        "output_price_per_1M": 0.50
    },
    "accounts/fireworks/models/kimi-k2-instruct": {
        "input_price_per_1M": 0.60,
        "output_price_per_1M": 0.60
    },
    "accounts/fireworks/models/qwen2.5-32b-instruct": {
        "input_price_per_1M": 0.80,
        "output_price_per_1M": 0.80
    }
}

# Model parameter sizes in billions (B)
MODEL_PARAMETERS = {
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 8,
    "accounts/fireworks/models/gpt-oss-20b": 20,
    "accounts/fireworks/models/gpt-oss-120b": 120,
    "accounts/fireworks/models/gpt-oss-safeguard-20b": 20,
    "accounts/fireworks/models/gpt-oss-20b-instruct": 20,
    "accounts/fireworks/models/llama4-maverick-instruct-basic": 401,  # User specified 401B
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct": 32,
    "accounts/fireworks/models/qwen-2.5-vl-32b": 32,
    "accounts/fireworks/models/qwen-2.5-32b": 32,
    "accounts/fireworks/models/qwen2.5-32b-instruct": 32,
    "accounts/fireworks/models/kimi-k2": 0,  # Unknown
    "accounts/fireworks/models/kimi-k2-instruct": 0,  # Unknown
}

def get_model_parameters(model: str) -> int:
    """Get model parameter size in billions. Returns 0 if unknown."""
    # Try exact match first
    if model in MODEL_PARAMETERS:
        return MODEL_PARAMETERS[model]
    
    # Try to extract from model name
    model_lower = model.lower()
    if "8b" in model_lower or "-8b-" in model_lower:
        return 8
    elif "20b" in model_lower or "-20b-" in model_lower:
        return 20
    elif "32b" in model_lower or "-32b-" in model_lower:
        return 32
    elif "120b" in model_lower or "-120b-" in model_lower:
        return 120
    elif "401b" in model_lower or "-401b-" in model_lower or "maverick" in model_lower:
        return 401
    
    return 0  # Unknown

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        # Default pricing if model not found
        return (prompt_tokens * 0.50 / 1_000_000) + (completion_tokens * 0.50 / 1_000_000)
    
    pricing = MODEL_PRICING[model]
    input_cost = (prompt_tokens * pricing["input_price_per_1M"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output_price_per_1M"]) / 1_000_000
    return input_cost + output_cost

def generate_sql(question: str, schema: dict, model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct") -> tuple:
    """
    Generate SQL query from natural language question using Fireworks API.
    Uses streaming to measure TTFT (Time To First Token) and tokens/sec.
    
    Returns:
        tuple: (sql, metrics_dict) where metrics_dict contains:
            - latency_ms: Total API call latency in milliseconds
            - ttft_ms: Time to first token in milliseconds
            - tokens_per_sec: Generation speed (completion tokens per second)
            - total_tokens_per_sec: Overall throughput (total tokens per second)
            - cost: Cost in USD
            - prompt_tokens: Input tokens used
            - completion_tokens: Output tokens used
            - total_tokens: Total tokens used
    """
    # Build schema context for the prompt
    schema_text = "\n".join([
        f"Table {table}: {', '.join([col['name'] for col in cols])}"
        for table, cols in schema.items()
    ])
    
    prompt = f"""Convert this question to SQL. Use the following database schema:

{schema_text}

Question: {question}

SQL Query:"""
    
    # Use streaming to measure TTFT
    start_time = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=True
    )
    
    # Just measure TTFT from streaming
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                break  # Got first token, can break
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    
    # Now make the actual non-streaming call for content and accurate token counts
    call_start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    latency_ms = (time.time() - call_start_time) * 1000
    
    # Extract token usage
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else (prompt_tokens + completion_tokens)
    
    # Calculate tokens/sec metrics
    # Generation time = total latency - TTFT (time spent generating after first token)
    generation_time_sec = (latency_ms - ttft_ms) / 1000.0 if ttft_ms < latency_ms and ttft_ms > 0 else latency_ms / 1000.0
    tokens_per_sec = completion_tokens / generation_time_sec if generation_time_sec > 0 and completion_tokens > 0 else 0
    total_tokens_per_sec = total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0
    
    # Calculate cost
    cost = calculate_cost(prompt_tokens, completion_tokens, model)
    
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from API")
    sql = content.strip()
    # Clean up SQL (remove markdown code blocks if present)
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()
    
    metrics = {
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_tokens_per_sec": round(total_tokens_per_sec, 2),
        "cost": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    
    return sql, metrics

def get_foreign_keys(conn, schema: dict) -> dict:
    """Get foreign key relationships from the database."""
    foreign_keys = {}
    for table_name in schema.keys():
        fk_query = f"PRAGMA foreign_key_list({table_name})"
        try:
            fks = query_db(conn, fk_query, return_as_df=False)
            if fks:
                foreign_keys[table_name] = fks
        except Exception:
            pass
    return foreign_keys

def generate_sql_improved(question: str, schema: dict, foreign_keys: dict, model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct") -> tuple:
    """
    Generate SQL query with enhanced context including data types, PKs, FKs, and few-shot examples.
    
    Returns:
        tuple: (sql, metrics_dict) where metrics_dict contains:
            - latency_ms: API call latency in milliseconds
            - cost: Cost in USD
            - prompt_tokens: Input tokens used
            - completion_tokens: Output tokens used
            - total_tokens: Total tokens used
    """
    
    # Build detailed schema with data types and primary keys
    schema_parts = []
    for table_name, columns in schema.items():
        col_details = []
        for col in columns:
            col_str = col['name']
            # Add data type
            col_str += f" ({col['type']})"
            # Mark primary key
            if col['pk'] > 0:
                col_str += " [PRIMARY KEY]"
            # Mark nullable
            if col['notnull'] == 0:
                col_str += " [NULLABLE]"
            col_details.append(col_str)
        
        schema_parts.append(f"Table {table_name}:\n  " + "\n  ".join(col_details))
    
    # Build foreign key relationships
    fk_text = []
    for table_name, fks in foreign_keys.items():
        for fk in fks:
            fk_text.append(
                f"{table_name}.{fk['from']} -> {fk['table']}.{fk['to']}"
            )
    
    schema_text = "\n\n".join(schema_parts)
    fk_text_str = "\n".join(fk_text) if fk_text else "No explicit foreign keys defined."
    
    # Few-shot examples
    few_shot_examples = """Here are some examples:

Example 1:
Question: How many customers does each country have?
SQL: SELECT Country, COUNT(*) as CustomerCount FROM Customer GROUP BY Country ORDER BY CustomerCount DESC

Example 2:
Question: List all albums by the artist 'Aerosmith'
SQL: SELECT al.Title FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Aerosmith'"""
    
    prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries for a music store database.

DATABASE CONTEXT:
This is a digital music store database (Chinook) containing information about artists, albums, tracks, customers, invoices, playlists, and sales transactions.

DATABASE SCHEMA:
{schema_text}

FOREIGN KEY RELATIONSHIPS:
{fk_text_str}

SQL GUIDELINES:
- Use proper JOIN syntax to connect related tables
- Use table aliases for readability (e.g., 'g' for Genre, 't' for Track)
- For aggregations, use appropriate GROUP BY clauses
- Use ORDER BY for sorting results
- Use LIMIT when the question asks for "top N" or specific counts
- Pay attention to date filtering using strftime() for SQLite
- Use proper column names from the schema above
- If the query asks for "names" and the table has FirstName and LastName columns, select both columns separately (e.g., SELECT FirstName, LastName) rather than concatenating them

{few_shot_examples}

Now convert this question to SQL. Return ONLY the SQL query, no explanations or markdown formatting:

Question: {question}

SQL:"""
    
    # Use streaming to measure TTFT
    start_time = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=True
    )
    
    # Just measure TTFT from streaming
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                break  # Got first token, can break
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    
    # Now make the actual non-streaming call for content and accurate token counts
    call_start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    latency_ms = (time.time() - call_start_time) * 1000
    
    # Extract token usage
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else (prompt_tokens + completion_tokens)
    
    # Calculate tokens/sec metrics
    generation_time_sec = (latency_ms - ttft_ms) / 1000.0 if ttft_ms < latency_ms and ttft_ms > 0 else latency_ms / 1000.0
    tokens_per_sec = completion_tokens / generation_time_sec if generation_time_sec > 0 and completion_tokens > 0 else 0
    total_tokens_per_sec = total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0
    
    # Calculate cost
    cost = calculate_cost(prompt_tokens, completion_tokens, model)
    
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from API")
    sql = content.strip()
    # Clean up SQL (remove markdown code blocks if present)
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()
    
    metrics = {
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_tokens_per_sec": round(total_tokens_per_sec, 2),
        "cost": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    
    return sql, metrics

def normalize_column_name(col_name: str) -> str:
    """Normalize column names for comparison (remove function prefixes, lowercase)."""
    # Remove common SQL function prefixes and normalize
    col_lower = col_name.lower()
    # Handle COUNT(...), AVG(...), SUM(...) patterns
    if '(' in col_lower and ')' in col_lower:
        # Extract the alias or column name after AS
        return col_lower
    return col_lower

def normalize_row_for_comparison(row: dict) -> dict:
    """Normalize a row for comparison by sorting keys and normalizing column names."""
    # Create normalized dict with lowercase keys
    normalized = {}
    for key, value in row.items():
        norm_key = normalize_column_name(key)
        # Handle numeric values (round floats for comparison)
        if isinstance(value, float):
            normalized[norm_key] = round(value, 10)
        else:
            normalized[norm_key] = value
    return dict(sorted(normalized.items()))

def order_matters_for_question(question: str, expected_sql: str = "") -> bool:
    """
    Determine if order matters for a given question.
    Order matters if the question asks for:
    - "top N", "first N", "last N"
    - "most", "least", "highest", "lowest", "longest", "shortest"
    - "best", "worst"
    - Or if the expected SQL has ORDER BY with LIMIT
    """
    question_lower = question.lower()
    sql_lower = expected_sql.lower()
    
    # Check for ordering keywords in question
    ordering_keywords = [
        "top ", "first ", "last ", "most ", "least ", 
        "highest", "lowest", "longest", "shortest",
        "best ", "worst ", "order by"
    ]
    
    if any(keyword in question_lower for keyword in ordering_keywords):
        return True
    
    # Check if SQL has ORDER BY with LIMIT (indicates ordering is important)
    if "order by" in sql_lower and "limit" in sql_lower:
        return True
    
    return False

def compare_results(actual: list, expected: list, order_matters: bool = False) -> bool:
    """
    Compare actual query results with expected results.
    
    Args:
        actual: Actual query results
        expected: Expected query results
        order_matters: If True, compare lists in order. If False, compare as sets (order-agnostic).
    """
    # Normalize both lists for comparison
    actual_normalized = [normalize_row_for_comparison(row) for row in actual]
    expected_normalized = [normalize_row_for_comparison(row) for row in expected]
    
    if order_matters:
        return actual_normalized == expected_normalized
    else:
        # Order-agnostic comparison: convert to sets of tuples
        actual_set = set(tuple(sorted(row.items())) for row in actual_normalized)
        expected_set = set(tuple(sorted(row.items())) for row in expected_normalized)
        return actual_set == expected_set

def score_query(generated_sql: str, actual_result: list, expected_result: list, 
                has_syntax_error: bool = False, question: str = "", expected_sql: str = "") -> dict:
    """
    Score a query with partial credit.
    
    Scoring breakdown:
    - Syntax validity: 30% (SQL executes without errors)
    - Result matching: 70% (exact match = 70%, partial match = 70% * match_ratio)
    
    Args:
        generated_sql: The SQL query generated by the model
        actual_result: Results from executing the generated SQL
        expected_result: Expected results from ground truth
        has_syntax_error: Whether the SQL had a syntax error
        question: The natural language question (used to determine if order matters)
        expected_sql: The expected SQL query (used to determine if order matters)
    
    Returns dict with:
    - total_score: float between 0.0 and 1.0
    - syntax_score: float (0.0 or 0.3)
    - result_score: float (0.0 to 0.7)
    - breakdown: dict with detailed scoring info
    """
    # Determine if order matters for this question
    order_matters = order_matters_for_question(question, expected_sql)
    score_breakdown = {
        "syntax_score": 0.0,
        "result_score": 0.0,
        "total_score": 0.0,
        "details": {}
    }
    
    # Syntax Score (30%)
    if not has_syntax_error:
        score_breakdown["syntax_score"] = 0.3
        score_breakdown["details"]["syntax_valid"] = True
    else:
        score_breakdown["syntax_score"] = 0.0
        score_breakdown["details"]["syntax_valid"] = False
        score_breakdown["total_score"] = 0.0
        return score_breakdown
    
    # Result Score (70%)
    if not expected_result:
        # Empty expected result - check if actual is also empty
        if not actual_result:
            score_breakdown["result_score"] = 0.7
            score_breakdown["details"]["exact_match"] = True
        else:
            score_breakdown["result_score"] = 0.0
            score_breakdown["details"]["exact_match"] = False
    else:
        # Normalize both result sets
        actual_normalized = [normalize_row_for_comparison(row) for row in actual_result]
        expected_normalized = [normalize_row_for_comparison(row) for row in expected_result]
        
        # Check for exact match (using order-agnostic comparison if order doesn't matter)
        if order_matters:
            exact_match = actual_normalized == expected_normalized
        else:
            # Order-agnostic comparison: convert to sets
            actual_set = set(tuple(sorted(row.items())) for row in actual_normalized)
            expected_set = set(tuple(sorted(row.items())) for row in expected_normalized)
            exact_match = actual_set == expected_set
        
        if exact_match:
            score_breakdown["result_score"] = 0.7
            score_breakdown["details"]["exact_match"] = True
            score_breakdown["details"]["matching_rows"] = len(expected_result)
            score_breakdown["details"]["total_expected_rows"] = len(expected_result)
            score_breakdown["details"]["order_matters"] = order_matters
        else:
            # Partial credit: count matching rows
            # Always use set-based comparison for partial matching
            actual_set = set(tuple(sorted(row.items())) for row in actual_normalized)
            expected_set = set(tuple(sorted(row.items())) for row in expected_normalized)
            
            # Count exact row matches
            matching_rows = len(actual_set & expected_set)
            
            # Also check column structure match
            if actual_normalized and expected_normalized:
                actual_cols = set(actual_normalized[0].keys())
                expected_cols = set(expected_normalized[0].keys())
                column_match_ratio = len(actual_cols & expected_cols) / len(expected_cols) if expected_cols else 0.0
            else:
                column_match_ratio = 0.0
            
            # Calculate partial score: 70% * (matching_rows / total_expected_rows) * column_match_ratio
            # But give at least some credit if columns match
            if len(expected_result) > 0:
                row_match_ratio = matching_rows / len(expected_result)
                # Weight: 85% row matching, 15% column structure
                score_breakdown["result_score"] = 0.7 * (0.85 * row_match_ratio + 0.15 * column_match_ratio)
            else:
                score_breakdown["result_score"] = 0.0
            
            score_breakdown["details"]["exact_match"] = False
            score_breakdown["details"]["matching_rows"] = matching_rows
            score_breakdown["details"]["total_expected_rows"] = len(expected_result)
            score_breakdown["details"]["column_match_ratio"] = column_match_ratio
            score_breakdown["details"]["order_matters"] = order_matters
    
    # Total score
    score_breakdown["total_score"] = score_breakdown["syntax_score"] + score_breakdown["result_score"]
    
    return score_breakdown

def main(model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct", improved: bool = False):
    """
    Run evaluation on a specific model.
    
    Args:
        model: Model identifier to use for evaluation
        improved: If True, use improved prompt with enhanced context
    """
    # Load database and schema
    conn = load_db()
    schema = get_schema(conn)
    foreign_keys = get_foreign_keys(conn, schema) if improved else None
    
    # Load evaluation data
    with open("evaluation_data.json", "r") as f:
        eval_data = json.load(f)
    
    # Initialize metrics tracking
    total_latency_ms = 0.0
    total_ttft_ms = 0.0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_tokens_per_sec = 0.0
    total_generation_tokens_per_sec = 0.0
    
    results = {
        "model": model,
        "prompt_type": "improved" if improved else "baseline",
        "total": len(eval_data),
        "correct": 0,
        "syntax_errors": 0,
        "wrong_results": 0,
        "total_score": 0.0,
        "average_score": 0.0,
        "metrics": {
            "total_latency_ms": 0.0,
            "average_latency_ms": 0.0,
            "total_ttft_ms": 0.0,
            "average_ttft_ms": 0.0,
            "total_cost": 0.0,
            "average_cost_per_query": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "average_tokens_per_sec": 0.0,
            "average_generation_tokens_per_sec": 0.0
        },
        "details": []
    }
    
    model_display_name = model.split("/")[-1] if "/" in model else model
    prompt_type = "IMPROVED" if improved else "BASELINE"
    print(f"Running {prompt_type} evaluation on {model_display_name}...\n")
    print("=" * 80)
    
    for i, test_case in enumerate(eval_data, 1):
        question = test_case["question"]
        expected_result = test_case["expected_result"]
        expected_sql = test_case.get("sql", "")
        
        print(f"\n{i}. {question}")
        
        # Generate SQL using Fireworks API
        try:
            if improved:
                if foreign_keys is None:
                    foreign_keys = get_foreign_keys(conn, schema)
                generated_sql, api_metrics = generate_sql_improved(question, schema, foreign_keys, model=model)
            else:
                generated_sql, api_metrics = generate_sql(question, schema, model=model)
            
            # Track metrics
            total_latency_ms += api_metrics["latency_ms"]
            total_ttft_ms += api_metrics.get("ttft_ms", 0)
            total_cost += api_metrics["cost"]
            total_prompt_tokens += api_metrics["prompt_tokens"]
            total_completion_tokens += api_metrics["completion_tokens"]
            total_tokens += api_metrics["total_tokens"]
            if api_metrics.get("tokens_per_sec", 0) > 0:
                total_generation_tokens_per_sec += api_metrics["tokens_per_sec"]
            if api_metrics.get("total_tokens_per_sec", 0) > 0:
                total_tokens_per_sec += api_metrics["total_tokens_per_sec"]
            
            print(f"   Generated SQL: {generated_sql}")
            ttft_str = f"TTFT: {api_metrics.get('ttft_ms', 0):.2f}ms" if api_metrics.get("ttft_ms") else ""
            tokens_sec_str = f" | {api_metrics.get('tokens_per_sec', 0):.1f} tok/s" if api_metrics.get("tokens_per_sec") else ""
            print(f"   Latency: {api_metrics['latency_ms']:.2f}ms {ttft_str}{tokens_sec_str} | Cost: ${api_metrics['cost']:.6f} | Tokens: {api_metrics['total_tokens']}")
            
            # Execute generated SQL
            try:
                actual_result = query_db(conn, generated_sql, return_as_df=False)
                
                # Determine if order matters for this question
                order_matters = order_matters_for_question(question, expected_sql)
                
                # Score the query
                score_info = score_query(generated_sql, actual_result, expected_result, 
                                        has_syntax_error=False, question=question, expected_sql=expected_sql)
                results["total_score"] += score_info["total_score"]
                
                # Compare results (using order-agnostic comparison if order doesn't matter)
                if compare_results(actual_result, expected_result, order_matters=order_matters):
                    print(f"   ✓ Correct! Score: {score_info['total_score']:.2f} (Syntax: {score_info['syntax_score']:.2f}, Results: {score_info['result_score']:.2f})")
                    results["correct"] += 1
                    results["details"].append({
                        "question": question,
                        "status": "correct",
                        "generated_sql": generated_sql,
                        "score": score_info["total_score"],
                        "score_breakdown": score_info,
                        "metrics": api_metrics
                    })
                else:
                    print(f"   ✗ Wrong results. Score: {score_info['total_score']:.2f} (Syntax: {score_info['syntax_score']:.2f}, Results: {score_info['result_score']:.2f})")
                    if score_info["details"].get("matching_rows") is not None:
                        print(f"      Matching rows: {score_info['details']['matching_rows']}/{score_info['details']['total_expected_rows']}")
                    results["wrong_results"] += 1
                    results["details"].append({
                        "question": question,
                        "status": "wrong_results",
                        "generated_sql": generated_sql,
                        "expected": expected_result[:3],  # First 3 rows
                        "actual": actual_result[:3],
                        "score": score_info["total_score"],
                        "score_breakdown": score_info,
                        "metrics": api_metrics
                    })
            except Exception as e:
                print(f"   ✗ SQL execution error: {e}")
                score_info = score_query(generated_sql, [], expected_result, 
                                        has_syntax_error=True, question=question, expected_sql=expected_sql)
                results["total_score"] += score_info["total_score"]
                results["syntax_errors"] += 1
                results["details"].append({
                    "question": question,
                    "status": "syntax_error",
                    "generated_sql": generated_sql,
                    "error": str(e),
                    "score": score_info["total_score"],
                    "score_breakdown": score_info,
                    "metrics": api_metrics
                })
        except Exception as e:
            print(f"   ✗ API error: {e}")
            score_info = {"total_score": 0.0, "syntax_score": 0.0, "result_score": 0.0}
            results["total_score"] += 0.0
            results["details"].append({
                "question": question,
                "status": "api_error",
                "error": str(e),
                "score": 0.0,
                "score_breakdown": score_info,
                "metrics": {"latency_ms": 0.0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
    
    # Calculate averages
    results["average_score"] = results["total_score"] / results["total"] if results["total"] > 0 else 0.0
    results["metrics"]["total_latency_ms"] = total_latency_ms
    results["metrics"]["average_latency_ms"] = total_latency_ms / results["total"] if results["total"] > 0 else 0.0
    results["metrics"]["total_ttft_ms"] = total_ttft_ms
    results["metrics"]["average_ttft_ms"] = total_ttft_ms / results["total"] if results["total"] > 0 else 0.0
    results["metrics"]["total_cost"] = total_cost
    results["metrics"]["average_cost_per_query"] = total_cost / results["total"] if results["total"] > 0 else 0.0
    results["metrics"]["total_prompt_tokens"] = total_prompt_tokens
    results["metrics"]["total_completion_tokens"] = total_completion_tokens
    results["metrics"]["total_tokens"] = total_tokens
    # Calculate average tokens/sec (only for queries that had valid metrics)
    valid_queries = sum(1 for detail in results["details"] if detail.get("metrics", {}).get("tokens_per_sec", 0) > 0)
    results["metrics"]["average_tokens_per_sec"] = total_tokens_per_sec / valid_queries if valid_queries > 0 else 0.0
    results["metrics"]["average_generation_tokens_per_sec"] = total_generation_tokens_per_sec / valid_queries if valid_queries > 0 else 0.0
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"\nEvaluation Summary ({prompt_type} - {model_display_name}):")
    print(f"  Total questions: {results['total']}")
    print(f"  Correct: {results['correct']}")
    print(f"  Wrong results: {results['wrong_results']}")
    print(f"  Syntax errors: {results['syntax_errors']}")
    print(f"  Accuracy: {results['correct'] / results['total'] * 100:.1f}%")
    print(f"  Average Score: {results['average_score']:.3f} / 1.000")
    print(f"  Total Score: {results['total_score']:.3f} / {results['total']:.1f}")
    print(f"\n  Performance Metrics:")
    print(f"  Average Latency: {results['metrics']['average_latency_ms']:.2f}ms")
    print(f"  Average TTFT: {results['metrics']['average_ttft_ms']:.2f}ms")
    print(f"  Average Generation Speed: {results['metrics']['average_generation_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Average Throughput: {results['metrics']['average_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Total Cost: ${results['metrics']['total_cost']:.6f}")
    print(f"  Average Cost per Query: ${results['metrics']['average_cost_per_query']:.6f}")
    print(f"  Total Tokens: {results['metrics']['total_tokens']:,} (Prompt: {results['metrics']['total_prompt_tokens']:,}, Completion: {results['metrics']['total_completion_tokens']:,})")
    
    # Generate filename based on model and prompt type
    model_safe_name = model.split("/")[-1].replace("-", "_").replace(".", "_")
    prompt_type = 'improved' if improved else 'baseline'
    
    # Create directory if it doesn't exist
    results_dir = f"eval_results_{prompt_type}"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f"eval_results_{model_safe_name}_{prompt_type}.json")
    
    # Save detailed results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")
    conn.close()
    
    return results

def get_model_alternatives(base_model: str) -> list:
    """Get alternative model name variations to try if the base model fails."""
    alternatives = [base_model]  # Try original first
    
    # Common variations based on Fireworks model naming conventions
    if "gpt-oss" in base_model.lower():
        if "120b" in base_model:
            # gpt-oss-120b variations
            alternatives.extend([
                "accounts/fireworks/models/gpt-oss-120b",  # Correct name from Fireworks docs
                "accounts/fireworks/models/gpt-oss-120b-instruct"
            ])
        elif "20b" in base_model:
            # Based on search: correct name is gpt-oss-20b (not gpt-oss-safeguard-20b)
            alternatives.extend([
                "accounts/fireworks/models/gpt-oss-20b",  # Correct name from Fireworks docs
                "accounts/fireworks/models/gpt-oss-safeguard-20b",  # Try original if user specified
                "accounts/fireworks/models/gpt-oss-20b-instruct"
            ])
    elif "llama4" in base_model.lower() or "maverick" in base_model.lower():
        alternatives.extend([
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "accounts/fireworks/models/llama4-maverick-instruct",
            "accounts/fireworks/models/llama4-maverick-basic"
        ])
    elif "kimi-k2" in base_model.lower():
        alternatives.extend([
            "accounts/fireworks/models/kimi-k2",
            "accounts/fireworks/models/kimi-k2-thinking",
            "accounts/fireworks/models/kimi-k2-instruct",
            "accounts/fireworks/models/kimi-k2-thinking-instruct"
        ])
    elif "qwen" in base_model.lower() or "qwen2" in base_model.lower():
        # Based on search results: correct format is qwen2p5-vl-32b-instruct (confirmed)
        alternatives.extend([
            "accounts/fireworks/models/qwen2p5-vl-32b-instruct",  # Correct name confirmed from Fireworks docs
            "accounts/fireworks/models/qwen-2.5-vl-32b",  # Try user's original format
            "accounts/fireworks/models/qwen-2.5-vl-32b-instruct",
            "accounts/fireworks/models/qwen2.5-vl-32b-instruct"
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_alternatives = []
    for alt in alternatives:
        if alt not in seen:
            seen.add(alt)
            unique_alternatives.append(alt)
    
    return unique_alternatives

def run_multi_model_evaluation(models: list = None, improved: bool = True):  # type: ignore
    """
    Run evaluation across multiple models and generate comparison summary.
    
    Args:
        models: List of model identifiers to evaluate. If None, uses default models.
        improved: If True, use improved prompt with enhanced context
    """
    if models is None:
        # Try all requested models, with fallback to original model if they fail
        original_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        models = [
            original_model,  # Original working model (always included)
            "accounts/fireworks/models/gpt-oss-20b",  # Correct model name (not gpt-oss-safeguard-20b)
            "accounts/fireworks/models/gpt-oss-120b",  # Added 120B model
            "accounts/fireworks/models/llama4-maverick-instruct-basic",  # Replaced kimi-k2-thinking
            "accounts/fireworks/models/qwen2p5-vl-32b-instruct"  # Correct model name (confirmed from Fireworks docs)
        ]
    
    all_results = {}
    comparison_summary = []
    
    print("=" * 80)
    prompt_type = "IMPROVED" if improved else "BASELINE"
    print(f"Running Multi-Model Evaluation ({prompt_type} prompt)")
    print("=" * 80)
    
    original_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    
    for model in models:
        print(f"\n\n{'='*80}")
        print(f"Evaluating: {model}")
        print(f"{'='*80}\n")
        
        # Try model and alternatives if it fails
        model_alternatives = get_model_alternatives(model)
        success = False
        actual_model_used = None
        
        for alt_model in model_alternatives:
            try:
                if alt_model != model:
                    print(f"  Trying alternative: {alt_model}")
                results = main(model=alt_model, improved=improved)
                all_results[alt_model] = results
                actual_model_used = alt_model
                success = True
                
                # Add to comparison summary
                model_display_name = alt_model.split("/")[-1] if "/" in alt_model else alt_model
                # If this was a fallback, note it in the display name
                if alt_model == original_model and model != original_model:
                    model_display_name = f"{model.split('/')[-1]} (fallback: {model_display_name})"
                
                comparison_summary.append({
                    "model": model_display_name,
                    "requested_model": model.split("/")[-1] if "/" in model else model,
                    "actual_model": alt_model.split("/")[-1] if "/" in alt_model else alt_model,
                    "is_fallback": alt_model == original_model and model != original_model,
                    "prompt_type": "improved" if improved else "baseline",
                    "parameters_B": get_model_parameters(alt_model),
                    "accuracy": results["correct"] / results["total"] * 100 if results["total"] > 0 else 0.0,
                    "average_score": results["average_score"],
                    "correct": results["correct"],
                    "wrong_results": results["wrong_results"],
                    "syntax_errors": results["syntax_errors"],
                    "average_latency_ms": results["metrics"]["average_latency_ms"],
                    "average_ttft_ms": results["metrics"].get("average_ttft_ms", 0.0),
                    "average_tokens_per_sec": results["metrics"].get("average_generation_tokens_per_sec", 0.0),
                    "total_cost": results["metrics"]["total_cost"],
                    "average_cost_per_query": results["metrics"]["average_cost_per_query"],
                    "total_tokens": results["metrics"]["total_tokens"]
                })
                break  # Success, move to next model
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
                    if alt_model != model_alternatives[-1]:  # Not the last one
                        print(f"  ⚠️  Model not found: {alt_model}, trying alternatives...")
                    continue  # Try next alternative
                else:
                    # Different error - might be API key, network, etc.
                    print(f"  ✗ Error with {alt_model}: {e}")
                    if alt_model == model_alternatives[-1]:  # Last alternative
                        print(f"  All alternatives failed for {model}")
                    continue
        
        # If model failed and it's not the original, try using original as fallback
        if not success and model != original_model:
            print(f"\n⚠️  Model {model} not found, using original model as fallback...")
            try:
                print(f"  Using fallback: {original_model}")
                results = main(model=original_model, improved=improved)
                all_results[f"{model}_fallback_{original_model}"] = results
                actual_model_used = original_model
                success = True
                
                # Add to comparison summary with fallback notation
                model_display_name = f"{model.split('/')[-1]} (fallback: {original_model.split('/')[-1]})"
                comparison_summary.append({
                    "model": model_display_name,
                    "requested_model": model.split("/")[-1] if "/" in model else model,
                    "actual_model": original_model.split("/")[-1] if "/" in original_model else original_model,
                    "is_fallback": True,
                    "prompt_type": "improved" if improved else "baseline",
                    "parameters_B": get_model_parameters(original_model),
                    "accuracy": results["correct"] / results["total"] * 100 if results["total"] > 0 else 0.0,
                    "average_score": results["average_score"],
                    "correct": results["correct"],
                    "wrong_results": results["wrong_results"],
                    "syntax_errors": results["syntax_errors"],
                    "average_latency_ms": results["metrics"]["average_latency_ms"],
                    "average_ttft_ms": results["metrics"].get("average_ttft_ms", 0.0),
                    "average_tokens_per_sec": results["metrics"].get("average_generation_tokens_per_sec", 0.0),
                    "total_cost": results["metrics"]["total_cost"],
                    "average_cost_per_query": results["metrics"]["average_cost_per_query"],
                    "total_tokens": results["metrics"]["total_tokens"]
                })
            except Exception as e:
                print(f"  ✗ Fallback also failed: {e}")
        
        if not success:
            print(f"\n⚠️  Could not evaluate {model} - all attempts failed")
            print(f"   Tried alternatives: {', '.join(model_alternatives)}")
            if model != original_model:
                print(f"   Fallback to {original_model} also failed")
            print(f"   Please check the model name in Fireworks dashboard")
            print(f"   You can find available models at: https://app.fireworks.ai/models\n")
    
    # Print comparison summary
    print("\n\n" + "=" * 110)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 110)
    print(f"{'Model':<28} {'Size':<8} {'Acc%':<8} {'Score':<8} {'Latency':<10} {'TTFT':<10} {'Tok/s':<10} {'Cost/Q':<12} {'Total$':<10}")
    print("-" * 110)
    
    for summary in comparison_summary:
        model_name = summary["model"][:26]  # Truncate if too long
        params = summary.get("parameters_B", 0)
        params_str = f"{params}B" if params > 0 else "?"
        ttft = summary.get("average_ttft_ms", 0.0)
        tok_per_sec = summary.get("average_tokens_per_sec", 0.0)
        print(f"{model_name:<28} {params_str:<8} {summary['accuracy']:>6.1f}%  {summary['average_score']:>6.3f}  "
              f"{summary['average_latency_ms']:>8.1f}ms  {ttft:>8.1f}ms  {tok_per_sec:>8.1f}  "
              f"${summary['average_cost_per_query']:>10.6f}  ${summary['total_cost']:>8.6f}")
    
    # Save comparison summary
    comparison_filename = f"model_comparison_{'improved' if improved else 'baseline'}.json"
    with open(comparison_filename, "w") as f:
        json.dump({
            "prompt_type": "improved" if improved else "baseline",
            "models_evaluated": len(comparison_summary),
            "summary": comparison_summary,
            "detailed_results": all_results
        }, f, indent=2)
    
    print(f"\nComparison summary saved to {comparison_filename}")
    
    return comparison_summary, all_results

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if "--multi-model" in sys.argv or "--all-models" in sys.argv:
        # Determine prompt type: --baseline explicitly sets improved=False, otherwise default to improved=True
        if "--baseline" in sys.argv:
            improved = False
        elif "--improved" in sys.argv or "--all-models" in sys.argv:
            improved = True
        else:
            # Default to improved if neither --baseline nor --improved specified
            improved = True
        run_multi_model_evaluation(improved=improved)
    elif "--original-model" in sys.argv or "--llama" in sys.argv:
        # Run original working model (llama-v3p1-8b-instruct) with both prompts
        original_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        print("Running original model (llama-v3p1-8b-instruct) with both baseline and improved prompts...\n")
        
        print("=" * 80)
        print("BASELINE PROMPT")
        print("=" * 80)
        main(model=original_model, improved=False)
        
        print("\n\n" + "=" * 80)
        print("IMPROVED PROMPT")
        print("=" * 80)
        main(model=original_model, improved=True)
    elif "--improved" in sys.argv:
        # Default model with improved prompt
        main(improved=True)
    elif len(sys.argv) > 1 and sys.argv[1].startswith("accounts/"):
        # Custom model specified
        model = sys.argv[1]
        improved = "--improved" in sys.argv
        main(model=model, improved=improved)
    else:
        # Default: baseline prompt with default model
        main(improved=False)