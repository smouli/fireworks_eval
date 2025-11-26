# FireworksAI Applied AI Take-Home Assessment

This take-home reflects things that you would typically spend time on day-to-day in the role. It helps us understand your ability to:

1. **Understand a customer's problem** - Analyze business requirements and technical constraints
2. **Iterate on a solution** - Apply AI engineering methods to improve model performance
3. **Evaluate model quality** - Define metrics and systematically measure performance
4. **Model selection** - Choose appropriate models based on requirements and trade-offs
5. **Communicate effectively** - Present technical findings and recommendations clearly

Below is an email from Andrea, VP of Analytics at MelodyStream (a large music streaming platform), who is evaluating AI models to build an internal business intelligence tool that converts natural language queries to SQL.

## What We're Looking For

In responding to this take-home, you should:

1. **Define and measure quality** - Create an evaluation framework with clear metrics
2. **Iterate to improve performance** - Apply AI engineering techniques (prompt engineering, few-shot examples, etc..) to improve baseline results.
3. **Select and justify a model** - Choose an appropriate model and clearly explain your reasoning (cost, latency, accuracy trade-offs)
4. **Why FireworksAI**: Make a case to the client why FireworksAI is the right platform to solve this problem.
5. **Communicate professionally** - Draft a response email with findings, recommendations, code, and instructions

## Submission Guidelines

- **Time limit:** Please spend no more than **4 hours** on this assessment. Provide a next steps plan for how you would continue to improve the model.
- **Submission deadline:** Within 24 hours of receiving this assessment
- **Format:** Submit as you would a real customer communication (email response with readable clean code and instructions on how to run it)
- **Resources:** You may use the internet, documentation, python packages and any tools you'd use in your day-to-day work. If you use AI to generate code, please include the source of the AI tool in the email.
- **Questions:** If you have questions during the assessment, please reach out to [ravi@fireworks.ai]

**Note on scope:** We're more interested in your approach, thought process, and ability to make progress in a time-boxed manner than achieving perfect accuracy. 
Focus on demonstrating sound business, engineering, and communication skills.

## Email from the Customer

**From:** Andrea Chen <andrea.chen@melodystream.com>
**To:** Solutions Team <solutions@fireworks.ai>
**Subject:** Help Needed: Text-to-SQL for Internal BI Tool

Hi Fireworks team,

Following up on our conversation last week about building an internal business intelligence tool. Our analytics and business operations teams spend a lot of time writing SQL queries against our music catalog database, and we want to enable them to ask questions in natural language instead.

**Our situation:**

We have a production database (SQLite) with our music catalog data - artists, albums, tracks, customers, invoices, playlists, etc. Our teams need to answer questions like "What are the top-selling genres in Germany?" or "Which support rep has the most customers?" throughout the day.

**What we've tried:**

We put together a quick proof of concept using a simple prompt:

```
Convert this question to SQL:
{question}
```

We tested it on a handful of questions and the results were... mixed. Sometimes it works great, sometimes it hallucinates table names or writes invalid SQL. We're not sure how to systematically evaluate whether this is "good enough" or how to improve it.

**What we need help with:**

1. How do we measure if the model is actually working well? What's a good accuracy target?
2. How can we improve the performance from where we are now?
3. Which model should we use? We care about accuracy, but also cost and speed since this will be used frequently throughout the day.

**What we're providing:**

- Our music database (you can download it via the setup script in this repo)
- A set of test questions we've created with ground truth SQL queries for evaluation
- Utility code for querying the database

I'd love to hear your recommendations on how to move forward. We're hoping to make a decision on this in the next couple weeks.

Thanks,
Andrea Chen
VP of Analytics, MelodyStream

---

## Getting Started

### Setup

Run the setup script to create a virtual environment and download the database:

```bash
./setup.sh
```

Or manually:

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Download database
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```

### Database Schema

The database contains 11 tables modeling a digital music store:

- **Artist, Album, Track** - Music catalog
- **Customer, Employee** - People
- **Invoice, InvoiceLine** - Sales transactions
- **Playlist, PlaylistTrack** - Curated collections
- **Genre, MediaType** - Classification

Use the provided utility functions to explore:

```python
from utils import load_db, query_db, print_table_schema

# Load database
conn = load_db()

# View schema
print_table_schema(conn)

# Run a query
results = query_db(conn, "SELECT * FROM Artist LIMIT 5")
```

### Evaluation Data

A file `evaluation_data.json` is provided with 10 test cases. Each test case includes:

- **question**: Natural language query
- **sql**: Ground truth SQL query
- **expected_result**: The actual results from running the query

This can get you started on your evals.

**Note:** Some test cases may return duplicate rows (e.g., playlists with the same name). This reflects the actual database state and is expected.

Example format:

```json
[
  {
    "question": "What are the top 5 best-selling genres by total sales?",
    "sql": "SELECT g.Name, SUM(il.UnitPrice * il.Quantity) as TotalSales FROM Genre g JOIN Track t ON g.GenreId = t.GenreId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY TotalSales DESC LIMIT 5",
    "expected_result": [
      {"Name": "Rock", "TotalSales": 826.65},
      {"Name": "Latin", "TotalSales": 382.14},
      ...
    ]
  }
]
```


### Resources
1. [FireworksAI model library](https://app.fireworks.ai/models)
2. [FireworksAI docs](https://fireworks.ai/docs)
3. [FireworksAI OpenAI SDK](https://fireworks.ai/docs/tools-sdks/openai-compatibility#openai-compatibility)
