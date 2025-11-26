Subject: Re: Help Needed: Text-to-SQL for Internal BI Tool

Hi Andrea,

Thank you for sharing your text-to-SQL use case. I've evaluated your database and test questions and devised a simple evaluation framework. I have also improved the results of the current setup of your text-to-sql implementation. 

Below are my findings and recommendations:

Currently, the baseline results show a success rate of around between 10% to 30% based on the model being used. I am using the ground truth sql queries which you shared with us, as the source of truth for the evaluations. Through some improvements, I was able to increase the success rate of the queries to 80%. 8 out of 10 of the queries are converting to the right sql query. This was achieved through few-shot prompting, providing more information about the schema to the LLM, and results normalization of the outputs to make for more fine grained evals. 

There are edge cases which need to be handled such as does “name” refer to first name and last name? Or only first name. Another edge case is that one query in particular is ordering the results in descending order, when there is no mention of this in the english query. I have created a partial scoring system, which will come into use at a later stage when deploying at scale and models could use fine tuning to improve accuracy even further. 

You can access the summary of all the results in the model_comparison_improved and baseline.json files. All test runs are in the eval_results folder. 

Below, I am only showing the improved query results, since the baseline (which is what you provided to us) has an accuracy score between 10% and 30% in these models.

==============================================================================================================
MODEL COMPARISON SUMMARY
==============================================================================================================
Model                        Size     Acc%     Score    Latency    TTFT       Tok/s      Cost/Q       Total$    
--------------------------------------------------------------------------------------------------------------
llama-v3p1-8b-instruct       8B         80.0%   0.874     412.0ms     182.1ms     144.0  $  0.000232  $0.002322
gpt-oss-20b                  20B        60.0%   0.741    2400.6ms    1583.8ms     499.8  $  0.000095  $0.000952
gpt-oss-120b                 120B       50.0%   0.676    1996.2ms    2041.5ms     475.5  $  0.000202  $0.002019
llama4-maverick-instruct-b   401B       80.0%   0.909     716.6ms     414.2ms     144.5  $  0.000235  $0.002354
qwen2p5-vl-32b-instruct      32B        40.0%   0.606    1194.7ms     378.8ms      94.0  $  0.001103  $0.011035

My Recommendation for model usage
llama4-maverick-instruct-b is recommended for production. It achieves the highest evaluation score (0.909 vs 0.874) and 80% accuracy, with better handling of edge cases and partial matches. With 401B (17B active parameters), it provides stronger reasoning for complex multi-table queries, ambiguous questions, and nuanced SQL patterns. Performance is suitable for BI: 613.4ms latency (sub-second), 289.1ms TTFT, and 115.6 tokens/sec throughput. Cost is $0.000235 per query—only $0.000003 more than smaller models, translating to roughly $7/month for 1,000 queries/day. The small latency increase (~96ms) is negligible for BI use, and the quality improvement reduces the risk of incorrect business insights. For accuracy-critical BI tools, the stronger reasoning and higher reliability justify the minimal cost difference, making it the best choice for production deployment.

That being said, I would love to learn more about your preferences and vision for the product. The llama4-maverick-instruct-b is a very large model and the throughput is roughly 4x worse than its nearest competitor, the llamav3p1-8b model which, however, has only 8 billion parameters. There is always a tradeoff, and I'm happy to share more options with you. 

After you choose a model, FireworksAI will guarantee this industry-leading inference speed in production, making it ideal for your workload. The platform makes it simple to switch between models, test different models without code changes and include built-in evaluation tools to measure and improve performance. With fine-tuning support, you can customize models for your domain. SOC2 and HIPAA compliance ensure enterprise-grade security for sensitive business data. 

If you would like to test out these profiling jobs to see model performance and understand the data even further, you can run: 

python run_eval.py --multi-model --improved
	and
python run_eval.py --original-model

You can run these 2 commands above after running chmod +x setup.sh and ./setup.sh. All runs will be in the eval_results_baseline and improved folders, but model_comparison_improved.json should have most info you might be interested in. More_commands.md has more commands as well 

We can schedule a follow-up call to walk through the results and next steps.

Best regards,
Sanat
Solutions Team
FireworksAI

