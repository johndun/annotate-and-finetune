## Script to initialize a git repo

aider --no-show-model-warnings --model claude-3-5-sonnet-20241022 src/annotate_and_finetune/data_science_agent/initialize_repo.py

aider --no-show-model-warnings --model claude-3-5-sonnet-20241022 \
    --read src/annotate_and_finetune/data_science_agent/cli_script_template.py \
    --read src/annotate_and_finetune/data_science_agent/data_io_template.py \
    src/annotate_and_finetune/data_science_agent/initialize_repo.py

Write a cli script that initializes an empty git repo. If `output_dir` exists, it should be emptied. If not, create it. Then, initialize an empty git repo.

Inputs:

- output_dir: str = "~/test_repo"


## Script to make a commit in a git repo

aider --no-show-model-warnings --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 --read src/data_science_agent/initialize_repo.py src/data_science_agent/git_commit.py

Write a cli script that commits all current changes in a git repo.

Inputs:

- commit_message: str
- repo_path: str = "~/test_repo"

## Script to generate a data sample

aider --no-show-model-warnings --model claude-3-5-sonnet-20241022-v2 src/annotate_and_finetune/data_science_agent/get_data_sample.py

Write a cli script that prints random samples from a dataset. Needs to handle cases where strings and list types are extremely long by truncating to a reasonable maximum number of items. Samples should be printed as json-like.

Inputs:

- data_path: str
- n_samples: int = 5



# Input
- dataset

# Initialize a repo
python -m annotate_and_finetune.data_science_agent.initialize_repo --repo-path ~/test_repo

# Generate a file in the repo containing 3 random records from the data
python -m annotate_and_finetune.data_science_agent.get_data_sample --data-path ~/data/imdb-train.jsonl --output-path ~/test_repo/sample_data.md --n-samples 3

# Generate a file in the repo containing a data schema
python -m annotate_and_finetune.data_science_agent.generate_data_schema \
    --data-sample-path ~/test_repo/sample_data.md \
    --output-path ~/test_repo/data_schema.md \
    --model claude-3-5-sonnet-20241022

# Commit all changes in the repo
python -m annotate_and_finetune.data_science_agent.git_commit \
    --repo-path ~/test_repo \
    --commit-message "Initial commit"

# Run an EDA task
python -m annotate_and_finetune.data_science_agent.generate_eda_script \
    --repo-path ~/test_repo \
    --data-path ~/data/imdb-train.jsonl \
    --task "Generate univariate summary statistics. Include missing value counts. Include distinct value counts. Include a table of frequency counts for fields with fewer than 20 distinct values." \
    --script-name univariate_summaries.py \
    --model claude-3-5-sonnet-20241022

# Run an EDA task
python -m annotate_and_finetune.data_science_agent.generate_eda_script \
    --repo-path ~/test_repo \
    --data-path ~/data/imdb-train.jsonl \
    --task "Analyze movie review sentiment patterns with the following steps:\n\n1. Text preprocessing:\n   - Remove common stop words (excluding basic negation words like 'not')\n   - Convert to lowercase\n   - Remove punctuation\n\n2. For both positive and negative reviews:\n   - Calculate average review length\n   - Find most common meaningful words (excluding stop words)\n   - Calculate relative word frequencies (percentage of total words)\n   - Identify distinctive words (words that appear significantly more in one sentiment vs the other)\n\n3. Create comparison metrics:\n   - Top 20 words unique to each sentiment\n   - Chi-square test for word frequency differences\n   - Word frequency ratios between positive and negative reviews\n\nUse pandas for data manipulation and numpy/scipy for statistical calculations." \
    --script-name word_distributions.py \
    --model claude-3-5-sonnet-20241022

# Summarize EDA logs
python -m annotate_and_finetune.data_science_agent.summarize_eda_output \
    --input-path ~/test_repo/data_schema.md \
    --model claude-3-5-sonnet-20241022
python -m annotate_and_finetune.data_science_agent.summarize_eda_output \
    --input-path ~/test_repo/sample_data.md \
    --model claude-3-5-sonnet-20241022
python -m annotate_and_finetune.data_science_agent.summarize_eda_output \
    --input-path ~/test_repo/logs/univariate_summaries.log \
    --model claude-3-5-sonnet-20241022
python -m annotate_and_finetune.data_science_agent.summarize_eda_output \
    --input-path ~/test_repo/logs/word_distributions.log \
    --model claude-3-5-sonnet-20241022

python -m annotate_and_finetune.data_science_agent.revise_eda_task \
    --task "Create a table comparing the most common words across the different label values" \
    --input-path ~/test_repo/logs/word_distributions.log \
    --model claude-3-5-sonnet-20241022
