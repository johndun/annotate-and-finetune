## Script to initialize a git repo

aider --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 --read cli_script_template.py src/data_science_agent/initialize_repo.py

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

aider --no-show-model-warnings --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 --read data_io_template.py --read cli_script_template.py src/data_science_agent/get_data_sample.py

Write a cli script that prints random samples from a dataset. Needs to handle cases where strings and list types are extremely long by truncating to a reasonable maximum number of items. Samples should be printed as json-like.

Inputs:

- data_path: str
- n_samples: int = 5



# Input
- dataset

# Initialize a repo
python src/data_science_agent/initialize_repo.py --repo-path ~/test_repo

# Generate a file in the repo containing 5 random records from the data
python src/data_science_agent/get_data_sample.py --data-path dummy_data.jsonl --output-path ~/test_repo/sample_data.md

# Generate a file in the repo containing a data schema
python src/data_science_agent/get_data_schema.py \
    --data-sample-path ~/test_repo/sample_data.md \
    --output-path ~/test_repo/data_schema.md \
    --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0

# Commit all changes in the repo
python src/data_science_agent/git_commit.py \
    --repo-path ~/test_repo \
    --commit-message "Initial commit"

# Generate a task
python src/data_science_agent/get_eda_script_task.py \
    --schema-path ~/test_repo/data_schema.md \
    --task "Generate univariate summary statistics for the dataset" \
    --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0

# Input
- EDA task

# Generate subtasks
  - script
  - script execution



## Prompt for task translation

aider --no-show-model-warnings --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 --read src/data_science_agent/get_data_schema.py src/data_science_agent/get_eda_script_task.py

Write a cli script that will generate detailed script requirements for a data science task using an LLM. The below YAML file describes the inputs to PromptModule2.

Your script should have the following inputs:
- schema_path: str  # Path to the data schema
- task: str  # The EDA task to generate detailed requirements for

```yaml
task: |-
  Generate exploratory data analysis (EDA) script requirements for a data science task. EDA scripts should only print outputs (to be used to inform future analyses and/or write research summary documents). Script should input a single dataset (schema defined below) and may have additional command line arguments.

  Example:

  <example>
  Write a cli script that prints random samples from a dataset. Needs to handle cases where strings and list types are extremely long by truncating to a reasonable maximum number of items. Samples should be printed as json-like.

  Inputs:
  - data_path: str
  - n_samples: int = 5
  </example>
inputs:
  - name: schema
    description: The data schema
  - name: task
    description: A data science task
outputs:
  - name: thinking
    description: Begin by thinking step by step
  - name: eda_script_task
    description: Task containing requirements for an exploratory data analysis script
```