import yaml
import os

import polars as pl

from llmpipe import (
    Input, Output, JsonlinesOutput, TabularOutput,
    PromptModule, RevisorModule,
    read_data, write_data, load_json_files
)
from annotate_and_finetune import run_annotation


os.environ["AWS_REGION_NAME"] = "us-east-1"
model = "..."
verbose = False
num_proc = 4
n_samples = 20

allowed_labels = [
    {"label": "1", "description": "The category of interest"},
    {"label": "2", "description": "Everything else"}
]
task = "Label requests."
context_col = "request"
context_description = "A request from a user"
id_col = "id"

data_path = "..."

batch_size = 10


samples = load_json_files(data_path)
samples = pl.from_dicts(samples).with_row_index("id").to_dicts()
print(len(samples))
print(samples[0])



annotation_prompt = f"""\
task: {task}
inputs:
  - name: {context_col}
    description: {context_description}
  - name: allowed_labels
    description: The set of allowed labels
outputs:
  - name: thinking
    description: Begin by thinking step by step
  - name: label
    description: A label selected from `allowed_labels`
    evaluations:
      - type: llm
        value: Exactly matches one of `allowed_labels`
"""

annotation_config = yaml.safe_load(annotation_prompt)
annotator = PromptModule(**annotation_config)
print(annotator.prompt)

annotated_samples = run_annotation(
    config=annotation_config,
    samples=samples,
    n_samples=n_samples,
    num_proc=num_proc,
    model=model,
    verbose=verbose,
    allowed_labels=allowed_labels
)
annotated_samples = pl.from_dicts(annotated_samples).drop("thinking", "allowed_labels").to_dicts()
annotated_samples[-1]



batch_annotation_prompt = f"""\
task: Label Alexa user requests.
inputs:
  - name: {context_col}
    description: {context_description}
  - name: allowed_labels
    description: The set of allowed labels
outputs:
  - name: thinking
    description: Begin by thinking step by step
  - name: labels
    type: tabular
    description: A table of labeled requests
    fields:
      - name: id
        description: The id from `{context_col}`
      - name: label
        description: A label selected from `allowed_labels`
        evaluations:
          - type: llm
            value: Exactly matches one of `allowed_labels`
"""
batch_annotation_config = yaml.safe_load(batch_annotation_prompt)
batch_annotator = PromptModule(**batch_annotation_config)
print(batch_annotator.prompt)






# Get batched requests
batches = []
for i in range(0, len(samples), batch_size):
    batch = [
        f"| id | {context_col} |",
        f"|----|---------|"
    ]
    batch.extend([f"| {x['id']} | {x[context_col]} |" for x in samples[i:i + batch_size]])
    batches.append("\n".join(batch))

batched_samples = [{"requests": x} for x in batches]

batch_annotated_samples = run_annotation(
    config=batch_annotation_config,
    samples=batched_samples,
    n_samples=n_samples,
    num_proc=num_proc,
    model=model,
    verbose=verbose,
    allowed_labels=allowed_labels
)

labels = chain(*batch_annotated_samples["labels"])
labels = list(chain(*[x["labels"] for x in batch_annotated_samples]))
df = df.join(
    pl.from_dicts(labels).with_columns(id=pl.col("id").cast(pl.UInt32)),
    on="id", how="inner"
)