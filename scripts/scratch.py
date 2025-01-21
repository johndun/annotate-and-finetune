import yaml
import json
import polars as pl
from itertools import chain

from llmpipe import (
    Input, Output, JsonlinesOutput,
    PromptModule, RevisorModule,
    read_data, write_data
)
from annotate_and_finetune import run_annotation, run_finetuning, split_data


model = "anthropic/claude-3-5-sonnet-20241022"
verbose = False

num_proc = 2
n_samples = 10
annotation_batch_size = 10

allowed_labels = [
    {"label": "SPORTS", "description": "A dialog related to sports"},
    {"label": "OTHER", "description": "Everything else"}
]
task = "Label dialogs."
details = ""
context_col = "dialogs"
context_description = "A table of dialogs between a user and an assistant"
id_col = "id"
data_path = "~/data/taskmaster2/taskmaster2_dialogs.jsonl"
model_path = "/Users/johndunavent/models/roberta-base"
val_test_prop = 0.2
output_path = "~/models/deleteme"



samples_df = read_data(data_path, as_df=True).with_row_index(id_col)
if "label" in samples_df.columns:
    samples_df = samples_df.rename({"label": "gt_label"})
samples = samples_df.to_dicts()


single_annotation_config = f"""\
task: {task}
details: {details}
outputs:
  - name: thinking
    description: Begin by thinking step by step
  - name: label
    description: A label selected from `allowed_labels`
    inputs:
      - name: {context_col}
        description: {context_description}
      - name: allowed_labels
        description: The set of allowed labels
"""

batch_annotation_config = f"""\
task: {task}
details: {details}
inputs:
  - name: {context_col}
    description: {context_description}
  - name: allowed_labels
    description: The set of allowed labels
outputs:
  - name: thinking
    description: Begin by thinking step by step
  - name: labels
    type: jsonlines
    description: A table with annotated labels
    fields:
      - name: id
        description: An id from `{context_col}`
      - name: label
        description: A label selected from `allowed_labels`
"""

annotation_config = yaml.safe_load(
    single_annotation_config
    if annotation_batch_size == 1 else
    batch_annotation_config
)
# annotator = PromptModule(**annotation_config)
# print(annotator.prompt)

if annotation_batch_size == 1:
    annotated_samples = run_annotation(
        config=annotation_config,
        samples=samples,
        n_samples=n_samples,
        num_proc=num_proc,
        model=model,
        verbose=verbose,
        allowed_labels=allowed_labels
    )
    annotated_samples = (
        pl.from_dicts(annotated_samples)
        .drop("thinking", "allowed_labels")
        .to_dicts()
    )
else:
    batches = []
    for i in range(0, len(samples), annotation_batch_size):
        batch = [{k: x[k] for k in ("id", "dialog")} for x in samples[i: i + annotation_batch_size]]
        batches.append("\n".join([json.dumps(x) for x in batch]))

    batched_samples = [{context_col: x} for x in batches]

    batch_annotated_samples = run_annotation(
        config=annotation_config,
        samples=batched_samples,
        n_samples=n_samples,
        num_proc=num_proc,
        model=model,
        verbose=verbose,
        allowed_labels=allowed_labels
    )

    labels = list(chain(*[x["labels"] for x in batch_annotated_samples if x["labels"] is not None]))
    annotated_samples = pl.from_dicts(samples).join(
        pl.from_dicts(labels).with_columns(id=pl.col("id").cast(pl.UInt32)),
        on="id", how="inner"
    ).to_dicts()


train_samples, val_samples, test_samples = split_data(
    annotated_samples,
    [1 - 2 * val_test_prop, val_test_prop, val_test_prop]
)

run_finetuning(
    train_data=train_samples,
    val_data=val_samples,
    test_data=test_samples,
    model_path=model_path,
    output_path=output_path,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
)