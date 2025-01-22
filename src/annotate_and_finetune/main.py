from pathlib import Path
from typing import Annotated
import yaml
import json
import polars as pl
from itertools import chain
import typer
from typer import Option

from llmpipe import read_data
from annotate_and_finetune.annotate import run_annotation
from annotate_and_finetune.finetune import run_finetuning
from annotate_and_finetune.split_data import split_data


def load_config(config_path: str) -> dict:
    """Load and parse YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: Annotated[str, Option(help="Path to YAML config file")] = None,
    num_proc: Annotated[int, Option(help="Number of processes for annotation")] = 2,
    verbose: Annotated[bool, Option(help="Enable verbose output")] = False,
):
    """Run the full annotation and fine-tuning pipeline.
    
    Args:
        config_path: Path to YAML config file containing pipeline settings
        num_proc: Number of processes for parallel annotation
        verbose: Enable verbose output
    """
    print("Loading config file...")
    config = load_config(config_path)
    
    # Extract config values
    model = config.get("model", "anthropic/claude-3-sonnet-20240229")
    allowed_labels = config["allowed_labels"]
    task = config["task"]
    details = config.get("details")
    context_col = config["context_col"]
    context_description = config["context_description"]
    id_col = config.get("id_col", "id")
    data_path = str(Path(config["data_path"]).expanduser())
    model_path = str(Path(config["model_path"]).expanduser())
    val_test_prop = config.get("val_test_prop", 0.2)
    output_path = str(Path(config["output_path"]).expanduser())
    
    # Extract training parameters from config
    n_samples = config.get("n_samples", 10)
    annotation_batch_size = config.get("annotation_batch_size", 10)
    num_epochs = config.get("num_epochs", 1)
    learning_rate = config.get("learning_rate", 0.00001)
    batch_size = config.get("batch_size", 8)

    print(f"Loading data from {data_path}...")
    samples_df = read_data(data_path, as_df=True).with_row_index(id_col)
    if "label" in samples_df.columns:
        samples_df = samples_df.rename({"label": "gt_label"})
    samples = samples_df.to_dicts()

    # Configure annotation
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

    print("\nStarting annotation phase...")
    print(f"Using model: {model}")
    print(f"Annotation batch size: {annotation_batch_size}")
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

    print("\nSplitting data into train/val/test sets...")
    train_samples, val_samples, test_samples = split_data(
        annotated_samples,
        [1 - 2 * val_test_prop, val_test_prop, val_test_prop]
    )

    print("\nStarting fine-tuning phase...")
    print(f"Using model: {model_path}")
    print(f"Training for {num_epochs} epochs")
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


def main():
    """CLI entry point."""
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(run_pipeline)
    app()


if __name__ == "__main__":
    main()
