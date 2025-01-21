from pathlib import Path
from typing import Annotated, Dict, List
import yaml

import typer
from typer import Option
import polars as pl

from llmpipe import PromptModule, read_data, write_data

def create_label_frequency_table(data: List[Dict]) -> str:
    """Create markdown table of label frequencies."""
    # Count label frequencies
    df = pl.DataFrame(data)
    freq = (
        df
        .select('label')
        .group_by('label')
        .agg(count=pl.col("label").count())
        .sort('count', descending=True)
        .head(500)
    )
    
    # Create markdown table
    table = "| Label | Count |\n|---|---|\n"
    for row in freq.iter_rows():
        table += f"| {row[0]} | {row[1]} |\n"
    return table

def refine_labels(
    prompt_yaml_path: Annotated[str, Option(help="Path to yaml file with prompt config")] = "scripts/ex_anno_refinement_prompt.yaml",
    input_data_path: Annotated[str, Option(help="Path to input dataset")] = "~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl",
    output_data_path: Annotated[str, Option(help="Path to save refined labels")] = "~/data/taskmaster2/refined_labels.jsonl",
    num_proc: Annotated[int, Option(help="Number of processes to use")] = 1,
    model: Annotated[str, Option(help="LiteLLM model identifier")] = "claude-3-5-sonnet-20241022",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Refine annotation labels using an LLM."""
    
    # Expand user paths
    input_data_path = str(Path(input_data_path).expanduser())
    output_data_path = str(Path(output_data_path).expanduser())
    
    # Load config
    with open(prompt_yaml_path) as f:
        config = yaml.safe_load(f)

    config["model"] = model
    config["verbose"] = verbose
    
    # Initialize prompt
    prompt = PromptModule(**config)
    if verbose:
        print(prompt.prompt)
    
    # Load data and create frequency table
    samples = read_data(input_data_path)
    label_table = create_label_frequency_table(samples)
    
    # Run prompt with label table
    results = prompt(labels=label_table, num_proc=num_proc)
    
    # Extract refined labels and save
    write_data(results["refined_labels"], output_data_path)

if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(refine_labels)
    app()
