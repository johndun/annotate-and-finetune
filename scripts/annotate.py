from pathlib import Path
from typing import Annotated
import random
import yaml

import typer
from typer import Option
import polars as pl

from llmpipe import PromptModule, read_data, write_data


def annotate(
    prompt_yaml_path: Annotated[str, Option(help="Path to yaml file with prompt config")] = "scripts/ex_annotation_prompt.yaml",
    input_data_path: Annotated[str, Option(help="Path to input dataset")] = "~/data/taskmaster2/taskmaster2_dialogs.jsonl",
    output_data_path: Annotated[str, Option(help="Path to save annotated dataset")] = "~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl",
    n_samples: Annotated[int, Option(help="Number of random samples to process")] = None,
    num_proc: Annotated[int, Option(help="Number of processes to use")] = 1,
    model: Annotated[str, Option(help="LiteLLM model identifier")] = "claude-3-5-sonnet-20241022",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Run a prompt from a yaml config file on a dataset."""
    
    # Expand user paths
    input_data_path = str(Path(input_data_path).expanduser())
    output_data_path = str(Path(output_data_path).expanduser())
    
    # Load config
    with open(prompt_yaml_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize prompt
    prompt = PromptModule(
        inputs=config["inputs"],
        outputs=config["outputs"],
        model=model,
        verbose=verbose
    )
    
    # Load data
    samples = read_data(input_data_path)
    
    # Sample if requested
    if n_samples is not None:
        n_samples = min(n_samples, len(samples))
        samples = random.sample(samples, n_samples)

    data = pl.from_dicts(samples).to_dict(as_series=False)
    
    # Run prompt
    results = prompt(**data, num_proc=num_proc)
    
    # Save results 
    write_data(results, output_data_path)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(annotate)
    app()
