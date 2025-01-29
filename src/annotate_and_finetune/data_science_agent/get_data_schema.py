import json
import random
from typing import Any, Dict, List, Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2


def get_data_schema(
    data_sample_path: Annotated[str, Option(help="Path to dataset samples")],
    output_path: Annotated[str, Option(help="Path to save the schema")] = None,
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
):
    """Print random samples from a dataset with truncated long values."""
    # Read the data
    with open(data_sample_path, "r") as f:
        data_sample = f.read()

    module = PromptModule2(
        task="Generate a schema for a dataset as a markdown table. Columns should include name, type, and description.",
        inputs=[
            Input("data_samples", "A small set of examples from a dataset"),
        ],
        outputs=[
            Output("thinking", "Begin by thinking step by step"),
            Output("data_schema", "Data schema")
        ],
        model=model
    )
    print(module.prompt)
    response = module(data_samples=data_sample)
    print(response["data_schema"])

    # Save schema if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(response["data_schema"])
        print(f"\nSaved schema to {output_path}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(get_data_schema)
    app()
