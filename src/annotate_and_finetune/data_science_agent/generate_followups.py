import json
import random
from typing import Any, Dict, List, Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2

from annotate_and_finetune.data_science_agent.collect_files import collect_files


def generate_followups(
    repo_path: Annotated[str, Option(help="Working directory")],
    output_path: Annotated[str, Option(help="Path to save the outputs")] = None,
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Draft followups using EDA results."""
    # Read the schema
    with open(f"{repo_path}/data_schema.md", "r") as f:
        data_schema = f.read()

    # Read the data samples
    with open(f"{repo_path}/sample_data.md", "r") as f:
        data_samples = f.read()

    logs = collect_files(f"{repo_path}/notes")
    txt = []
    for k, v in logs.items():
        txt.append(f"<{k}>\n{v}\n</{k}>")
    eda_results = "\n\n".join(txt)

    module = PromptModule2(
        task="Propose up to 3 follow up exploratory data analyses based on current results. Limit analyses to those that can be conducted with a single script using pandas, scipy, nltk, and numpy. Only text-based summaries for now (so no graphs).",
        inputs=[
            Input("data_samples", "A small set of examples from a dataset"),
            Output("data_schema", "The data schema as a markdown table"),
            Input("eda_results", "Current data analysis results"),
        ],
        outputs=[
            Output("thinking", "Begin by thinking step by step"),
            Output("followups", "One or more follow up exploratory data analysis tasks")
        ],
        model=model,
        verbose=verbose
    )
    if verbose:
        print(module.prompt)
    response = module(
        data_samples=data_samples,
        data_schema=data_schema,
        eda_results=eda_results
    )

    # Save if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(response["followups"])
        print(f"\nSaved to {output_path}")
    else:
        print(response["followups"])

if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(generate_followups)
    app()
