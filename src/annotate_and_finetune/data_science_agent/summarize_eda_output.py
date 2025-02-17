import json
import random
from typing import Any, Dict, List, Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2


def summarize_eda_output(
    repo_path: Annotated[str, Option(help="Working directory")],
    script_name: Annotated[str, Option(help="Script name")] = None,
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Draft document text using EDA results."""
    # Read the data
    with open(f"{repo_path}/logs/{script_name}.log", "r") as f:
        eda_results = f.read()
    with open(f"{repo_path}/data_schema.md", "r") as f:
        data_schema = f.read()

    module = PromptModule2(
        task="Summarize outputs of an exploratory data analysis script. Use markdown headers for organization. Incorporate all of the relevant information from the EDA results. Focus on coverage. Content will be revised and consolidated into a final document. Include markdown tables where appropriate. Include methodology and explainers for any statistical techniques used. Include a section on insights and takeaways.",
        inputs=[
            Input("data_schema", "The data schema"),
            Input("eda_results", "Exploratory data analysis results")

        ],
        outputs=[
            Output("thinking", "Begin by thinking step by step"),
            Output("document", "A document containing a detailed, comprehensive summary. No title.")
        ],
        model=model,
        verbose=verbose
    )
    if verbose:
        print(module.prompt)
    response = module(data_schema=data_schema, eda_results=eda_results)

    output_path = f"{repo_path}/notes/{script_name}.md"
    with open(output_path, "w") as f:
        f.write(response["document"])
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(summarize_eda_output)
    app()
