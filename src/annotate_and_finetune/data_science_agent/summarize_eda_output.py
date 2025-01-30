import json
import random
from typing import Any, Dict, List, Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2


def summarize_eda_output(
    input_path: Annotated[str, Option(help="Path to eda output")],
    output_path: Annotated[str, Option(help="Path to save the schema")] = None,
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Draft document text using EDA results."""
    # Read the data
    with open(input_path, "r") as f:
        eda_results = f.read()

    module = PromptModule2(
        task="Summarize outputs of an exploratory data analysis script. Use markdown headers for organization. Incorporate all of the relevant information from the EDA results. Focus on coverage. Content will be revised and consolidated into a final document. Include markdown tables where appropriate. Include methodology and explainers for any statistical techniques used. Include a section on insights and takeaways.",
        inputs=[
            Input("eda_results", "Exploratory data analysis results"),
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
    response = module(eda_results=eda_results)

    # Save if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(response["document"])
        print(f"\nSaved to {output_path}")
    else:
        print(response["document"])


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(summarize_eda_output)
    app()
