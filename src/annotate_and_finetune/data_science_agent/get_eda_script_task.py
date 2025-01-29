from typing import Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2


EDA_SCRIPT_TASK = """\
Generate concise exploratory data analysis (EDA) script requirements for a data science task. EDA scripts should only print outputs (to be used to inform future analyses and/or write research summary documents). Script should input a single dataset (schema defined below) and may have additional command line arguments.

Organize the requirements to facilitate and simplify implementation.

Example:

<example>
Write a cli script that prints random samples from a dataset. Needs to handle cases where strings and list types are extremely long by truncating to a reasonable maximum number of items. Samples should be printed as json-like.

Inputs:
- data_path: str
- n_samples: int = 5
</example>
"""


def get_eda_script_task(
    schema_path: Annotated[str, Option(help="Path to the data schema")],
    task: Annotated[str, Option(help="The EDA task to generate detailed requirements for")],
    output_path: Annotated[str, Option(help="Path to save the requirements")] = None,
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
):
    """Generate detailed requirements for a data science EDA task using an LLM."""
    # Read the schema
    with open(schema_path, "r") as f:
        schema = f.read()

    module = PromptModule2(
        task=EDA_SCRIPT_TASK,
        inputs=[
            Input("schema", "The data schema"),
            Input("task", "A data science task"),
        ],
        outputs=[
            Output("thinking", "Begin by thinking step by step"),
            Output("eda_script_task", "Task containing requirements for an exploratory data analysis script")
        ],
        model=model
    )

    response = module(schema=schema, task=task)
    print(response["eda_script_task"])

    # Save requirements if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(response["eda_script_task"])
        print(f"\nSaved requirements to {output_path}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(get_eda_script_task)
    app()
