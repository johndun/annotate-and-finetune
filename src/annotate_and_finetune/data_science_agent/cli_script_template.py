from typing import Annotated

import typer
from typer import Option


# All inputs should be Option type, even if no default is provided
def yaml_module(
        prompt_path: Annotated[str, Option(help="Path to a yaml file containing the prompt configuration")],
        input_data_path: Annotated[str, Option(help="Dataset to run prompt on")],
        output_data_path: Annotated[str, Option(help="Path to save processed dataset")],
        num_proc: Annotated[int, Option(help="Number of processes to use is dataset mode")] = 1,
        model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022",
        verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Execute a prompt from a yaml file on a dataset."""
    ...


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals = False)
    app.command()(yaml_prompt)
    app()
