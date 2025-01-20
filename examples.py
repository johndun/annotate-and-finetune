# Use `typer` to define command line scripts:

from typing import Annotated

import typer
from typer import Option


def yaml_module(
        prompt_path: Annotated[str, Option(help="Path to a yaml file containing the prompt configuration")] = None,
        input_data_path: Annotated[str, Option(help="Dataset to run prompt on")] = None,
        output_data_path: Annotated[str, Option(help="Path to save processed dataset")] = None,
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


# Use read_data and write_data functions from llmpipe for data io

def read_data(path: str, as_df: bool = False, **kwargs) -> List[Dict]:
    """Reads tab separated (with header, .txt) or json lines (.jsonl) data from disk.

    Args:
        path: Path to the data file
        as_df: If true, return a polars dataframe
        kwargs: Arguments based to polars read data function

    Returns:
        List[Dict]: Data records/samples as a list of dictionaries
    """
    ...

def write_data(samples: List[Dict], path: str):
    """Writes data as tab separated (with header, .txt) or json lines (.jsonl) to disk.

    Args:
        samples: Data records/samples as a list of dictionaries
        path: Path to the data file
    """
    ...