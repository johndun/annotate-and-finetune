from typing import Annotated
from pathlib import Path
import typer
from typer import Option

from annotate_and_finetune.data_science_agent.initialize_repo import initialize_repo
from annotate_and_finetune.data_science_agent.get_data_sample import get_data_sample
from annotate_and_finetune.data_science_agent.generate_data_schema import generate_data_schema
from annotate_and_finetune.data_science_agent.git_commit import git_commit


def initialize_project(
    repo_path: Annotated[str, Option(help="Directory to initialize project in")],
    data_path: Annotated[str, Option(help="Dataset path")],
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2",
    verbose: Annotated[bool, Option(help="Stream output to stdout")] = False
):
    """Initialize a project."""
    initialize_repo(
        repo_path=repo_path
    )
    get_data_sample(
        data_path=data_path,
        n_samples=3,
        output_path=f"{repo_path}/sample_data.md"
    )
    generate_data_schema(
        data_sample_path=f"{repo_path}/sample_data.md",
        output_path=f"{repo_path}/data_schema.md",
        model=model,
        verbose=verbose
    )
    git_commit(
        repo_path=repo_path,
        commit_message="Initial commit"
    )



if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(initialize_project)
    app()
