import os
import shutil
import subprocess
from typing import Annotated

import typer
from typer import Option

from llmpipe import Input, Output
from llmpipe.prompt_module2 import PromptModule2


def run_aider(
    message_file: Annotated[str, Option(help="Message file to send to aider")],
    working_dir: Annotated[str, Option(help="Working directory to run the command in")],
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2"
):
    """
    Executes a command line script and returns its output.

    Args:
        message_file (str): Message to send to aider.
        working_dir (str): The directory to run the command in.
    """

    command_str = f"""aider --no-show-model-warnings --stream --model {model} --message-file {message_file} --yes --auto-commits"""
    try:
        subprocess.run(
            command_str,
            shell=True,
            capture_output=False,
            text=True,
            cwd=working_dir
        )
        return None

    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit status {e.returncode}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


AIDER_MESSAGE_TEMPLATE = """\
{task}

Write an exploratory data analysis (EDA) python script to complete the task. EDA scripts should only print outputs (to be used to inform future analyses and/or write research summary documents). Script should input a single dataset (schema defined below) and may have additional command line arguments.

After completing execute the script, execute the script using the inputs below and write stdout to a file (with the same filename as the script) in the logs/ directory.

Data path: {data_path}

Data schema:

{schema}

Sample data:

{data_samples}
"""


def run_aider_task(
    task: Annotated[str, Option(help="Task")],
    data_path: Annotated[str, Option(help="Dataset path")],
    repo_path: Annotated[str, Option(help="Working directory")],
    model: Annotated[str, Option(help="A LiteLLM model identifier")] = "claude-3-5-sonnet-20241022-v2"
):
    """Generate detailed requirements for a data science EDA task using an LLM."""
    # Read the schema
    with open(f"{repo_path}/data_schema.md", "r") as f:
        schema = f.read()

    # Read the data samples
    with open(f"{repo_path}/sample_data.md", "r") as f:
        data_samples = f.read()

    message = AIDER_MESSAGE_TEMPLATE.format(
        task=task,
        data_path=data_path,
        schema=schema,
        data_samples=data_samples
    )
    message_file = f"{repo_path}/prompt.txt"
    with open(message_file, "w") as f:
        f.write(message)

    run_aider(message_file=message_file, working_dir=repo_path, model=model)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(run_aider_task)
    app()
