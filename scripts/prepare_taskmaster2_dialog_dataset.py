"""Prepare Taskmaster-2 dataset for training.

Downloads the dataset from:
https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020
"""

from pathlib import Path
from typing import Annotated, List, Dict
import json

import typer
from typer import Option


def format_conversation(utterances: List[Dict]) -> str:
    """Convert utterances into a string with USER and ASSISTANT turns.
    Skips the first utterance if it's from ASSISTANT."""
    conversation = []
    start_idx = 1 if utterances and utterances[0]["speaker"] == "ASSISTANT" else 0
    
    for utterance in utterances[start_idx:]:
        speaker = utterance["speaker"]
        text = utterance["text"]
        conversation.append(f"{speaker}: {text}")
    return "\n".join(conversation)


def prepare_taskmaster2_dialog_dataset(
    input_dir: Annotated[
        Path,
        Option(help="Directory containing TM-2-2020 JSON files")
    ] = Path.home() / "Taskmaster/TM-2-2020/data",
    output_path: Annotated[
        Path,
        Option(help="Output path for the processed jsonl file")
    ] = Path.home() / "data/taskmaster2/taskmaster2_dialogs.jsonl",
):
    """Process Taskmaster-2 files into a single jsonlines file.
    
    Each line will contain a dictionary with a string representation of the dialog the label.
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each JSON file
    with output_path.open('w') as outf:
        for json_file in input_dir.glob('*.json'):
            label = json_file.stem  # e.g. "flights" from "flights.json"
            
            # Read and process conversations
            data = json.loads(json_file.read_text())
            for conv in data:
                # Convert dialog to string format and write as JSON line
                dialog_str = format_conversation(conv['utterances'])
                outf.write(json.dumps({
                    'dialog': dialog_str,
                    'label': label
                }) + '\n')

    print(f"Processed Taskmaster-2 dialog dataset saved to: {output_path}")


if __name__ == "__main__":
    typer.run(prepare_taskmaster2_dialog_dataset)
