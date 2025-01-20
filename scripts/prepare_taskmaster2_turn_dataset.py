"""Prepare Taskmaster-2 turn dataset.

Creates a dataset with one row per dialog turn, including cumulative context
from all previous turns in the conversation.
"""

from pathlib import Path
from typing import Annotated, List, Dict
import json

import typer
from typer import Option


def process_dialog_to_turns(dialog: str, label: str) -> List[Dict]:
    """Convert a dialog string into a list of cumulative turn records."""
    turns = []
    dialog_lines = dialog.split('\n')
    
    for i in range(len(dialog_lines)):
        # Build context from all previous turns plus current
        context = '\n'.join(dialog_lines[:i+1])
        turn = {
            'context': context,
            'label': label,
            'turn_index': i
        }
        turns.append(turn)
    
    return turns


def prepare_taskmaster2_turn_dataset(
    input_file: Annotated[
        Path,
        Option(help="Input jsonl file from prepare_taskmaster2_dialog_dataset.py")
    ] = Path.home() / "data/taskmaster2/taskmaster2_dialog.jsonl",
    output_dir: Annotated[
        Path,
        Option(help="Output directory for processed files")
    ] = Path.home() / "data/taskmaster2",
):
    """Process Taskmaster-2 dialog dataset into turn-level records with cumulative context."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "taskmaster2_turns.jsonl"

    # Process each dialog
    with output_file.open('w') as outf:
        for line in open(input_file):
            dialog_data = json.loads(line)
            
            # Convert dialog into turn records
            turns = process_dialog_to_turns(
                dialog_data['dialog'],
                dialog_data['label']
            )
            
            # Write each turn as a JSON line
            for turn in turns:
                outf.write(json.dumps(turn) + '\n')

    print(f"Processed Taskmaster-2 turn dataset saved to: {output_file}")


if __name__ == "__main__":
    typer.run(prepare_taskmaster2_turn_dataset)
