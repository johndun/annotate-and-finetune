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
    """Convert a dialog string into a list of cumulative turn records.
    Each turn contains a USER message followed by an ASSISTANT response."""
    turns = []
    dialog_lines = dialog.split('\n')
    
    # Process lines in pairs of USER + ASSISTANT messages
    for i in range(0, len(dialog_lines)-1, 2):
        # Skip if we don't have both USER and ASSISTANT messages
        if i+1 >= len(dialog_lines):
            break
            
        # Verify we have USER then ASSISTANT pattern
        current_line = dialog_lines[i]
        next_line = dialog_lines[i+1]
        if not (current_line.startswith('USER:') and next_line.startswith('ASSISTANT:')):
            continue
            
        # Build context from all previous turns plus current pair
        context = '\n'.join(dialog_lines[:i+2])
        turn = {
            'context': context,
            'label': label,
            'turn_index': i//2,  # Index counts pairs rather than individual messages
            'user_message': current_line,
            'assistant_message': next_line
        }
        turns.append(turn)
    
    return turns


def prepare_taskmaster2_turn_dataset(
    input_file: Annotated[
        Path,
        Option(help="Input jsonl file from prepare_taskmaster2_dialog_dataset.py")
    ] = Path.home() / "data/taskmaster2/taskmaster2_dialogs.jsonl",
    output_path: Annotated[
        Path,
        Option(help="Output path for the processed jsonl file")
    ] = Path.home() / "data/taskmaster2/taskmaster2_turns.jsonl",
):
    """Process Taskmaster-2 dialog dataset into turn-level records with cumulative context."""
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each dialog
    with output_path.open('w') as outf:
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

    print(f"Processed Taskmaster-2 turn dataset saved to: {output_path}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(prepare_taskmaster2_turn_dataset)
    app()
