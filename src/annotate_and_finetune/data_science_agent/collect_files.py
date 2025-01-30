import os
from pathlib import Path
from typing import Dict


def collect_files(directory_path: str) -> Dict[str, str]:
    """
    Collect contents of all text files in the given directory path.

    Args:
        directory_path: Path to directory containing text files

    Returns:
        Dictionary with filenames (without extensions) as keys and file contents as values
    """
    contents = {}
    path = Path(directory_path)

    if not path.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                # Read file content
                content = file_path.read_text(encoding='utf-8')

                # Use stem (filename without extension) as key
                key = file_path.stem

                contents[key] = content

            except UnicodeDecodeError:
                # Skip files that can't be read as text
                continue

    return contents