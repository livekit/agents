"""Atomic process observations shared by SDK and third-party applications."""

import json
from pathlib import Path


def write_marker(path: Path, value: object) -> None:
    """Publish a complete JSON value; a reader never observes a partial file."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value))
    partial.rename(path)
