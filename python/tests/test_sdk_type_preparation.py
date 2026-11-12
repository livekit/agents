"""SDK typing preparation protects directories it does not own."""

import runpy
from pathlib import Path

import pytest

prepare = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / ".github" / "prepare_sdk_types.py")
)["prepare"]

pytestmark = pytest.mark.unit


def test_sdk_type_preparation_refuses_to_replace_an_unowned_directory(tmp_path: Path) -> None:
    existing = tmp_path / "types"
    existing.mkdir()
    artifact = existing / "important.txt"
    artifact.write_text("keep")

    with pytest.raises(ValueError, match="Refusing to replace"):
        prepare(existing)

    assert artifact.read_text() == "keep"
