from __future__ import annotations

import pytest

from livekit.agents.beta.workflows.utils import ReadBack

pytestmark = pytest.mark.unit


def test_first_read_back_is_natural() -> None:
    read_back = ReadBack()

    assert read_back.instruction(natural="natural", spelled="spelled") == "natural"


def test_every_read_back_after_the_first_is_spelled() -> None:
    read_back = ReadBack()
    read_back.instruction(natural="natural", spelled="spelled")

    assert read_back.instruction(natural="natural", spelled="spelled") == "spelled"
    assert read_back.instruction(natural="natural", spelled="spelled") == "spelled"
