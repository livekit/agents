from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools_services import _resolve_flower_destination

from livekit.agents import ToolError


def test_location_only() -> None:
    assert _resolve_flower_destination("Penthouse Suite", None) == "Penthouse Suite"


def test_recipient_only() -> None:
    assert _resolve_flower_destination(None, "Diane Okafor") == "Diane Okafor"


def test_location_wins_when_both_are_given() -> None:
    # the observed failure: "Penthouse Suite, Diane Okafor" jammed into one field.
    # with both known, the stored destination is the location alone.
    assert _resolve_flower_destination("Penthouse Suite", "Diane Okafor") == "Penthouse Suite"


def test_values_are_trimmed() -> None:
    assert _resolve_flower_destination("  412 ", None) == "412"


def test_blank_location_falls_back_to_recipient() -> None:
    assert _resolve_flower_destination("   ", "Diane Okafor") == "Diane Okafor"


def test_neither_raises_tool_error() -> None:
    with pytest.raises(ToolError):
        _resolve_flower_destination(None, None)
    with pytest.raises(ToolError):
        _resolve_flower_destination("  ", "")
