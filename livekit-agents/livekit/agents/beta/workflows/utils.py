from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ...llm.chat_context import Instructions
from ...types import NOT_GIVEN, NotGivenOr


class DtmfEvent(str, Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"
    STAR = "*"
    POUND = "#"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


def dtmf_event_to_code(event: DtmfEvent) -> int:
    if event.value.isdigit():
        return int(event.value)
    elif event.value == "*":
        return 10
    elif event.value == "#":
        return 11
    elif event.value in ["A", "B", "C", "D"]:
        # DTMF codes 10-15 are used for letters A-D
        return ord(event.value) - ord("A") + 12
    else:
        raise ValueError(f"Invalid DTMF event: {event}")


def format_dtmf(events: list[DtmfEvent]) -> str:
    return " ".join(event.value for event in events)


@dataclass
class InstructionParts:
    """Customizable instruction sections for built-in workflow tasks.

    Each field overrides that section when set; leave as ``NOT_GIVEN`` to
    preserve the workflow's built-in default. Set to ``""`` to remove a
    section entirely.

    Args:
        persona: Agent persona/identity — who the agent is and how it behaves.
        extra: Extra instructions appended to the prompt. The simplest hook for
            adding domain context without touching defaults.
    """

    persona: NotGivenOr[Instructions | str] = NOT_GIVEN
    extra: Instructions | str = ""
