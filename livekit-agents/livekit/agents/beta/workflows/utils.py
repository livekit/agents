from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ...llm.chat_context import Instructions
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given


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
    """Modular instruction sections for built-in workflow tasks.

    Maps to the standard prompt taxonomy (Role, Context, Constraints).

    Each field overrides that section when set; leave as ``NOT_GIVEN`` to
    preserve the workflow's built-in default. Set to ``""`` to remove a
    section entirely.

    Args:
        role: Agent persona/identity.
        context: Background context — input channel rules, domain info, etc.
            Use ``Instructions(audio=..., text=...)`` for per-modality content.
        constraints: What the agent is not allowed to do.
        extra: Extra instructions to add to the prompt. The simplest hook for
            adding domain context without touching defaults.
    """

    role: NotGivenOr[Instructions | str] = NOT_GIVEN
    context: NotGivenOr[Instructions | str] = NOT_GIVEN
    constraints: NotGivenOr[Instructions | str] = NOT_GIVEN
    extra: Instructions | str = ""

    def copy(self) -> InstructionParts:
        return InstructionParts(
            role=self.role,
            context=self.context,
            constraints=self.constraints,
            extra=self.extra,
        )


def build_instructions(
    parts: NotGivenOr[InstructionParts],
    defaults: InstructionParts,
    directive: Instructions | str,
) -> Instructions:
    """Assemble a modality-aware ``Instructions`` from user overrides and workflow defaults.

    Section order: role → context → directive → constraints → extra.

    Args:
        parts: User-supplied overrides. ``NOT_GIVEN`` fields fall back to ``defaults``.
        defaults: Workflow's built-in values. All fields must be set (no ``NOT_GIVEN``).
        directive: Non-customizable tool-call instructions, pre-resolved by the
            workflow (confirmation logic applied before this call).
    """

    def _resolve(
        value: NotGivenOr[Instructions | str], default: NotGivenOr[Instructions | str]
    ) -> Instructions | str:
        return value if is_given(value) else (default if is_given(default) else "")

    if not is_given(parts):
        parts = InstructionParts()

    sections: list[Instructions | str] = [
        _resolve(parts.role, defaults.role),
        _resolve(parts.context, defaults.context),
        directive,
        _resolve(parts.constraints, defaults.constraints),
        parts.extra,
    ]

    instructions = Instructions("")
    for section in sections:
        if not section:
            continue
        if instructions:
            instructions += "\n"
        instructions += section
    return instructions
