from __future__ import annotations

from enum import Enum
from typing import Any

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


class WorkflowInstructions(Instructions):
    """Customizable instruction sections for built-in workflow tasks.

    Extends :class:`Instructions` with ``persona`` and ``extra`` fields
    that workflow tasks resolve against their own templates and defaults.

    Each field overrides that section when set; leave as ``NOT_GIVEN`` to
    preserve the workflow's built-in default. Set to ``""`` to remove a
    section entirely.
    """

    def __init__(
        self,
        audio: str = "",
        *,
        text: str | None = None,
        persona: NotGivenOr[Instructions | str] = NOT_GIVEN,
        extra: Instructions | str = "",
    ) -> None:
        super().__init__(audio, text=text)
        self.persona: NotGivenOr[Instructions | str] = persona
        self.extra: Instructions | str = extra

    def resolve(
        self,
        *,
        template: str,
        default_persona: str,
        **format_kwargs: Any,
    ) -> Instructions:
        """Resolve into a final :class:`Instructions` by formatting the template."""
        return Instructions.resolve_template(
            template,
            persona=self.persona if is_given(self.persona) else default_persona,
            extra=self.extra,
            **format_kwargs,
        )
