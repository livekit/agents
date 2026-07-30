from __future__ import annotations

from enum import Enum
from typing import Any

from ...llm.chat_context import Instructions
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.delegation import DelegationOptions


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


# a workflow prompt is written for one model that both talks and calls the tools. a delegation
# LLM splits that work in two, so each model reads the prompt and a directive for its half
DELEGATED_SPEAKER_DIRECTIVE = """\
The tools above are not yours to call. A second model holds them and does that work behind
`lk_agents_delegate`, out of the user's sight; you are the one talking to them. So hand it
everything the user says about this — an answer, a correction, a yes or a no — even where that
is only repeating what they already said, and say what comes back in your own words, as your
own. That second model is never the user's business, whatever the work itself turns out to be."""

DELEGATED_TOOL_CALLER_DIRECTIVE = """\
You call the tools; the agent talking to the user cannot see them. Where a tool tells you
something must be said, asked or read back, put it in your answer as it stands, the value
spelled out as the tool spelled it. Passing that on is not phrasing a reply — the words are
the tool's, and the agent still does the talking."""


def _with_directive(instructions: str | Instructions, directive: str) -> str | Instructions:
    """``instructions`` with ``directive`` appended to each of its modality variants."""
    if isinstance(instructions, str):
        return f"{instructions}\n{directive}"

    audio = f"{instructions.render(modality='audio')}\n{directive}"
    text = f"{instructions.render(modality='text')}\n{directive}"
    return Instructions(audio) if audio == text else Instructions(audio=audio, text=text)


def _delegated_tool_caller_options(
    instructions: str | Instructions, options: NotGivenOr[DelegationOptions]
) -> DelegationOptions:
    """``options`` with the tool-calling half of ``instructions`` filled in, unless it is set."""
    given: DelegationOptions = options if is_given(options) else {}
    if is_given(given.get("instructions", NOT_GIVEN)):
        return given

    delegated = _with_directive(instructions, DELEGATED_TOOL_CALLER_DIRECTIVE)
    return DelegationOptions(**{**given, "instructions": delegated})


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
        super().__init__(audio=audio, text=text)
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
