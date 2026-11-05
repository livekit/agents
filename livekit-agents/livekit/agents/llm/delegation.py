from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

from ..log import logger
from ..types import APIConnectOptions
from .chat_context import ChatContext, ChatMessage
from .decisions import DecisionModel, Probability, ProbabilityResult, _format_transcript
from .llm import LLM
from .tool_context import Tool, function_tool, is_function_tool, is_raw_function_tool

if TYPE_CHECKING:
    from ..inference.llm import LLMModels

DEFAULT_DETECTOR_MODEL = "google/gemma-4-31b-it"

DEFAULT_PROMPT = (
    "A caller talks to a voice agent that cannot act on its own: only its backend can, through "
    "the backend tools listed in the input, and the voice agent has to hand each such request to "
    "the backend. Sometimes it forgets, and answers or promises the action itself. "
    "True when the caller's last turn asks for something only the backend tools can do, or the "
    "agent says it is doing, has done, or will check something that needs them, and no tool "
    "call in the conversation has done it yet. False for small talk, a question the conversation "
    "already answers, a request the agent is still collecting details for, or work a tool call "
    "already did."
)


@dataclass
class DelegationCreatedEvent:
    """A delegation started, by the model or by ``delegate()``."""

    id: str
    source: Literal["model", "manual"] = "model"
    """``manual`` when the application started it with ``delegate()``."""


@dataclass
class DelegationCheck:
    """A finished user turn the model did not delegate, for a detector to judge."""

    chat_ctx: ChatContext
    """The conversation up to and including the turn, with the agent's reply to it."""
    tools: list[Tool] = field(default_factory=list)
    """What delegated work can do."""
    instructions: str | None = None
    """The agent's instructions, where a delegation policy usually lives."""


DelegationDetectorFn = Callable[[DelegationCheck], Awaitable[bool | str]]
"""Judges a turn: ``False`` leaves it, ``True`` delegates it, and a string delegates it with that
string as a note to the delegated work."""


class DelegationDetectionOptions(TypedDict, total=False):
    detector: DelegationDetectorFn
    """Defaults to ``DelegationDetector()``."""
    grace_period: float
    """How long the model has to delegate on its own once it is quiet after the turn; 0.5 s."""


DEFAULT_GRACE_PERIOD = 0.5


class DelegationDetector:
    def __init__(
        self,
        model: LLM | DecisionModel | LLMModels | str = DEFAULT_DETECTOR_MODEL,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_items: int = 20,
    ) -> None:
        """Judge whether a turn the model did not delegate needed delegated work.

        Args:
            model: An LLM, which answers through a forced tool call; a decision model, which
                delegates at a probability of 0.5 or more; or a LiveKit Inference LLM name.
            prompt: The question asked about each turn.
            max_items: How many of the newest conversation items the model reads.
        """
        if isinstance(model, str):
            from ..inference import LLM as InferenceLLM

            model = InferenceLLM(model)
        self._model = model
        self._prompt = prompt
        self._max_items = max_items

    @property
    def model(self) -> LLM | DecisionModel:
        return self._model

    async def __call__(self, check: DelegationCheck) -> bool:
        tools: list[str] = []
        for tool in check.tools:
            if is_function_tool(tool):
                tools.append(f"- {tool.info.name}: {tool.info.description or ''}")
            elif is_raw_function_tool(tool):
                tools.append(f"- {tool.info.name}: {tool.info.raw_schema.get('description', '')}")
        parts = ["## Backend tools", "\n".join(tools) or "(none)"]
        if check.instructions:
            parts += ["## The voice agent's instructions", check.instructions]
        recent = ChatContext(items=check.chat_ctx.items[-self._max_items :])
        parts += ["## Conversation", _format_transcript(recent, include_context_events=True)]
        input = "\n\n".join(parts)

        if isinstance(self._model, DecisionModel):
            response = await self._model.evaluate(
                input=input, decisions={"delegate": Probability(self._prompt)}
            )
            result = response.results.get("delegate")
            if not isinstance(result, ProbabilityResult):
                logger.warning("delegation decision failed", extra={"errors": response.errors})
                return False
            return result.value >= 0.5

        @function_tool
        async def report(delegate: bool) -> None:
            """Report the verdict.

            Args:
                delegate: Whether the turn needs the backend now.
            """

        collected = await self._model.chat(
            chat_ctx=ChatContext(
                items=[
                    ChatMessage(
                        role="system",
                        content=[f"{self._prompt}\nJudge only from the input, then call report."],
                    ),
                    ChatMessage(role="user", content=[input]),
                ]
            ),
            tools=[report],
            tool_choice="required",
            # a verdict that arrives late is worth nothing, so a failure is not retried
            conn_options=APIConnectOptions(max_retry=0),
        ).collect()
        call = next((c for c in collected.tool_calls if c.name == "report"), None)
        try:
            args = json.loads(call.arguments) if call is not None else None
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            logger.warning("delegation detector did not report a verdict")
            return False
        return args.get("delegate") is True
