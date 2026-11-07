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
from .tool_context import (
    ProviderTool,
    Tool,
    function_tool,
    is_function_tool,
    is_raw_function_tool,
)

if TYPE_CHECKING:
    from ..inference.llm import LLMModels

DEFAULT_DETECTOR_MODEL = "google/gemma-4-31b-it"

# how late a caller turn may start and still count as said over the agent's message: a turn
# starts when its first words are transcribed, which trails the speech itself (not the grace period)
_SPOKEN_OVER_SLACK = 1.0
# the backend's history keeps no overlap, so a turn said over the agent is delegated with this
# note; it quotes nothing the caller said, which would reach the backend as a developer message
_SPOKEN_OVER_NOTE = (
    "The caller's last turn was said while the agent was still speaking, so they may not have "
    "heard all of it."
)

DEFAULT_PROMPT = (
    "A caller talks to a voice agent that cannot act on its own: only its backend can, through "
    "the backend tools listed in the input, and the voice agent has to hand the backend each such "
    "request and each answer to the backend's questions. Sometimes it forgets, and answers, "
    "promises or goes quiet instead. True when the caller's last turn asks for something only the "
    "backend tools can do; or answers a question the agent asked for such work with a yes, a no, a "
    "choice or a detail, a bare 'okay', 'mm-hmm' or 'yep' given as that answer included, even said "
    "while the agent was still speaking; or the agent says it is doing or has done something that "
    "needs the tools; and no tool call has done it yet. False for thanks, small talk and other "
    "turns that answer nothing, a question the conversation already answers, a request the agent "
    "is still collecting details for, work a tool call already did, or a turn after which the "
    "agent asks the caller a new question they have not answered yet."
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

    async def __call__(self, check: DelegationCheck) -> bool | str:
        tools: list[str] = []
        for tool in check.tools:
            if is_function_tool(tool):
                tools.append(f"- {tool.info.name}: {tool.info.description or ''}")
            elif is_raw_function_tool(tool):
                tools.append(f"- {tool.info.name}: {tool.info.raw_schema.get('description', '')}")
            elif isinstance(tool, ProviderTool):
                tools.append(f"- {tool.id}: a tool the provider hosts")
        parts = ["## Backend tools", "\n".join(tools) or "(none)"]
        if check.instructions:
            parts += ["## The voice agent's instructions", check.instructions]
        # the transcript keeps no overlap: a caller turn that started while an agent message was
        # playing reads like an ordinary reply, so it is marked from the messages' playback times
        messages = [m for m in check.chat_ctx.items if isinstance(m, ChatMessage)]
        playing = [
            (m.metrics["started_speaking_at"], m.metrics["stopped_speaking_at"])
            for m in messages
            if m.role == "assistant"
            and "started_speaking_at" in m.metrics
            and "stopped_speaking_at" in m.metrics
        ]
        spoken_over = {
            m.id
            for m in messages
            if m.role == "user"
            and any(
                a <= m.metrics.get("started_speaking_at", m.created_at) <= b + _SPOKEN_OVER_SLACK
                for a, b in playing
            )
        }
        lines = []
        for item in check.chat_ctx.items[-self._max_items :]:
            line = _format_transcript(ChatContext(items=[item]), include_context_events=True)
            if line and item.id in spoken_over:
                line = line.replace("user:", "user (said while the agent was still speaking):", 1)
            if line:
                lines.append(line)
        parts += ["## Conversation", "\n".join(lines)]
        input = "\n\n".join(parts)
        if isinstance(self._model, DecisionModel):
            response = await self._model.evaluate(
                input=input, decisions={"delegate": Probability(self._prompt)}
            )
            result = response.results.get("delegate")
            if not isinstance(result, ProbabilityResult):
                logger.warning("delegation decision failed", extra={"errors": response.errors})
                return False
            delegate = result.value >= 0.5
        else:

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
                            content=[
                                f"{self._prompt}\nJudge only from the input, then call report."
                            ],
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
            delegate = args.get("delegate") is True

        last = next((m for m in reversed(messages) if m.role == "user"), None)
        if delegate and last is not None and last.id in spoken_over:
            return _SPOKEN_OVER_NOTE
        return delegate
