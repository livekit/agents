"""Which of an expert's open questions a delegation answers.

The model facing the person sends their reply to the expert that asked, as a call of its tool.
When the expert has questions open, the reply may answer one of them, be a request of its own,
or fit more than one; a detector reads the conversation and decides.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..llm.chat_context import ChatContext, ChatMessage
from ..llm.llm import LLM
from ..llm.tool_context import AskedQuestion, function_tool
from ..types import APIConnectOptions

if TYPE_CHECKING:
    from ..inference.llm import LLMModels

DEFAULT_DETECTOR_MODEL = "google/gemma-4-31b-it"

DEFAULT_PROMPT = (
    "An agent talks with a person and hands work to an expert. The expert asked the person the "
    "open questions listed in the input, and the agent now sends the expert the request in the "
    "input. Decide what the request is, from the person's own words in the conversation: the id "
    "of the one question their latest reply answers or responds to, with a yes, a no, a choice, "
    "a detail, a correction or a question about it; 'new' when it answers none of them; "
    "'unclear' when the reply could answer more than one of them and neither the conversation "
    "nor the request shows which. When the reply answers several and the request is about one "
    "of them, that one."
)

InputRequiredVerdict = AskedQuestion | Literal["new", "unclear"]
"""The question the request answers, ``new`` for a request of its own, or ``unclear``."""


@dataclass
class InputRequiredCheck:
    """A delegation while the expert has questions open, for a detector to judge."""

    request: str
    """What the model sends the expert."""
    questions: list[AskedQuestion]
    """The expert's open questions the person spoke after, oldest first."""
    chat_ctx: ChatContext
    """The conversation, up to and including the person's reply."""


InputRequiredDetectorFn = Callable[[InputRequiredCheck], Awaitable[InputRequiredVerdict]]


class InputRequiredDetector:
    def __init__(
        self,
        model: LLM | LLMModels | str = DEFAULT_DETECTOR_MODEL,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_items: int = 20,
    ) -> None:
        """Judge which open question a delegation answers.

        Args:
            model: An LLM, which answers through a forced tool call, or a LiveKit Inference
                model name.
            prompt: The question asked about each delegation.
            max_items: How many of the newest conversation items the model reads.
        """
        if isinstance(model, str):
            from ..inference import LLM as InferenceLLM

            model = InferenceLLM(model)
        self._model = model
        self._prompt = prompt
        self._max_items = max_items

    @property
    def model(self) -> LLM:
        return self._model

    async def __call__(self, check: InputRequiredCheck) -> InputRequiredVerdict:
        ids = {f"Q{i}": q for i, q in enumerate(check.questions, 1)}
        lines = []
        for item in check.chat_ctx.items[-self._max_items :]:
            if item.type == "message" and item.role in ("user", "assistant") and item.text_content:
                lines.append(f"{item.role}: {item.text_content}")
            elif item.type == "function_call":
                lines.append(f"call {item.name}: {item.arguments}")
            elif item.type == "function_call_output":
                lines.append(f"result of {item.name}: {item.output}")
        input = "\n\n".join(
            [
                "## Open questions",
                "\n".join(f"- {key}: {q.question}" for key, q in ids.items()),
                "## Conversation",
                "\n".join(lines),
                "## Request",
                check.request,
            ]
        )
        choices = [*ids, "new", "unclear"]

        @function_tool
        async def report(verdict: str) -> None:
            """Report the verdict.

            Args:
                verdict: One of the question ids, 'new' or 'unclear'.
            """

        collected = await self._model.chat(
            chat_ctx=ChatContext(
                items=[
                    ChatMessage(
                        role="system",
                        content=[f"{self._prompt}\nAnswer with one of: {', '.join(choices)}."],
                    ),
                    ChatMessage(role="user", content=[input]),
                ]
            ),
            tools=[report],
            tool_choice="required",
            # the delegation waits on it: a failure is not retried
            conn_options=APIConnectOptions(max_retry=0),
        ).collect()
        call = next((c for c in collected.tool_calls if c.name == "report"), None)
        try:
            verdict = json.loads(call.arguments).get("verdict") if call is not None else None
        except (json.JSONDecodeError, AttributeError):
            verdict = None
        if verdict == "new":
            return "new"
        if verdict == "unclear":
            return "unclear"
        if verdict in ids:
            return ids[verdict]
        raise ValueError(f"the detector reported no verdict: {verdict!r}")
