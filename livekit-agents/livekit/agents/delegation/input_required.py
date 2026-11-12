"""Which of an expert's open questions a delegation answers.

The model facing the person sends their reply to the expert that asked, as a call of its tool.
When the expert has questions open, the reply may answer one of them, be a request of its own,
or fit more than one; a detector reads the conversation and decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from livekit import rtc

from ..llm._provider_format.utils import tool_updates_as_notes
from ..llm.chat_context import ChatContext, ChatMessage
from ..llm.llm import LLM
from ..llm.tool_context import AskedQuestion, function_tool
from ..types import APIConnectOptions

if TYPE_CHECKING:
    from ..inference.llm import LLMModels
    from ..metrics import AgentMetrics
    from ..voice.run_result import RunUsage

DEFAULT_DETECTOR_MODEL = "google/gemma-4-31b-it"

DEFAULT_PROMPT = (
    "An agent talks with a person and hands work to an expert. The expert asked the person the "
    "open questions listed in the input, each about a request the agent sent it earlier, and the "
    "agent now sends the expert the request in the input. Decide, from the person's own words in "
    "the conversation: the id of the open question whose earlier request this one continues, the "
    "same thing with an answer, a detail, a change or a question added; 'new' only when it is "
    "about something else; 'unclear' when it could continue more than one of them and neither "
    "the conversation nor the request shows which."
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
    usage: RunUsage | None = None
    """The run's usage, which the detector's model call adds to."""


class InputRequiredDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(
        self,
        model: LLM | LLMModels | str = DEFAULT_DETECTOR_MODEL,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_items: int = 20,
    ) -> None:
        """Judge which open question a delegation answers, reporting the model's usage as
        ``metrics_collected``.

        Args:
            model: An LLM, which answers through a forced tool call, or a LiveKit Inference
                model name.
            prompt: The question asked about each delegation.
            max_items: How many of the newest conversation items the model reads.
        """
        super().__init__()
        # a model made from a name is the detector's own, so aclose closes it
        self._owned_model: LLM | None = None
        if isinstance(model, str):
            from ..inference import LLM as InferenceLLM

            model = self._owned_model = InferenceLLM(model)
        self._model = model
        self._model.on("metrics_collected", self._forward_metrics)
        self._prompt = prompt
        self._max_items = max_items

    @property
    def model(self) -> LLM:
        return self._model

    async def aclose(self) -> None:
        self._model.off("metrics_collected", self._forward_metrics)
        if self._owned_model is not None:
            await self._owned_model.aclose()

    def _forward_metrics(self, metrics: AgentMetrics) -> None:
        self.emit("metrics_collected", metrics)

    async def detect(self, check: InputRequiredCheck) -> InputRequiredVerdict:
        """The question the delegation answers, ``new`` or ``unclear``; a subclass overriding it
        judges another way."""
        ids = {f"Q{i}": q for i, q in enumerate(check.questions, 1)}
        lines = []
        # a tool's later updates read as notes, as the model facing the person reads them
        outputs = {i.id for i in check.chat_ctx.items if i.type == "function_call_output"}
        for item in tool_updates_as_notes(check.chat_ctx).items[-self._max_items :]:
            if item.type == "message" and item.role in ("user", "assistant") and item.text_content:
                lines.append(f"{item.role}: {item.text_content}")
            elif item.type == "message" and item.id in outputs and item.text_content:
                lines.append(f"note: {item.text_content}")
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
        if check.usage is not None:
            check.usage._add_completion(collected.usage)
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
