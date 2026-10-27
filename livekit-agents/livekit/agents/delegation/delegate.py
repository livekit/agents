"""Where a conversation hands reasoning and tool use.

A delegate answers one request inside the caller's turn and the caller phrases the result.
``A2ADelegate`` (``a2a.py``) reaches an expert over A2A, an endpoint of this project or any A2A
server; the interface is here for anything else that answers a request, with what a delegation
leaves on the call that made it.
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

from ..llm.chat_context import ChatContext, ChatItem, ChatMessage, FunctionCall
from ..voice.run_result import RunResult, RunUsage

Progress = Callable[[str], Awaitable[None] | None]
"""``progress=`` and ``say=``: take the expert's status messages, worded by the caller or said
as written; ``ctx.update`` and ``ctx.say`` in a tool, a print in a script."""


class DelegationFailed(Exception):
    """The expert ended the task ``failed``; the message is what it said."""


class DelegationResult(RunResult[object]):
    """What a delegate's ``run`` returns: a finished ``RunResult`` with the task behind it.

    ``state`` is the A2A terminal state; ``task_id`` and ``context_id`` name the expert's task;
    ``endpoint``, ``transport`` and ``profile`` say where and how it was reached over A2A.
    """

    def __init__(
        self,
        *,
        output: object,
        state: str = "completed",
        task_id: str = "",
        context_id: str = "",
        input_required: str | None = None,
        new_messages: Sequence[ChatItem] | None = None,
        usage: RunUsage | None = None,
        interrupted: bool = False,
        endpoint: str = "",
        transport: str = "",
        profile: str = "",
        verbatim: str | list[str] | None = None,
    ) -> None:
        super().__init__(output_type=None)
        self.state = state
        self.task_id = task_id
        self.context_id = context_id
        self.endpoint = endpoint
        self.transport = transport
        self.profile = profile
        self._run_output = output
        self._input_required = input_required
        self._verbatim = verbatim
        self._interrupted = interrupted
        self._run_messages = list(new_messages or [])
        self._run_history = list(new_messages or [])
        self._run_usage = usage if usage is not None else RunUsage()
        self._done_fut.set_result(None)


TASK_ID_KEY = "lk.delegation.task_id"
CONTEXT_ID_KEY = "lk.delegation.context_id"
ENDPOINT_KEY = "lk.delegation.endpoint"
TRANSPORT_KEY = "lk.delegation.transport"
PROFILE_KEY = "lk.delegation.profile"
"""``FunctionCall.extra``: the expert's task that answered the call and its context, where the
expert was reached, over which binding, under which profile (``livekit`` or ``a2a``), so a
dashboard joins the delegation to the call as it ran, and the next delegation to a stock expert
goes back to the context it keeps."""


def record_delegation(function_call: FunctionCall, result: DelegationResult) -> None:
    """Writes the expert's task id and how it was reached on the call that delegated."""
    recorded = (
        (TASK_ID_KEY, result.task_id),
        (CONTEXT_ID_KEY, result.context_id),
        (ENDPOINT_KEY, result.endpoint),
        (TRANSPORT_KEY, result.transport),
        (PROFILE_KEY, result.profile),
    )
    for key, value in recorded:
        if value:
            function_call.extra[key] = value


async def report(progress: Progress | None, text: str) -> None:
    """Hands a status message to ``progress``, sync or async."""
    if progress is None or not text:
        return

    out = progress(text)
    if inspect.isawaitable(out):
        await out


def answer_message(text: str) -> list[ChatItem]:
    """The expert's answer as the one assistant message a delegation adds; none for no text."""
    if not text:
        return []
    return [ChatMessage(role="assistant", content=[text], created_at=time.time())]


class Delegate(ABC):
    """An expert the conversation model hands work to."""

    @abstractmethod
    async def run(
        self,
        prompt: str,
        *,
        progress: Progress | None = None,
        say: Progress | None = None,
        context_id: str | None = None,
        message_history: ChatContext | Sequence[ChatItem] | None = None,
        usage: RunUsage | None = None,
    ) -> DelegationResult:
        """One request, answered once. ``context_id`` is the caller's session, when it has one;
        ``message_history`` the conversation so far; ``progress`` takes the expert's status
        messages and ``say`` its exact text."""

    async def aclose(self) -> None:  # noqa: B027
        """Release what the delegate holds."""


__all__ = [
    "CONTEXT_ID_KEY",
    "ENDPOINT_KEY",
    "PROFILE_KEY",
    "TASK_ID_KEY",
    "TRANSPORT_KEY",
    "Delegate",
    "DelegationFailed",
    "DelegationResult",
    "Progress",
    "answer_message",
    "record_delegation",
    "report",
]
