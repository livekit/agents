"""Where a conversation hands reasoning and tool use.

A delegate answers one request inside the caller's turn and the caller phrases the result.
``A2ADelegate`` (``a2a.py``) reaches an expert over A2A, an endpoint of this project or any A2A
server; the interface is here for anything else that answers a request. The ``delegate`` tool
of ``tool.py`` and ``LiveSession(delegate=)`` take any ``Delegate``.
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

from ..llm.chat_context import ChatContext, ChatItem, ChatMessage
from ..voice.run_result import RunResult, RunUsage

Progress = Callable[[str], Awaitable[None] | None]
"""``progress=``: takes the expert's status messages; ``ctx.update`` on a call, a print in a
script."""


class DelegationFailed(Exception):
    """The expert ended the task ``failed``; the message is what it said."""


class DelegationResult(RunResult[object]):
    """What a delegate's ``run`` returns: a finished ``RunResult`` with the task behind it.

    ``state`` is the A2A terminal state; ``task_id`` and ``context_id`` name the expert's task; ``endpoint``, ``transport``
    and ``profile`` say where and how it was reached over A2A.
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
        verbatim: str | None = None,
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
        context_id: str | None = None,
        message_history: ChatContext | Sequence[ChatItem] | None = None,
        usage: RunUsage | None = None,
    ) -> DelegationResult:
        """One request, answered once. ``context_id`` is the caller's session, when it has one;
        ``message_history`` the conversation so far; ``progress`` takes the expert's status
        messages."""

    async def aclose(self) -> None:  # noqa: B027
        """Release what the delegate holds."""


__all__ = [
    "Delegate",
    "DelegationFailed",
    "DelegationResult",
    "Progress",
    "answer_message",
    "report",
]
