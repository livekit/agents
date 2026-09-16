"""The extension read as Python: what goes into a task, and what comes back out of one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..llm.chat_context import ChatContext, ChatItem
from ..voice.served_request import Directive

TaskState = Literal["working", "completed", "failed", "canceled", "input-required"]
"""How far a task has got, in A2A's task states.

``working`` is intermediate and repeats; the rest end the task. ``input-required`` ends it
too — the answer is a question, nothing waits on it, and the reply is a new task.
"""


@dataclass
class TaskInput:
    """One message sent on a context: a person's turn, or an agent asking for work.

    Exactly one of ``text`` and ``instruction`` is set, and which one is the difference
    between a person's turn and a delegation on the wire. ``instruction`` may be empty: a
    duplex model delegates without saying anything, and the receiver answers the last user
    message in ``chat_ctx``.
    """

    text: str | None = None
    """A person's words."""
    instruction: str | None = None
    """What an agent is asking for, in its words."""
    chat_ctx: ChatContext = field(default_factory=ChatContext.empty)
    """The whole conversation the sender holds. The receiver takes what it has not seen, by
    item id, and ignores the rest; a delta is never computed by the sender."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data, handed to the handler untouched. JSON-serializable."""

    def __post_init__(self) -> None:
        if self.closing:
            self.text = self.text if self.text is not None else ""
        if (self.text is None) == (self.instruction is None):
            raise ValueError("a TaskInput carries exactly one of `text` and `instruction`")

    closing: bool = False
    """The conversation is over: nothing is being asked, and the receiver may drop it."""

    @property
    def is_delegation(self) -> bool:
        return self.instruction is not None

    @property
    def body(self) -> str:
        return self.instruction if self.instruction is not None else (self.text or "")


@dataclass
class TaskUpdate:
    """One event from a task: where it has got to, what it produced, what to say about it."""

    state: TaskState = "working"
    text: str = ""
    """Relayed text: what the caller's conversation model is given to phrase, or what a chat
    client shows. Empty when the event carries only an item."""
    item: ChatItem | None = None
    """The typed item behind the event, for rendering and history."""
    verbatim: bool = False
    """Say ``text`` as written rather than phrasing it, once."""
    directive: Directive | None = None
    """Acted on after the text is said. Only on ``completed``."""
