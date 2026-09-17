"""The extension read as Python: what goes into a task, and what comes back out of one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..llm.chat_context import ChatContext, ChatItem
from ..voice.served_request import Directive

TaskState = Literal["working", "completed", "failed", "canceled", "input-required"]
"""How far a task has got: ``working`` repeats, and every other state ends it.

``input-required`` ends it too — the answer is a question, and the reply is a new task.
"""


@dataclass
class TaskInput:
    """One message on a context: a person's turn, or an agent asking for work.

    Exactly one of ``text`` and ``instruction`` is set, and which one is what tells the two
    apart on the wire.
    """

    text: str | None = None
    """A person's words."""
    instruction: str | None = None
    """What an agent is asking for, in its words. Empty where it says nothing of its own."""
    chat_ctx: ChatContext = field(default_factory=ChatContext.empty)
    """The whole conversation the sender holds; the receiver takes the delta by item id."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data, handed to the handler untouched. JSON-serializable."""
    closing: bool = False
    """The conversation is over: nothing is being asked, and the receiver may drop it."""

    def __post_init__(self) -> None:
        if self.closing and self.text is None and self.instruction is None:
            self.text = ""  # a goodbye asks for nothing and still has to be one of the two
        if (self.text is None) == (self.instruction is None):
            raise ValueError("a TaskInput carries exactly one of `text` and `instruction`")

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
