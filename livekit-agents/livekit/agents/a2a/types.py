"""The extension read as Python: what goes into a task, and what comes back out of one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..llm.chat_context import ChatContext, ChatItem
from ..voice.served_request import Directive

TaskControl = Literal["prewarm", "interrupt", "close"]
"""A message that is not a turn: ``prewarm`` starts the context's session, ``interrupt``
stops task responses, and ``close`` ends the context."""

TaskState = Literal["working", "completed", "failed", "canceled", "input-required"]
"""How far a task has got: ``working`` repeats, and every other state ends it.

``input-required`` ends it too — the answer is a question, and the reply is a new task.
"""


@dataclass
class TaskInput:
    """One message on a context: a person's turn, an agent asking for work, or a control.

    A turn sets exactly one of ``text`` and ``instruction``, and which one is what tells the
    two apart on the wire; a control sets neither.
    """

    text: str | None = None
    """A person's words."""
    instruction: str | None = None
    """What an agent is asking for, in its words. Empty where it says nothing of its own."""
    chat_ctx: ChatContext = field(default_factory=ChatContext.empty)
    """The whole history the sender holds; the receiver takes the delta by item id."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data, handed to the handler untouched. JSON-serializable."""
    control: TaskControl | None = None
    """Set when the message asks nothing and takes no turn: start the session, interrupt
    tasks, or end the context."""
    interrupting: list[str] | None = None
    """Tasks whose response to interrupt, keeping their background tool calls running; empty
    for every task of the context."""
    conversation_id: str | None = None
    """The sender's conversation, whose database the receiver persists into."""
    caller_session_id: str | None = None
    """The sender's session in that conversation, which the receiver's session hangs under."""
    context_id: str | None = None
    """The A2A context the request continues, or None to open a new one."""

    def __post_init__(self) -> None:
        if self.control is not None:
            if self.text is not None or self.instruction is not None:
                raise ValueError("a control message carries neither `text` nor `instruction`")
        elif (self.text is None) == (self.instruction is None):
            raise ValueError("a TaskInput carries exactly one of `text` and `instruction`")
        if self.control == "interrupt" and self.interrupting is None:
            raise ValueError("an interrupt carries `interrupting`, the tasks it interrupts")
        if self.control in ("prewarm", "close") and self.interrupting is not None:
            raise ValueError(f"a {self.control} message interrupts nothing")

    @property
    def is_turn(self) -> bool:
        return self.control is None

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
