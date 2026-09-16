"""Where a conversation hands reasoning and tool use."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from typing_extensions import TypedDict

if TYPE_CHECKING:
    # core reads DelegationOptions from here, and must not need the a2a extra to do it
    from ..a2a import TaskInput, TaskUpdate

DELEGATE_TOOL_NAME = "lk_agents_delegate"
"""The one tool a delegate is reached through. Fixed, so a duplex model can synthesize it."""


@runtime_checkable
class DelegateStream(Protocol):
    """The updates of one delegation, until it declares a terminal state."""

    async def __anext__(self) -> TaskUpdate: ...
    def __aiter__(self) -> Any: ...
    async def __aenter__(self) -> Any: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    async def cancel(self, reason: str = "") -> None: ...
    async def aclose(self) -> None: ...


class Delegate(ABC):
    """An expert the conversation hands work to, here or behind a socket.

    A delegate is attached to one ``AgentSession`` or one ``Agent`` and closed by it, so one
    holding a connection or a session of its own releases it in :meth:`aclose`. Give each
    conversation its own.
    """

    @abstractmethod
    def submit(self, task_input: TaskInput) -> DelegateStream:
        """Start one delegation and hand back the stream of its updates."""

    async def aclose(self) -> None:  # noqa: B027
        """Release what the delegate holds. Called by whatever it is attached to."""


class DelegationOptions(TypedDict, total=False):
    """Configuration for delegation, as a plain dict::

    AgentSession(delegate=..., delegation_options={"metadata": {"customer_id": "c-42"}})
    """

    metadata: dict[str, Any]
    """Application data attached to every delegation. JSON-serializable. Defaults to ``{}``."""
    announce: bool
    """Whether answering the dispatch note is what acknowledges a delegation. Defaults to True.

    A model cannot be relied on to write a line alongside the tool call, and realtime models
    routinely emit the call and no speech. Turn it off for a model that does write one — the
    tool then asks for it, which costs no round trip.
    """


def resolve_delegation_options(config: DelegationOptions | None = None) -> DelegationOptions:
    """Fill in defaults for missing keys."""
    opts = DelegationOptions(metadata={}, announce=True)
    opts.update(config or {})
    return opts
