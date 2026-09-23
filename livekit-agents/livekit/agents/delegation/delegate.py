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

    @property
    def task_id(self) -> str:
        """The far side's id for this delegation. Empty until its first event has arrived."""
        ...

    async def __anext__(self) -> TaskUpdate: ...
    def __aiter__(self) -> Any: ...
    async def __aenter__(self) -> Any: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    async def cancel(self, reason: str = "") -> None:
        """Ask the far side to stop the work; best-effort."""
        ...

    async def aclose(self) -> None:
        """Stop reading and release what this stream holds here."""
        ...


class Delegate(ABC):
    """An expert the conversation hands work to, here or behind a socket.

    Attached to one ``AgentSession`` or one ``Agent`` and closed by it, so give each
    conversation its own.
    """

    @abstractmethod
    def submit(self, task_input: TaskInput) -> DelegateStream:
        """Start one delegation and hand back the stream of its updates."""

    @property
    def endpoint(self) -> str | None:
        """Where the far side lives, stable across restarts, or None when it has no address.

        A persisted session finds the conversation it last had there by this.
        """
        return None

    def resume(self, context_id: str) -> bool:
        """Continue an earlier conversation on the far side instead of opening a new one.

        Returns False, changing nothing, once this delegate has fixed a context of its own.
        """
        return False

    async def aclose(self) -> None:  # noqa: B027
        """Release what the delegate holds. Called by whatever it is attached to."""


class DelegationOptions(TypedDict, total=False):
    """A delegate and how the conversation reaches it, as a plain dict::

    AgentSession(delegate={"delegate": A2ADelegate(url), "metadata": {"customer_id": "c-42"}})
    """

    delegate: Delegate | None
    """Where the work goes. Defaults to ``None``, which offers the tool to nobody."""
    metadata: dict[str, Any]
    """Application data attached to every delegation. JSON-serializable. Defaults to ``{}``."""
    announce: bool
    """Whether answering the dispatch note is what acknowledges a delegation, default True.

    Turn it off for a model that reliably writes its own line alongside the call, which
    costs no round trip.
    """


def resolve_delegation_options(
    config: Delegate | DelegationOptions | None = None,
) -> DelegationOptions:
    """Fill in defaults, taking a bare delegate as the one-key form of the same thing."""
    opts = DelegationOptions(delegate=None, metadata={}, announce=True)
    if isinstance(config, Delegate):
        opts["delegate"] = config
    elif config is not None:
        opts.update(config)
    return opts
