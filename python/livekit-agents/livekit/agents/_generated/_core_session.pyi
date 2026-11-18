# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from a2a.types.a2a_pb2 import Message, Task

from .delegation_remote import A2AJournalEntry, InputRequest

class RemoteAgentState:
    """Retained communication with one configured remote, scoped to the selected conversation."""
    @property
    def endpoint(self) -> str:
        """Pinned destination, without credentials."""
    @property
    def events(self) -> list[A2AJournalEntry]:
        """Observed communication in local journal order."""
    @property
    def messages(self) -> dict[str, Message]:
        """Observed messages in either direction, indexed by message ID; includes contextless replies."""
    @property
    def tasks(self) -> dict[str, RemoteTask]:
        """Observed tasks, indexed by task ID; their protocol history may be absent."""
    @property
    def contexts(self) -> dict[str, RemoteContextState]:
        """Grouping of the same observations by context ID, without another message history."""

class RemoteContextState:
    """Observed messages and tasks grouped by their remote context ID."""
    @property
    def messages(self) -> dict[str, Message]: ...
    @property
    def tasks(self) -> dict[str, RemoteTask]: ...

class RemoteTask:
    """A task's current protocol observation and its task-owned local routing facts."""
    @property
    def task(self) -> Task:
        """The observed SDK Task, including an unspecified status when only deltas were received."""
    @property
    def input_requests(self) -> dict[str, InputRequest]:
        """Retained input occurrences, indexed by occurrence ID independently of question wording."""
    @property
    def cancel_requested(self) -> bool:
        """Explicit local intent; only an observed task state confirms remote cancellation."""
    async def cancel(self) -> None:
        """Request cancellation through the current bound native owner and its configured
        remote client. Detached views and tasks without a live driver cannot dispatch.
        Success requires persisted observation of the canceled task state.
        """
