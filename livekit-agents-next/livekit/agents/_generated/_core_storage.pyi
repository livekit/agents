# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from .._core import Streams, Thread
from .storage_class import TaskState, ThreadInfo, ThreadItemPage, ThreadPage, ThreadTask

class Storage:
    """Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
    directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
    """
    def __init__(
        self,
        *,
        directory: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: float | None = None,
        renew_interval: float | None = None,
    ) -> None:
        """`directory`: threads in that local directory, resolved to an absolute path once.
        Otherwise agentstore through the public API derived from `url` (the project's
        `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
        """
    @property
    def location(self) -> str:
        """The absolute local directory, or the public API agentstore is reached through."""
    async def create_thread(
        self,
        *,
        scope: str | None = None,
        attributes: dict[str, object] | None = None,
        idle_ttl_seconds: int = 0,
    ) -> ThreadInfo:
        """A new `AT_` thread, returned once its metadata and database exist. `attributes` maps
        strings to strings.
        """
    async def get_thread(self, thread_id: str) -> ThreadInfo:
        """Unknown, deleted or expired threads raise `ThreadNotFound`."""
    async def list_threads(
        self,
        scope: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> ThreadPage:
        """The threads of one scope, newest first; `before` is the previous page's cursor."""
    async def update_thread(
        self,
        thread_id: str,
        *,
        attributes: dict[str, object] | None = None,
        idle_ttl_seconds: int | None = None,
    ) -> ThreadInfo:
        """Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
        the activity boundary that moves expiry.
        """
    async def delete_thread(self, thread_id: str) -> None:
        """Makes the thread unavailable and reclaims its content; repeat-safe."""
    async def get_thread_items(
        self,
        thread_id: str,
        *,
        agent: str | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> ThreadItemPage:
        """The thread's chat items, oldest first, of one agent's lane when `agent` is given."""
    async def get_thread_tasks(
        self,
        thread_id: str,
        *,
        status: TaskState | None = None,
    ) -> list[ThreadTask]:
        """The thread's A2A tasks, of one `status` when given."""
    async def open(self, thread_id: str, holder: str) -> Thread:
        """The existing thread's handle, its writes under `holder`. Takes no lease."""
    async def streams(self) -> Streams:
        """The storage's response streams, shared by its threads."""
    async def aclose(self) -> None:
        """Releases backend resources once open handles settled; repeat-safe."""
