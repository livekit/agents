"""The voice door's lane (plan 3.3.1, 3.3.7): what a ``VoiceSession`` does with its storage.

A ``VoiceSession`` keeps the call's history in memory. Given a thread, ``LiveLane`` opens the
agent's lane at ``start()``, takes the lease per voice turn, from the end of the person's turn to the commit of
the reply, and releases it without draining: a text that landed in the inbox meanwhile is
answered by the worker waiting on it. When an acquire returns a newer version than the one last
committed, the lane moved under the call (a text turn ran) and the session's history
is reloaded once, so the call knows about the text; the reply waits on that acquire before the
model reads the history. Storage failures are logged and never end the call.

The run is ``_core``'s: ``Thread.open_run`` takes the lease and loads the lane, ``Run.commit``
writes what the call added, fenced on the version loaded.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Coroutine
from typing import TypeVar

from .. import utils
from .._core import Run, Thread
from .._generated.storage_class import ItemRow
from ..llm import ChatContext
from ..log import logger
from .backends import Storage
from .context_store import ContextStore
from .protocol import StorageError, VersionConflict, history_of, item_to_payload

T = TypeVar("T")

HOLD_LIMIT = 30.0
"""A turn that produced no reply (a skipped turn) still releases the lane within this long."""
LEASE_WAIT = 60.0
"""Plan 3.3.2: a turn that has not taken the lane after this long runs uncommitted."""
TURN_ITEMS = ("message", "function_call", "function_call_output")
"""What a reload hands the agent's history: the turns, not the records only the journal keeps."""


class LiveLane:
    """One call's lane: opened at ``start()``, a run per voice turn, closed with the session."""

    def __init__(
        self,
        storage: Storage,
        thread_id: str,
        *,
        agent_name: str,
        history: ChatContext,
    ) -> None:
        self._storage = storage
        self._thread_id = thread_id
        self._agent_name = agent_name
        self._history = history
        self._store = ContextStore()
        self._thread: Thread | None = None
        self._version: int | None = None
        self._run: Run | None = None
        self._chain: asyncio.Task[object] | None = None
        self._hold_timer: asyncio.TimerHandle | None = None
        self._closed = False

    async def open(self) -> None:
        """Load history and the field snapshot used by explicit context operations."""
        self._thread = await self._storage.open(self._thread_id)
        row = await self._thread.lane_info(self._agent_name)
        self._version = row.version if row is not None else 0

        if not self._history.items:
            self._history.items = history_of(await self._thread.history(self._agent_name)).items

        self._store._rebase(row.state if row is not None else None)

    @property
    def store(self) -> ContextStore:
        return self._store

    def turn_started(self) -> Awaitable[ChatContext | None]:
        """The person's turn ended: take the lane for the reply.

        Awaited before the reply is generated, so the model reads a history the lane has
        settled: the result is the stored history when the lane moved under the call and the
        session's history was reloaded from it, else None.
        """
        return self._schedule(self._acquire())

    def turn_ended(self) -> None:
        """The reply is done: commit what the call added and release, without draining."""
        self._schedule(self._commit())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._chain is not None:
            with contextlib.suppress(Exception):
                await self._chain

        try:
            await self._commit()
        finally:
            self._store.close()
            if self._thread is not None:
                self._thread.close()

    def _schedule(self, coro: Coroutine[object, object, T]) -> asyncio.Future[T | None]:
        """Runs ``coro`` after whatever was scheduled before it; ``None`` once closed."""
        if self._closed:
            coro.close()
            done: asyncio.Future[T | None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done

        previous = self._chain

        async def after() -> T | None:
            if previous is not None:
                with contextlib.suppress(Exception):
                    await previous
            return await coro

        task = asyncio.create_task(after(), name="LiveLane")
        self._chain = task
        return task

    # -- the run --

    async def _acquire(self) -> ChatContext | None:
        if self._run is not None:
            return None

        try:
            run = await self._take()
        except StorageError as e:
            logger.warning(
                "the voice turn could not take its lane; the turn runs uncommitted",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            return None
        if run is None:
            return None

        try:
            self._store._rebase(run.state)
        except Exception:
            await run.abandon("failed", "context-conflict")
            raise

        moved = self._version is not None and run.version != self._version
        if moved:
            self._reload_from(run)

        self._version = run.version
        self._run = run
        self._hold_timer = asyncio.get_running_loop().call_later(HOLD_LIMIT, self.turn_ended)
        return history_of(run.history) if moved else None

    async def _take(self) -> Run | None:
        """The lane for one voice turn, waited for; ``None`` for a run id that already
        committed."""
        if self._thread is None:
            self._thread = await self._storage.open(self._thread_id)
        run = await self._thread.open_run(
            self._agent_name, utils.shortuuid("voice_"), channel="voice", wait_timeout=LEASE_WAIT
        )
        if run.closed:
            await run.commit(run.history)
            return None
        return run

    def _reload_from(self, run: Run) -> None:
        """The lane moved under the call (a text turn ran): take its history."""
        merge_history(self._history, history_of(run.history))
        logger.info(
            "the lane moved under the call; history reloaded",
            extra={"threadId": self._thread_id, "lane": self._agent_name, "version": run.version},
        )

    async def _commit(self) -> None:
        if self._run is None:
            await self._acquire()
            if self._run is None:
                return

        run = self._run
        self._run = None
        self._cancel_hold_timer()

        # the A2A `messageId` an item came from stays with it
        message_ids = {item.id: item.message_id for item in run.history}
        history = [
            ItemRow(id=item.id, message_id=message_ids.get(item.id), payload=item_to_payload(item))
            for item in self._history.items
        ]
        state = self._store._pending_state
        try:
            self._version = await run.commit(history, state)
            self._store._acknowledge(state)
        except StorageError as e:
            logger.warning(
                "the voice turn's commit was not written",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            if isinstance(e, VersionConflict):
                raise

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None


def merge_history(history: ChatContext, stored: ChatContext, *, turns_only: bool = False) -> None:
    """What ``stored`` has that ``history`` does not, inserted in place by creation time.

    ``turns_only`` keeps the journal's own records (handoffs, configuration) out of an agent's
    working history.
    """
    present = {item.id for item in history.items}
    missing = [item for item in stored.items if item.id not in present]
    if turns_only:
        missing = [item for item in missing if item.type in TURN_ITEMS]

    history.insert(missing)


__all__ = ["HOLD_LIMIT", "LEASE_WAIT", "TURN_ITEMS", "LiveLane", "merge_history"]
