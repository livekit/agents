"""The voice door's lane (plan 3.3.1, 3.3.7): what a ``LiveSession`` does with its storage.

A ``LiveSession`` keeps the call's history in memory. Given a thread, ``LiveLane`` opens the
agent's lane at ``start()``, takes the lease per voice turn, from the end of the person's turn to the commit of
the reply, and releases it without draining: a text that landed in the inbox meanwhile is
answered by the worker waiting on it. When an acquire returns a newer version than the one last
committed, the lane moved under the call (a text turn ran) and the session's history and state
are reloaded once, so the call knows about the text; the reply waits on that acquire before the
model reads the history. Storage failures are logged and never end the call.

The run is the two statements of plan 5.2 over ``_core.ThreadDb``: the acquire that takes the
lease and loads the lane, and the fenced commit of what the call added.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, JsonValue

from .. import utils
from .._core import Lease, ThreadDb
from .._generated.storage import AcquireRequest, CommitRequest, EventWrite, Fence, ItemWrite, Loaded
from ..llm import ChatContext, ChatItem
from ..log import logger
from .agentdb import AgentDBStorage
from .protocol import StorageError, dump_state, history_of, item_to_payload, load_state

T = TypeVar("T")

HOLD_LIMIT = 30.0
"""A turn that produced no reply (a skipped turn) still releases the lane within this long."""
LEASE_WAIT = 60.0
"""Plan 3.3.2: a turn that has not taken the lane after this long runs uncommitted."""
TURN_ITEMS = ("message", "function_call", "function_call_output")
"""What a reload hands the agent's history: the turns, not the records only the journal keeps."""


@dataclass
class _Held:
    """A lane taken for one voice turn: what was loaded, and the lease that holds it."""

    db: ThreadDb
    run_id: str
    token: str
    lease: Lease
    loaded: Loaded


class LiveLane:
    """One call's lane: opened at ``start()``, a run per voice turn, closed with the session."""

    def __init__(
        self,
        storage: AgentDBStorage,
        thread_id: str,
        *,
        agent_name: str,
        history: ChatContext,
        state: object,
    ) -> None:
        self._storage = storage
        self._thread_id = thread_id
        self._agent_name = agent_name
        self._history = history
        self._state = state
        self._state_type: type[object] | None = type(state) if state is not None else None
        # the holder is this lane, not the process: a second call of one worker on the same
        # lane waits like a call from any other worker
        self._holder = f"{storage.holder}/{utils.shortuuid()}"
        self._db: ThreadDb | None = None
        self._version: int | None = None
        self._held: _Held | None = None
        self._chain: asyncio.Task[object] | None = None
        self._hold_timer: asyncio.TimerHandle | None = None
        self._closed = False

    async def open(self) -> object:
        """Loads the lane: the stored history when the session has none, the state in place.

        Returns the state object the session should hold: the one it was given, updated from
        the lane, or the lane's when it was given none.
        """
        self._db = await self._storage.open(self._thread_id)
        row = await self._db.lane(self._agent_name)
        self._version = row.version if row is not None else 0

        if not self._history.items:
            self._history.items = history_of(await self._db.history(self._agent_name)).items

        if row is not None and row.state is not None and self._state_type is not None:
            loaded = load_state(row.state, self._state_type)
            self._state = _assign_state(self._state, loaded)

        return self._state

    def turn_started(self) -> Awaitable[ChatContext | None]:
        """The person's turn ended: take the lane for the reply.

        Awaited before the reply is generated, so the model reads a history the lane has
        settled: the result is the stored history when the lane moved under the call and the
        session's history and state were reloaded from it, else None.
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
            if self._db is not None:
                self._db.close()

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
        if self._held is not None:
            return None

        try:
            held = await self._take()
        except StorageError as e:
            logger.warning(
                "the voice turn could not take its lane; the turn runs uncommitted",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            return None
        if held is None:
            return None

        moved = self._version is not None and held.loaded.version != self._version
        if moved:
            self._reload_from(held.loaded)

        self._version = held.loaded.version
        self._held = held
        self._hold_timer = asyncio.get_running_loop().call_later(HOLD_LIMIT, self.turn_ended)
        return history_of(held.loaded.items) if moved else None

    async def _take(self) -> _Held | None:
        """The acquire of plan 3.3.2, waited for with backoff; ``None`` for a run id that
        already committed."""
        if self._db is None:
            self._db = await self._storage.open(self._thread_id)
        db = self._db
        token = utils.shortuuid("lease_")
        run_id = utils.shortuuid("voice_")
        request = AcquireRequest(
            lane=self._agent_name,
            holder=self._holder,
            lease_token=token,
            run_id=run_id,
            ts_ms=_now_ms(),
            run_started={"task_id": run_id, "stream_id": run_id, "channel": "voice"},
        )

        deadline = time.monotonic() + LEASE_WAIT
        delay = 0.1
        while (loaded := await db.acquire(request)) is None:
            if time.monotonic() >= deadline:
                raise StorageError(f"lane {self._agent_name} was not released within {LEASE_WAIT}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)
            request.ts_ms = _now_ms()

        if loaded.already_committed:
            await db.release(self._agent_name, token)
            return None

        lease = db.start_renewal(self._agent_name, self._holder, token)
        return _Held(db=db, run_id=run_id, token=token, lease=lease, loaded=loaded)

    def _reload_from(self, loaded: Loaded) -> None:
        """The lane moved under the call (a text turn ran): take its history and state."""
        merge_history(self._history, history_of(loaded.items))
        if self._state_type is not None and loaded.lane.state is not None:
            self._state = _assign_state(
                self._state, load_state(loaded.lane.state, self._state_type)
            )
        logger.info(
            "the lane moved under the call; history and state reloaded",
            extra={
                "threadId": self._thread_id,
                "lane": self._agent_name,
                "version": loaded.version,
            },
        )

    async def _commit(self) -> None:
        if self._held is None:
            await self._acquire()
            if self._held is None:
                return

        held = self._held
        self._held = None
        self._cancel_hold_timer()
        held.lease.stop()

        try:
            result = await held.db.commit(self._commit_request(held))
        except StorageError as e:
            logger.warning(
                "the voice turn's commit was not written",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            return

        self._version = result.version

    def _commit_request(self, held: _Held) -> CommitRequest:
        """What the call added since the acquire: changed items, deleted ones, the state."""
        loaded = {row.id: _canonical(row.payload) for row in held.loaded.items}
        message_ids = {row.id: row.message_id for row in held.loaded.items if row.message_id}
        items: list[ItemWrite] = []
        kept: set[str] = set()

        for item in self._history.items:
            if not _stored(item):
                continue
            kept.add(item.id)
            payload = item_to_payload(item)
            if loaded.get(item.id) == _canonical(payload):
                continue
            items.append(
                ItemWrite(id=item.id, message_id=message_ids.get(item.id), payload=payload)
            )

        state: JsonValue = None
        if self._state_type is not None:
            dumped = dump_state(self._state, self._state_type)
            if dumped != held.loaded.lane.state:
                state = dumped

        return CommitRequest(
            lane=self._agent_name,
            run_id=held.run_id,
            fence=Fence(version=held.loaded.version, lease_token=held.token),
            items=items,
            delete_items=[item_id for item_id in loaded if item_id not in kept],
            events=[
                EventWrite(
                    ts_ms=_now_ms(),
                    type="conversation_item_added",
                    payload={"item": write.payload},
                )
                for write in items
                if write.id not in loaded
            ],
            state=state,
            run_ended=EventWrite(
                ts_ms=_now_ms(),
                type="run_ended",
                payload={"outcome": "completed", "reason": None},
            ),
        )

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _canonical(payload: JsonValue) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stored(item: ChatItem) -> bool:
    """Instructions are rendered per run, never stored (plan 5.1); ``agent_config_update`` is
    a runtime note to the model, not a chat item of the extension."""
    if item.type == "agent_config_update":
        return False
    return item.type != "message" or item.role not in ("system", "developer")


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


def _assign_state(target: object, loaded: object) -> object:
    """Copies ``loaded`` into ``target`` in place when they share a type, else ``loaded``."""
    if target is None or loaded is None or type(target) is not type(loaded):
        return loaded if loaded is not None else target

    if dataclasses.is_dataclass(target) and not isinstance(target, type):
        for field in dataclasses.fields(target):
            setattr(target, field.name, getattr(loaded, field.name))
        return target

    if isinstance(target, BaseModel):
        for name in type(target).model_fields:
            setattr(target, name, getattr(loaded, name))
        return target

    if isinstance(target, dict) and isinstance(loaded, dict):
        target.clear()
        target.update(loaded)
        return target

    return loaded


__all__ = ["HOLD_LIMIT", "LEASE_WAIT", "TURN_ITEMS", "LiveLane", "merge_history"]
