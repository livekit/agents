"""The voice door's lane (plan 3.3.1, 3.3.7): what a ``LiveSession`` does with its storage.

A ``LiveSession`` keeps the call's history in memory. Given a thread, ``LiveLane`` opens the
agent's lane at ``start()``. Under ``live`` (the default, docs/delegation.md) the call holds the
lane for the whole call: it takes the lease at ``open()``, renews it until ``close()``, commits
each reply without letting go, and drains the lane's inbox itself, so a text that arrives during
the call is a turn of the call, which the call's model sees at once and answers on the text's
task. Under ``turn`` it takes the lease per voice turn, from the end of the person's turn to the
commit of the reply, and releases it without draining: a text that landed in the inbox
meanwhile is answered by the worker waiting on it; when an acquire returns a newer version than
the one last committed, the lane moved under the call (a text turn ran) and the session's
history and state are reloaded once, before the model reads the history. Either way the commit
deletes only what the call itself removed. Storage failures are logged and never end the call.

The run is the two statements of plan 5.2 over ``_core.ThreadDb``: the acquire that takes the
lease and loads the lane, and the fenced commit of what the call added.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, JsonValue

from .. import utils
from .._core import Lease, Streams, ThreadDb
from .._generated.storage import (
    AcquireRequest,
    CommitRequest,
    EventWrite,
    Fence,
    InboxRow,
    ItemWrite,
    Loaded,
    TaskData,
    TaskInternal,
    TaskWrite,
)
from .._generated.storage_run import TaskRecord
from ..llm import ChatContext, ChatItem, FunctionCallOutput
from ..log import logger
from . import hub
from .agentdb import AgentDBStorage
from .protocol import StorageError, dump_state, history_of, item_to_payload, load_state

T = TypeVar("T")

OpenQuestion = tuple[FunctionCallOutput | None, str, str | None, bool, bool]
"""A question the call leaves open: the call's output, the question, the tool that takes the
answer, verbatim, asked."""

INBOX_BACKSTOP = 2.0
"""Under ``live``, how often the call reads its lane's inbox when nothing told it of a text:
a text door of the call's own worker tells it at once, through IPC; one on another worker
does not."""
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
        live: bool = False,
        on_texts: Callable[[list[InboxRow]], Awaitable[str | None]] | None = None,
    ) -> None:
        self._storage = storage
        self._live = live
        self._on_texts = on_texts
        # the lane's open questions at the start, and the ones the call answered since
        self._questions: dict[str, TaskRecord] = {}
        self._answered: list[str] = []
        self._inbox_task: asyncio.Task[None] | None = None
        # the items the call holds as of its last load or commit: the only ones it may delete
        self._known: set[str] = set()
        self._thread_id = thread_id
        self._agent_name = agent_name
        self._history = history
        self._state = state
        self._state_type: type[object] | None = type(state) if state is not None else None
        # the holder is this lane, not the process: a second call of one worker on the same
        # lane waits like a call from any other worker
        self._holder = f"{storage.process_holder}/{utils.shortuuid()}"
        self._db: ThreadDb | None = None
        self._version: int | None = None
        self._held: _Held | None = None
        self._chain: asyncio.Task[object] | None = None
        self._hold_timer: asyncio.TimerHandle | None = None
        self._closed = False
        self._pending_questions: dict[str, TaskRecord] = {}
        self._hub: hub.LaneHub | None = None
        # failures nothing awaited, for the journal of the next commit
        self._errors: list[str] = []

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

        self._known = {item.id for item in self._history.items}
        tasks = await self._db.tasks(self._agent_name, status="input-required", limit=50)
        self._questions = {
            row.task_id: TaskRecord(
                id=row.task_id,
                lane=row.lane,
                seq=row.seq,
                name=row.name,
                message_id=row.message_id,
                status=row.status,
                status_message=row.status_message,
                updated_at=row.updated_at,
                data=TaskData.model_validate(row.data or {}),
                internal=TaskInternal.model_validate(row.internal) if row.internal else None,
            )
            for row in reversed(tasks)
        }
        if self._live:
            try:
                await self._acquire()
            except StorageError as e:
                logger.warning(
                    "the call could not take its lane; it runs uncommitted",
                    extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
                )
            # a text door of this worker tells the call of a text at once
            self._hub = hub.lane_hub(self._thread_id, self._agent_name)
            self._hub.held = True
            hub.announce(self._thread_id, self._agent_name, self._holder, True)
            self._inbox_task = asyncio.create_task(self._follow_inbox(), name="LiveLane.inbox")
        return self._state

    def turn_started(self) -> Awaitable[ChatContext | None]:
        """The person's turn ended: take the lane for the reply (under ``live`` the call holds
        it already).

        Awaited before the reply is generated, so the model reads a history the lane has
        settled: the result is the stored history when the lane moved under the call and the
        session's history and state were reloaded from it, else None.
        """
        return self._schedule(self._acquire())

    async def record(self, item: ChatItem) -> None:
        """The person's message on the lane before the commit, for an expert that reads it
        while the call still works (docs/delegation.md): under the call's lease and version,
        without moving the version. The commit writes it again."""
        held = self._held
        if held is None or not _stored(item):
            return
        try:
            await held.db.write_items(
                self._agent_name,
                Fence(version=held.loaded.version, lease_token=held.token),
                [
                    ItemWrite(
                        id=item.id, message_id=None, payload=item_to_payload(item)
                    ).model_dump()
                ],
            )
        except StorageError as e:
            logger.warning(
                "the person's message could not be written ahead of the delegation",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )

    def take_questions(self) -> list[TaskRecord]:
        """The lane's open questions, oldest first, once: the call puts them to the person."""
        questions, self._questions = list(self._questions.values()), {}
        self._pending_questions = {q.id: q for q in questions}
        return questions

    def question_answered(self, task_id: str) -> None:
        """The call answered a question of the lane: its task completes at the next commit."""
        self._answered.append(task_id)

    def error(self, text: str) -> None:
        """A failure on the call that nothing awaited: an ``error`` event of the next commit."""
        self._errors.append(text)

    def turn_ended(self) -> None:
        """The reply is done: commit what the call added; under ``turn`` release, without
        draining, and under ``live`` keep the lane."""
        self._schedule(self._commit())

    async def close(self, *, questions: list[OpenQuestion] | None = None) -> None:
        """The call ended: what it added is committed, the lane let go, and ``questions``, the
        ones it leaves open, stay on the lane for the next door."""
        if self._closed:
            return
        self._closed = True
        if self._inbox_task is not None:
            await utils.aio.cancel_and_wait(self._inbox_task)
        if self._hub is not None:
            hub.drop(self._thread_id, self._agent_name)
            hub.announce(self._thread_id, self._agent_name, self._holder, False)

        if self._chain is not None:
            with contextlib.suppress(Exception):
                await self._chain

        try:
            await self._commit(release=True, tasks=_questions_left(questions or []))
        finally:
            if self._db is not None:
                self._db.close()

    async def _follow_inbox(self) -> None:
        """Under ``live``: a text that lands in the lane's inbox during the call is a turn of
        the call. The texts that waited together are one turn; the last one's task carries the
        reply, the earlier ones are merged into it, and the commit moves the inbox cursor."""
        assert self._hub is not None
        seen = self._hub.signals
        while True:
            await self._hub.since(seen, INBOX_BACKSTOP)
            seen = self._hub.signals
            if self._held is None or self._on_texts is None or self._db is None:
                continue
            try:
                rows = await self._db.inbox_pending(self._agent_name)
                for row in rows:
                    await self._db.pickup_task(self._agent_name, row.task_id)
            except StorageError as e:
                logger.warning("the call could not read its inbox", extra={"error": str(e)})
                continue
            if not rows:
                continue
            try:
                reply = await self._on_texts(rows)
            except Exception:
                logger.exception("the call could not answer a text")
                reply = None
            last = rows[-1]
            writes = [_answered(row, None, merged_into=last.task_id) for row in rows[:-1]]
            writes.append(_answered(last, reply))
            # shielded: closing the call stops the follow, never a commit in flight
            if await asyncio.shield(self._schedule(self._commit(tasks=writes, drained=last.id))):
                await self._write_answer_frames(writes, reply)
                for write in writes:
                    # the text door that waits on the task hears it, through the worker
                    hub.task_moved(write.task_id, thread_id=self._thread_id, lane=self._agent_name)

    async def _write_answer_frames(self, writes: list[TaskWrite], reply: str | None) -> None:
        """The texts' answer streams: the reply on the last one's, each ended."""
        from ..a2a.codec import StreamEvent, artifact_event, final_event, to_stream_response
        from ..a2a.extension import Extensions, to_json

        streams = self._storage.streams
        for write in writes:
            frames: list[StreamEvent] = []
            record = _record_of(write, self._agent_name)
            if reply and write is writes[-1]:
                frames.append(
                    artifact_event(
                        write.task_id,
                        self._thread_id,
                        artifact_id=utils.shortuuid("art_"),
                        text=reply,
                        append=False,
                        last_chunk=True,
                        exts=Extensions(livekit=True),
                    )
                )
            frames.append(final_event(record, self._thread_id, exts=Extensions(livekit=True)))
            for frame in frames:
                body = json.dumps(to_json(to_stream_response(frame)), separators=(",", ":"))
                streams.append(write.task_id, body.encode())
            await _close_stream(streams, write.task_id)

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
        if self._live and self._closed:
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
        if not self._live:
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

    async def _commit(
        self,
        *,
        release: bool = False,
        tasks: list[TaskWrite] | None = None,
        drained: int | None = None,
    ) -> bool:
        """``tasks`` and ``drained``: under ``live``, the texts the call answered and the inbox
        row they end at. ``True`` when the commit was written."""
        if self._held is None:
            await self._acquire()
            if self._held is None:
                return False

        held = self._held
        # under live the call keeps the lane until it closes
        keep = self._live and not release
        if not keep:
            self._held = None
            self._cancel_hold_timer()
            held.lease.stop()

        request = self._commit_request(held, keep=keep)
        reported = len(self._errors)
        answered = list(self._answered)
        request.tasks = [*(tasks or []), *self._answered_writes(answered)]
        request.drained_id = drained
        try:
            result = await held.db.commit(request)
        except StorageError as e:
            logger.warning(
                "the voice turn's commit was not written",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            if keep:
                # the lane is lost to the call: it takes the lane again for the next commit
                self._held = None
                held.lease.stop()
            return False

        self._version = result.version
        del self._errors[:reported]
        self._answered = [t for t in self._answered if t not in answered]
        self._known = {item.id for item in self._history.items}
        if keep:
            # what the lane holds now is the next commit's baseline
            try:
                held.loaded = held.loaded.model_copy(
                    update={
                        "version": result.version,
                        "items": await held.db.history(self._agent_name),
                        "lane": await held.db.lane(self._agent_name) or held.loaded.lane,
                    }
                )
            except StorageError as e:
                logger.warning("the call could not reload its lane", extra={"error": str(e)})
        return True

    def _commit_request(self, held: _Held, *, keep: bool = False) -> CommitRequest:
        """What the call added since the acquire: changed items, the ones the call removed,
        and the state."""
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
            delete_items=[
                item_id for item_id in loaded if item_id in self._known and item_id not in kept
            ],
            after="renew" if keep else "release",
            events=[
                EventWrite(
                    ts_ms=_now_ms(),
                    type="conversation_item_added",
                    payload={"item": write.payload},
                )
                for write in items
                if write.id not in loaded
            ]
            + [
                EventWrite(ts_ms=_now_ms(), type="error", payload={"error": text})
                for text in self._errors
            ],
            state=state,
            run_ended=EventWrite(
                ts_ms=_now_ms(),
                type="run_ended",
                payload={"outcome": "completed", "reason": None},
            ),
        )

    def _answered_writes(self, task_ids: list[str]) -> list[TaskWrite]:
        """The lane's questions the call answered: their tasks complete."""
        writes = []
        for task_id in task_ids:
            record = self._pending_questions.get(task_id)
            if record is None:
                continue
            writes.append(
                TaskWrite(
                    task_id=task_id,
                    status="completed",
                    message_id=record.message_id,
                    name=record.name,
                    status_message=record.status_message,
                    data=record.data.model_copy(
                        update={"metadata": {**record.data.metadata, "livekit/queued": False}}
                    ),
                )
            )
        return writes

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None


def _answered(row: InboxRow, reply: str | None, *, merged_into: str | None = None) -> TaskWrite:
    """A text the call answered: its task ``completed``, with the reply, or merged into the
    text that carried it."""
    task_id = row.task_id
    metadata: dict[str, JsonValue] = {}
    status_message: JsonValue = None
    artifacts: list[JsonValue] | None = None
    if merged_into is not None:
        metadata["livekit/mergedInto"] = merged_into
    elif reply:
        status_message = {
            "messageId": utils.shortuuid("msg_"),
            "role": "agent",
            "parts": [{"kind": "text", "text": reply}],
            "taskId": task_id,
        }
        artifacts = [
            {"artifactId": utils.shortuuid("art_"), "parts": [{"kind": "text", "text": reply}]}
        ]
    return TaskWrite(
        task_id=task_id,
        status="completed",
        message_id=row.message_id,
        status_message=status_message,
        data=TaskData(history=[row.message], artifacts=artifacts, metadata=metadata),
    )


def _record_of(write: TaskWrite, lane: str) -> TaskRecord:
    return TaskRecord(
        id=write.task_id,
        lane=lane,
        seq=0,
        name=write.name,
        message_id=write.message_id,
        status=write.status,
        status_message=write.status_message,
        updated_at=int(time.time()),
        data=write.data,
        internal=write.internal,
    )


async def _close_stream(streams: Streams, stream_id: str) -> None:
    try:
        await streams.close(stream_id)
    except StorageError as e:
        logger.info("a text's answer stream could not be closed", extra={"error": str(e)})


def _questions_left(questions: list[OpenQuestion]) -> list[TaskWrite]:
    """The questions a call leaves open: each becomes a task of the lane waiting on the
    person, so the next door, a text, answers it."""
    writes = []
    for output, question, tool, verbatim, asked in questions:
        task_id = utils.shortuuid("AST_")
        ask: dict[str, JsonValue] = {"question": question}
        if output is not None:
            ask["tool"] = output.name
            ask["call_id"] = output.call_id
        if tool:
            ask["answer_tool"] = tool
        if verbatim:
            ask["verbatim"] = True
        if not asked:
            ask["queued"] = True
        writes.append(
            TaskWrite(
                task_id=task_id,
                status="input-required",
                status_message={
                    "messageId": utils.shortuuid("msg_"),
                    "role": "agent",
                    "parts": [{"kind": "text", "text": question}],
                    "taskId": task_id,
                },
                data=TaskData(metadata={"livekit/queued": not asked}),
                internal=TaskInternal(ask=ask),
            )
        )
    return writes


def _now_ms() -> int:
    return int(time.time() * 1000)


def _canonical(payload: JsonValue) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stored(item: ChatItem) -> bool:
    """Instructions are rendered per run, never stored (plan 5.1); ``agent_config_update`` is
    a runtime note to the model, not a chat item of the extension; an empty user message is
    no turn."""
    if item.type == "agent_config_update":
        return False
    if item.type == "message" and item.role == "user" and not (item.text_content or "").strip():
        return False  # an empty transcript is no turn
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
