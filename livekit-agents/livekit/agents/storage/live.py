"""The voice door's lane (plan 3.3.1, 3.3.7): what a ``LiveSession`` does with its storage.

When the lane's text door is ``live`` (``Thread.call_holds_lane``) the call holds the lane for
the whole call, and the texts that land meanwhile are one turn of it. Otherwise it takes the lane
per voice turn, and a text turn between voice turns reaches the call at its next acquire.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, JsonValue

from .. import utils
from .._core import Lease, ThreadDb
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
    TaskRow,
    TaskWrite,
)
from .._generated.storage_run import TaskRecord
from ..llm import ChatContext, ChatItem, FunctionCallOutput
from ..log import logger
from .agentdb import AgentDBStorage
from .protocol import StorageError, dump_state, history_of, item_to_payload, load_state

T = TypeVar("T")
TextAnswer = tuple[str | None, "asyncio.Future[str | None] | None", bool]
"""What the call said to texts, what it says when work their turn let go on ends, and whether
their reply was cut off."""

INBOX_POLL = 0.25
"""Under ``live``, how often the call reads its lane's inbox for texts."""
HOLD_LIMIT = 30.0
"""A turn that produced no reply (a skipped turn) still releases the lane within this long."""
LEASE_WAIT = 60.0
"""Plan 3.3.2: a turn that has not taken the lane after this long runs uncommitted."""
TURN_ITEMS = ("message", "function_call", "function_call_output")
"""What a reload hands the agent's history: the turns, not the records only the journal keeps."""


@dataclass
class OpenQuestion:
    """A question the call leaves open: the lane keeps it for the next door to answer."""

    question: str
    verbatim: bool
    output: FunctionCallOutput | None
    """The output of the call that asked."""
    answer_ref: str | None
    """The expert task the answer continues, for a delegate's question."""


class LaneQuestion(BaseModel):
    """A question open on the lane when the call took it, as its task's ``ask`` holds it."""

    row: TaskRow
    question: str
    verbatim: bool = False
    call_id: str = ""
    tool: str = ""
    answer_to: list[tuple[str, str]] = []
    """The tools the answer goes to, each ``(tool, answer_ref)``."""

    @property
    def answer_ref(self) -> str | None:
        """The expert task the answer continues: a call hands it to the first tool only."""
        return self.answer_to[0][1] if self.answer_to else None


@dataclass
class _Held:
    """A lane taken for one voice turn: what was loaded, and the lease that holds it."""

    db: ThreadDb
    run_id: str
    token: str
    lease: Lease
    loaded: Loaded


class LiveLane:
    """One call's lane: opened at ``start()``, a run per voice turn, closed with the session.
    Storage failures are logged and never end the call."""

    def __init__(
        self,
        storage: AgentDBStorage,
        thread_id: str,
        *,
        agent_name: str,
        history: ChatContext,
        state: object,
        live: bool = False,
        on_texts: Callable[[list[InboxRow]], Awaitable[TextAnswer]] | None = None,
    ) -> None:
        self._storage = storage
        self._live = live
        self._on_texts = on_texts
        self._inbox_task: asyncio.Task[None] | None = None
        # the texts whose turn let work go on, until it ends: what the call said, and the wait
        self._waiting: dict[str, tuple[InboxRow, str | None, asyncio.Future[str | None]]] = {}
        self._answering: set[asyncio.Task[None]] = set()
        # the lane's open questions at the start, and those of them the call answered since
        self._questions: list[LaneQuestion] = []
        self._answered: dict[str, TaskRow] = {}
        # failures nothing awaited, for the journal of the next commit
        self._errors: list[str] = []
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
        self._known: set[str] = set()
        """The stored items the call has held: a commit deletes only those it dropped since."""
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

        self._known = {item.id for item in self._history.items if _stored(item)}
        rows = await self._db.tasks(self._agent_name, status="input-required")
        self._questions = [
            LaneQuestion.model_validate({**task.internal["ask"], "row": task})
            for task in reversed(rows)
            if isinstance(task.internal, dict) and isinstance(task.internal.get("ask"), dict)
        ]
        if self._live:
            await self._acquire()
            self._inbox_task = asyncio.create_task(self._follow_inbox(), name="LiveLane.inbox")
        return self._state

    def take_questions(self) -> list[LaneQuestion]:
        """The lane's open questions, oldest first, once: they are the call's to answer."""
        questions, self._questions = self._questions, []
        return questions

    def question_answered(self, row: TaskRow) -> None:
        """The call answered a question of the lane: its task completes at the next commit."""
        self._answered[row.task_id] = row

    def error(self, text: str) -> None:
        """A failure on the call that nothing awaited: an ``error`` event of the next commit."""
        self._errors.append(text)

    def turn_started(self) -> Awaitable[ChatContext | None]:
        """The person's turn ended: take the lane for the reply.

        Awaited before the reply is generated, so the model reads a history the lane has
        settled: the result is the stored history when the lane moved under the call and the
        session's history and state were reloaded from it, else None.
        """
        return self._schedule(self._acquire())

    def turn_ended(self) -> None:
        """The reply is done: commit what the call added; release without draining, or under
        ``live`` keep the lane."""
        self._schedule(self._commit())

    async def close(self, questions: Sequence[OpenQuestion] = ()) -> None:
        """The call ended: what it added is committed and the lane let go; ``questions``, the
        ones it leaves open, stay on the lane as tasks waiting on the person."""
        if self._closed:
            return
        self._closed = True
        # a text still waiting on its turn's work ends with what the call said to it
        waiting, self._waiting = self._waiting, {}
        for _, _, told in waiting.values():
            told.cancel()
        if self._inbox_task is not None:
            await utils.aio.cancel_and_wait(self._inbox_task)

        if self._chain is not None:
            with contextlib.suppress(Exception):
                await self._chain
        await asyncio.gather(*self._answering, return_exceptions=True)
        ended = [
            (_answered(row, said, background=False), said) for row, said, _ in waiting.values()
        ]

        left: list[TaskWrite] = []
        for question in questions:
            try:
                # the question's task is also the answer stream a text continues it on
                stream = await self._storage.streams.create()
            except StorageError as e:
                logger.warning("a question the call leaves open is lost", extra={"error": str(e)})
                continue
            left.append(_left_open(question, stream.stream_id))
        try:
            if await self._commit(release=True, tasks=[*left, *(write for write, _ in ended)]):
                for write, said in ended:
                    await self._write_answers([write], said)
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

    async def _follow_inbox(self) -> None:
        """Under ``live``: the texts that landed in the lane's inbox are one turn of the call.
        The last one's task carries what the call said, the earlier ones are merged into it, and
        the commit moves the inbox cursor past them."""
        while True:
            await asyncio.sleep(INBOX_POLL)
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
                reply, told, cut_off = await self._on_texts(rows)
            except Exception:
                logger.exception("the call could not answer a text")
                reply, told, cut_off = None, None, False
            last = rows[-1]
            writes = [_answered(row, None, merged_into=last.task_id) for row in rows[:-1]]
            # as in text, the last one stays working while work its turn let go on runs
            writes.append(
                _answered(
                    last,
                    reply,
                    background=True if told is not None else None,
                    interrupted=cut_off,
                )
            )
            if told is not None:
                # from here a close ends the text
                self._waiting[last.task_id] = (last, reply, told)
            # shielded: closing the call stops the follow, never a commit in flight
            if await asyncio.shield(self._schedule(self._commit(tasks=writes, drained=last.id))):
                await self._write_answers(writes, reply)
            if told is not None and last.task_id in self._waiting:
                task = asyncio.create_task(self._answer_told(last.task_id))
                self._answering.add(task)
                task.add_done_callback(self._answering.discard)

    async def _answer_told(self, task_id: str) -> None:
        """A text whose turn let work go on ends with what the call says when that work ends,
        else with what it said in the turn."""
        row, said, told = self._waiting[task_id]
        try:
            reply = await told or said
        except Exception:
            logger.exception("the call could not tell a text the end of its work")
            reply = said
        if self._waiting.pop(task_id, None) is None:
            return  # the call ended first, and its close answered the text
        write = _answered(row, reply, background=False)
        if await asyncio.shield(self._schedule(self._commit(tasks=[write]))):
            await self._write_answers([write], reply)

    async def _write_answers(self, writes: list[TaskWrite], reply: str | None) -> None:
        """The texts' answer streams: the reply on the last one's, each ended unless it waits on
        work that runs on."""
        from ..a2a.codec import StreamEvent, artifact_event, final_event, to_stream_response
        from ..a2a.extension import Extensions, to_json

        streams = self._storage.streams
        exts = Extensions(livekit=True)
        for write in writes:
            frames: list[StreamEvent] = []
            if reply and write is writes[-1] and write.status != "working":
                art = utils.shortuuid("art_")
                frames.append(
                    artifact_event(
                        write.task_id,
                        self._thread_id,
                        artifact_id=art,
                        text=reply,
                        append=False,
                        last_chunk=True,
                        exts=exts,
                    )
                )
            record = TaskRecord(
                id=write.task_id,
                lane=self._agent_name,
                seq=0,
                name=write.name,
                message_id=write.message_id,
                status=write.status,
                status_message=write.status_message,
                updated_at=int(time.time()),
                data=write.data,
                internal=write.internal,
            )
            frames.append(final_event(record, self._thread_id, exts=exts))
            for frame in frames:
                body = json.dumps(to_json(to_stream_response(frame)), separators=(",", ":"))
                streams.append(write.task_id, body.encode())
            if write.status == "working":
                continue
            try:
                await streams.close(write.task_id)
            except StorageError as e:
                logger.info("a text's answer stream could not be closed", extra={"error": str(e)})

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
        self._known |= {item.id for item in self._history.items if _stored(item)}
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
        tasks: Sequence[TaskWrite] = (),
        drained: int | None = None,
    ) -> bool:
        """Commits what the call added, with ``tasks`` and, under ``live``, the inbox cursor at
        ``drained``; ``True`` once written. Under ``live`` the call keeps the lane until
        ``release``."""
        if self._held is None:
            await self._acquire()
            if self._held is None:
                return False

        held = self._held
        keep = self._live and not release
        if not keep:
            self._held = None
            self._cancel_hold_timer()
            held.lease.stop()

        request = self._commit_request(held)
        errors = len(self._errors)
        request.events += [
            EventWrite(ts_ms=_now_ms(), type="error", payload={"error": text})
            for text in self._errors
        ]
        answered = dict(self._answered)
        request.tasks = [*tasks, *(_completed(row) for row in answered.values())]
        request.drained_id = drained
        request.after = "renew" if keep else "release"
        try:
            result = await held.db.commit(request)
        except StorageError as e:
            logger.warning(
                "the voice turn's commit was not written",
                extra={"threadId": self._thread_id, "lane": self._agent_name, "error": str(e)},
            )
            if keep:
                # the lane is lost to the call: the next turn takes it again
                self._held = None
                held.lease.stop()
            return False

        self._version = result.version
        self._known = {item.id for item in self._history.items if _stored(item)}
        del self._errors[:errors]
        for task_id in answered:
            self._answered.pop(task_id, None)
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
            # what another writer added is not the call's to drop
            delete_items=[i for i in loaded if i in self._known and i not in kept],
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


def _answered(
    row: InboxRow,
    reply: str | None,
    *,
    merged_into: str | None = None,
    background: bool | None = None,
    interrupted: bool = False,
) -> TaskWrite:
    """A text the call answered: its task ``completed`` with the reply, merged into the text
    that carried it, ``canceled`` with what was said as text ends a reply the person cut off, or
    with ``background`` ``working`` while work its turn let go on runs and ``completed`` once it
    ended."""
    from ..a2a.extension import (
        ANSWERED_BY_VOICE,
        BACKGROUND,
        INTERRUPTED,
        INTERRUPTION_REASON,
        MERGED_INTO,
    )

    metadata: dict[str, JsonValue] = {ANSWERED_BY_VOICE: True}
    status_message: JsonValue = None
    artifacts: list[JsonValue] | None = None
    if background is not None:
        metadata[BACKGROUND] = background
    if interrupted:
        metadata[INTERRUPTED] = True
        metadata[INTERRUPTION_REASON] = "new-message"
    if merged_into is not None:
        metadata[MERGED_INTO] = merged_into
    elif reply:
        parts: list[JsonValue] = [{"kind": "text", "text": reply}]
        if not interrupted:
            status_message = {
                "messageId": utils.shortuuid("msg_"),
                "role": "agent",
                "parts": parts,
                "taskId": row.task_id,
            }
        if not background:
            artifacts = [{"artifactId": utils.shortuuid("art_"), "parts": parts}]
    return TaskWrite(
        task_id=row.task_id,
        status="working" if background else "canceled" if interrupted else "completed",
        message_id=row.message_id,
        status_message=status_message,
        data=TaskData(history=[row.message], artifacts=artifacts, metadata=metadata),
    )


def _completed(row: TaskRow) -> TaskWrite:
    """A question of the lane the call answered: its task completes as it stands."""
    return TaskWrite(
        task_id=row.task_id,
        status="completed",
        message_id=row.message_id,
        name=row.name,
        status_message=row.status_message,
        data=TaskData.model_validate(row.data or {}),
    )


def _left_open(question: OpenQuestion, task_id: str) -> TaskWrite:
    """A question the call leaves open: a task of the lane waiting on the person, with the ask
    a text turn reads to send the answer on."""
    ask: dict[str, JsonValue] = {"question": question.question}
    if question.output is not None:
        ask["tool"] = question.output.name
        ask["call_id"] = question.output.call_id
        if question.answer_ref is not None:
            ask["answer_to"] = [[question.output.name, question.answer_ref]]
    if question.verbatim:
        ask["verbatim"] = True
    return TaskWrite(
        task_id=task_id,
        status="input-required",
        status_message={
            "messageId": utils.shortuuid("msg_"),
            "role": "agent",
            "parts": [{"kind": "text", "text": question.question}],
            "taskId": task_id,
        },
        data=TaskData(),
        internal=TaskInternal(ask=ask),
    )


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


__all__ = [
    "HOLD_LIMIT",
    "INBOX_POLL",
    "LEASE_WAIT",
    "TURN_ITEMS",
    "LaneQuestion",
    "LiveLane",
    "OpenQuestion",
    "merge_history",
]
