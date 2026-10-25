"""The live response of one message, and its durable answer stream (plan 6).

``WireHost`` is the host object of a wire turn: it runs the handler and turns the turn's calls
into the A2A frames of the response. Every frame goes through the ``ResponsePump``: appended to
the task's stream as it is produced, whether or not a client is still reading, and the stream
is closed at the terminal frame. The turn runs in a task of its own and emits here; the
response generator relays what has already been made durable. A client that goes away leaves
that generator suspended, which changes nothing: the turn runs to its commit, and a
``tasks/resubscribe`` replays the stream from any offset.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from a2a.types import a2a_pb2 as pb
from google.protobuf import json_format
from pydantic import JsonValue

from .._core import Boundary, Streams
from .._generated.storage_run import TaskRecord
from .._generated.storage_turn import Background, HandlerOutput, TurnContext, TurnResult
from ..log import logger
from ..storage import hub
from ..storage.context import EndHook, TokenHook, TurnHost, UpdateHook
from ..storage.protocol import TERMINAL_TASK_STATES, StorageError
from ..utils import shortuuid
from .codec import (
    StreamEvent,
    artifact_event,
    final_event,
    submitted_task,
    to_stream_response,
    working_event,
)
from .extension import Extensions, to_json

if TYPE_CHECKING:
    from .methods import TurnOutcome

_LAST_STATES = frozenset(
    {
        pb.TaskState.TASK_STATE_COMPLETED,
        pb.TaskState.TASK_STATE_FAILED,
        pb.TaskState.TASK_STATE_CANCELED,
        pb.TaskState.TASK_STATE_REJECTED,
        pb.TaskState.TASK_STATE_INPUT_REQUIRED,
    }
)
"""A relayed response ends at these: the task ended, or waits on a question."""

S = TypeVar("S")
T = TypeVar("T")

RELAY_DRAIN = 5.0
"""How long a response waits for the relay of a task another turn ran to reach its last frame,
in seconds."""

running: set[asyncio.Task[None]] = set()
"""The turns in flight, each pumping its stream; a test waits on them before its leak check."""


def start(turn: Coroutine[object, object, None]) -> None:
    """Runs one turn to its end, whatever happens to the response that asked for it."""
    task = asyncio.ensure_future(turn)
    running.add(task)
    task.add_done_callback(running.discard)


class ResponsePump(Generic[T]):
    """The frames of one message's response, in order, each durable before it is relayed, and
    what the turn produced (``T``) once they end."""

    def __init__(self, streams: Streams, stream_id: str) -> None:
        self._streams = streams
        self._stream_id = stream_id
        self._queue: asyncio.Queue[StreamEvent | pb.StreamResponse | BaseException | None] = (
            asyncio.Queue()
        )
        self._outcome: T | None = None
        self.acked: str | None = None
        """The task id the acknowledgment named, once it went out."""
        self.followed_final = False
        """The relayed stream carried the task's last frame."""
        self._following: asyncio.Future[None] | None = None
        self._finished = False
        self._owned = True

    @property
    def outcome(self) -> T:
        """What the turn produced; read once ``frames`` ended without an error."""
        if self._outcome is None:
            raise RuntimeError("the turn has not ended")
        return self._outcome

    def emit(self, event: StreamEvent) -> None:
        """One frame: appended to the stream (queued, sent in order, a failure is logged and does
        not fail the turn), then relayed."""
        self.append(self._stream_id, event)
        self._queue.put_nowait(event)

    def show(self, event: StreamEvent) -> None:
        """One frame to the response only: the stream has it already."""
        self._queue.put_nowait(event)

    def append(self, stream_id: str, event: StreamEvent) -> None:
        """One frame to a task's answer stream only: a task this response does not answer. A
        request of this process that follows the stream gets it at once too."""
        frame = to_stream_response(event)
        record = json.dumps(to_json(frame), separators=(",", ":"))
        self._streams.append(stream_id, record.encode())
        hub.publish(stream_id, frame)

    async def close_stream(self, stream_id: str) -> None:
        """Another task ended: its answer stream is closed, which ends its readers."""
        try:
            await self._streams.close(stream_id)
        except Exception as e:
            logger.warning(
                "an answer stream could not be closed",
                extra={"streamId": stream_id, "error": str(e)},
            )

    def follow(self, after: str | None, *, local: bool = False) -> None:
        """Relays this task's answer stream as another holder writes it, until its last frame
        or ``stop_following``. The frames are durable already: read from the stream from
        ``after``, or, with ``local`` (the holder is in this process), as the holder appends
        them."""

        def relayed(event: pb.StreamResponse) -> bool:
            """Relays one frame; ``True`` at the task's last."""
            if event.HasField("task"):
                return False  # the acknowledgment went out from here already
            self._queue.put_nowait(event)
            status = event.status_update.status.state if event.HasField("status_update") else None
            if status is not None and status in _LAST_STATES:
                self.followed_final = True
                return True
            return False

        async def relay() -> None:
            try:
                reader = self._streams.read(self._stream_id, after=after, live=True)
                while (frame := await reader.next()) is not None:
                    event = json_format.ParseDict(
                        json.loads(frame), pb.StreamResponse(), ignore_unknown_fields=True
                    )
                    if relayed(event):
                        return
            except StorageError as e:
                logger.info(
                    "the answer stream could not be relayed",
                    extra={"streamId": self._stream_id, "error": str(e)},
                )

        async def relay_local() -> None:
            queue = hub.follow(self._stream_id)
            try:
                while not relayed(await queue.get()):
                    pass
            finally:
                hub.unfollow(self._stream_id, queue)

        self._following = asyncio.ensure_future(relay_local() if local else relay())

    async def relayed(self, wait: float) -> None:
        """Waits up to ``wait`` seconds for the relay to reach the task's last frame."""
        if self._following is not None:
            await asyncio.wait({self._following}, timeout=wait)

    def relay_of(self, stream_id: str) -> None:
        """The response is another task's: its stream is relayed, never written or closed."""
        self._stream_id = stream_id
        self._owned = False

    def stop_following(self) -> None:
        if self._following is not None:
            self._following.cancel()
            self._following = None

    def finish(self, outcome: T, *, close: bool = True) -> None:
        """The last frame of the response was emitted: end the relay and the response, and
        close the stream unless the task waits on a question, whose answer continues it on this
        stream. The close waits for the frames queued before it, so it runs on its own: the
        lane's holder never waits for it. A second call changes nothing."""
        if self._finished:
            return
        self._finished = True
        self._outcome = outcome
        self.stop_following()
        if close:
            start(self._close())
        self._queue.put_nowait(None)

    async def fail(self, error: BaseException) -> None:
        """The turn raised: close the stream, hand the error to the relay."""
        if self._finished:
            logger.warning("a turn failed after its answer went out", extra={"error": str(error)})
            return
        self._finished = True
        self.stop_following()
        await self._close()
        self._queue.put_nowait(error)

    async def frames(self) -> AsyncIterator[StreamEvent | pb.StreamResponse]:
        """The response: every frame, then the turn's error when it had one."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _close(self) -> None:
        if not self._owned:
            return
        try:
            await self._streams.close(self._stream_id)
        except Exception as e:
            logger.warning(
                "an answer stream could not be closed",
                extra={"streamId": self._stream_id, "error": str(e)},
            )


@dataclass
class _Streamed:
    """The answer text one task streamed, under the artifact id it streamed it with."""

    artifact_id: str = field(default_factory=lambda: shortuuid("art_"))
    parts: list[str] = field(default_factory=list)


class WireHost(Generic[S]):
    """The host of a wire turn: the handler behind ``handle``, acknowledged on its first call
    (the ``submitted`` row is durable by then), and the live response's frames behind the
    hooks.

    Under ``live`` the request was acknowledged when it arrived, and this host may be the lane's
    holder, running other requests' tasks too: a frame of another task goes to that task's own
    answer stream, which its request relays, and ``on_end`` writes each task's last frames as
    its turn commits."""

    def __init__(
        self,
        host: Callable[[UpdateHook, TokenHook, EndHook], TurnHost[S]],
        pump: ResponsePump[TurnOutcome],
        *,
        task_id: str,
        context_id: str,
        lane: str,
        exts: Extensions,
    ) -> None:
        self._host = host(self.on_update, self.on_token, self.on_end)
        self._pump = pump
        self._task_id = task_id
        self._context_id = context_id
        self._lane = lane
        self._exts = exts
        self.live = False
        """The request's message is under the ``live`` policy."""
        self.follow_after: asyncio.Future[str | None] | None = None
        """Under ``live``, for a continued task: where its stream ended when the request came."""
        self._resent = False
        self.holds = False
        """Under ``live``, this request took the lane: it is the holder."""
        self._answers: dict[str, _Streamed] = {}
        self._acked = False
        self.final_sent = False
        """The response's last frame went out: from a turn of this host, or relayed."""
        self._started = time.monotonic()
        self._marks: dict[str, int] = {}

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        if not self.live:
            # the turn answers the task it runs under: this request's, or the one it continues
            self._task_id = ctx.task_id
        elif ctx.task_id == self._task_id:
            # this host runs its own task: its frames come from here, not from the stream
            self._pump.stop_following()
        self._ack(self._task_id, self._context_id)
        return await self._host.handle(ctx, boundary)

    async def on_update(self, task_id: str, text: str) -> None:
        event = working_event(task_id, self._context_id, text, exts=self._exts)
        if task_id == self._task_id:
            self._pump.emit(event)
        elif self.live:
            self._pump.append(task_id, event)
        # else a batched turn's progress belongs to the task that answers (3.3.4)

    async def on_token(self, task_id: str, delta: str) -> None:
        if not delta or (task_id != self._task_id and not self.live):
            return
        streamed = self._answers.setdefault(task_id, _Streamed())
        event = artifact_event(
            task_id,
            self._context_id,
            artifact_id=streamed.artifact_id,
            text=delta,
            append=bool(streamed.parts),
            last_chunk=False,
            exts=self._exts,
        )
        streamed.parts.append(delta)
        if task_id == self._task_id:
            self._mark("firstTokenMs")
            self._pump.emit(event)
        else:
            self._pump.append(task_id, event)

    async def on_end(self, record: TaskRecord, output: JsonValue) -> None:
        """Under ``live``: a task stopped running in a turn of this host. Its answer and its
        status go on its own stream; this request's response ends with its own task, unless
        the task only went on in the background."""
        # the row is committed: a request of this process that waits on the task reads it now
        hub.task_moved(record.id)
        events: list[StreamEvent] = []
        if record.status == "working":
            # what streamed acknowledged the work; the answer streams on its own later
            self._answers.pop(record.id, None)
        # a canceled task's answer is what ran, which replaces what streamed before the cancel
        if record.status in ("completed", "canceled") and record.data.artifacts:
            events.append(
                self._answer(record.id, output, str(record.data.artifacts[0]["artifactId"]))
            )
        events.append(final_event(record, self._context_id, exts=self._exts))
        if record.id != self._task_id or self.final_sent:
            # another task's, or this one's after its response ended (it asked, and the answer
            # continued it here): the frames go to its stream, which the answer's request relays
            for event in events:
                self._pump.append(record.id, event)
            if record.status in TERMINAL_TASK_STATES:
                start(self._pump.close_stream(record.id))
            return

        for event in events:
            self._pump.emit(event)
        if record.status == "working":
            return  # the task goes on in the background; its answer comes later
        from .methods import TurnOutcome

        self.final_sent = True
        self._mark("doneMs")
        outcome = TurnOutcome(
            result=TurnResult(
                task=record,
                output=output,
                thread_id=self._context_id,
                answered_by=record.id,
                run_id=None,
            ),
            context_id=self._context_id,
        )
        self._pump.finish(outcome, close=record.status in TERMINAL_TASK_STATES)

    async def background(self, wait: float, cancel: list[str]) -> Background:
        return await self._host.background(wait, cancel)

    async def arrived(self, task_id: str) -> None:
        await self._host.arrived(task_id)

    async def settled(self, task_id: str, wait: float) -> bool:
        return await self._host.settled(task_id, wait)

    async def accepted(self, task_id: str, resent: bool, holds: bool) -> None:
        """The turn found which task the message is: its own, the question it continues, or,
        for a resend of a message the lane has, the task that message made. A resend's response
        is that task's stream from its first frame, under its id. Under ``live`` a new message
        is acknowledged here, durable in the inbox, and its stream is relayed until a turn of
        this host runs its task; when it ``holds`` the lane, the other requests of this process
        hear from it from now on."""
        if holds:
            self.holds = True
            hub.lane_hub(self._context_id, self._lane).held = True
        if resent:
            self._resent = True
            self._task_id = task_id
            self._pump.relay_of(task_id)
            self._ack(task_id, self._context_id)
            self._pump.follow(None)
            return
        if not self.live:
            return
        self._ack(task_id, self._context_id)
        after = await self.follow_after if self.follow_after is not None else None
        if not self.final_sent:
            local = hub.held_here(self._context_id, self._lane) is not None
            self._pump.follow(after, local=local)

    async def discard(self, task_id: str) -> None:
        await self._host.discard(task_id)

    async def finish(self, outcome: TurnOutcome) -> None:
        """The frames after the commit: the acknowledgment when no row of this request's own
        was run here (a merge), the answer's last chunk, the final status. Nothing when
        ``on_end`` or the relayed stream sent them already: when another turn ran the task (a
        resend, or under ``live``), the relay is given ``RELAY_DRAIN`` to reach its last frame."""
        state = outcome.result.task.status
        if (self._resent or self.live) and (
            state in TERMINAL_TASK_STATES or state == "input-required"
        ):
            await self._pump.relayed(RELAY_DRAIN)
        if self.final_sent or self._pump.followed_final:
            return
        # under live the holder wrote these frames to the stream already: shown, not appended
        emit = self._pump.show if self.live or self._resent else self._pump.emit
        record = outcome.result.task
        self._ack(record.id, outcome.context_id)
        artifacts = record.data.artifacts
        if record.id == self._task_id and artifacts:
            emit(self._answer(record.id, outcome.result.output, str(artifacts[0]["artifactId"])))
        emit(final_event(record, outcome.context_id, exts=self._exts))
        self._mark("doneMs")
        logger.debug("a turn's timing", extra={"taskId": self._task_id, **self._marks})

    def _ack(self, task_id: str, context_id: str) -> None:
        if self._acked:
            return

        self._acked = True
        self._mark("ackMs")
        self._pump.acked = task_id
        # a resend's stream has its acknowledgment already
        ack = self._pump.show if self._resent else self._pump.emit
        ack(submitted_task(task_id, context_id))

    def _mark(self, name: str) -> None:
        """Milliseconds from the host's creation to a point of the turn, the first time."""
        self._marks.setdefault(name, round((time.monotonic() - self._started) * 1000))

    def _answer(
        self, task_id: str, output: JsonValue, artifact_id: str
    ) -> pb.TaskArtifactUpdateEvent:
        """The answer artifact's last chunk: empty when the streamed tokens spelled the answer
        already, the whole answer under the row's artifact id otherwise."""
        text = output if isinstance(output, str) else json.dumps(output, separators=(",", ":"))
        streamed = self._answers.pop(task_id, None)
        if streamed is not None and "".join(streamed.parts) == text:
            return artifact_event(
                task_id,
                self._context_id,
                artifact_id=streamed.artifact_id,
                text="",
                append=True,
                last_chunk=True,
                exts=self._exts,
            )

        return artifact_event(
            task_id,
            self._context_id,
            artifact_id=artifact_id,
            text=text,
            append=False,
            last_chunk=True,
            exts=self._exts,
        )


__all__ = ["ResponsePump", "WireHost", "running", "start"]
