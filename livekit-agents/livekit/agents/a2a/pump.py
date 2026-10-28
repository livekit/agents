"""The live response of one message, and its durable answer stream (plan 6).

A turn appends each frame to its task's stream, and every response follows its own task's stream,
read from this worker's memory when this worker writes it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from google.protobuf import json_format
from pydantic import JsonValue

from .._core import Boundary, Streams
from .._generated.storage_run import TaskRecord
from .._generated.storage_turn import Background, HandlerOutput, TurnContext
from ..log import logger
from ..storage.context import TokenHook, TurnHost, UpdateHook
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
from .extension import Extensions, pb, to_json

if TYPE_CHECKING:
    from .methods import TurnOutcome

S = TypeVar("S")

_LAST_STATES = frozenset(
    {
        pb.TaskState.TASK_STATE_COMPLETED,
        pb.TaskState.TASK_STATE_FAILED,
        pb.TaskState.TASK_STATE_CANCELED,
        pb.TaskState.TASK_STATE_REJECTED,
        pb.TaskState.TASK_STATE_INPUT_REQUIRED,
    }
)
"""A followed stream's response ends at these: the task ended, or waits on a question."""

RELAY_DRAIN = 5.0
"""How long a response that follows a stream another worker writes waits for the task's last
frame once its row has one, in seconds, before it writes that frame from the row: a task failed
before any turn ran it has none."""

running: set[asyncio.Task[None]] = set()
"""The turns in flight, each pumping its stream; a test waits on them before its leak check."""


def start(turn: Coroutine[object, object, None]) -> None:
    """Runs one turn to its end, whatever happens to the response that asked for it."""
    task = asyncio.ensure_future(turn)
    running.add(task)
    task.add_done_callback(running.discard)


class ResponsePump:
    """The frames of one message's response, in order: those of the task stream it follows."""

    def __init__(self, streams: Streams, stream_id: str) -> None:
        self._streams = streams
        self._stream_id = stream_id
        self._queue: asyncio.Queue[StreamEvent | pb.StreamResponse | BaseException | None] = (
            asyncio.Queue()
        )
        self._following: asyncio.Future[None] | None = None
        self._finished = False
        self.acked: str | None = None
        """The task id the acknowledgment named, once it went out."""

    def append(self, stream_id: str, event: StreamEvent) -> None:
        """One frame to a task's stream: queued, sent in order, a failure logged."""
        record = json.dumps(to_json(to_stream_response(event)), separators=(",", ":"))
        self._streams.append(stream_id, record.encode())

    def show(self, event: StreamEvent | pb.StreamResponse) -> None:
        """One frame to the response only."""
        self._queue.put_nowait(event)

    def forget(self, stream_id: str) -> None:
        """The task waits on a question: its stream is read from the store from now on."""
        self._streams.forget(stream_id)

    def follow(self, stream_id: str, after: str | None, *, local: bool) -> None:
        """The response relays a task's stream from ``after`` up to the frame where the task ends
        or waits on a question; ``local`` when this worker writes it."""

        async def relay() -> None:
            try:
                reader = self._streams.read(stream_id, after=after, live=True, local=local)
                while (frame := await reader.next()) is not None:
                    event = json_format.ParseDict(
                        json.loads(frame), pb.StreamResponse(), ignore_unknown_fields=True
                    )
                    if event.HasField("task"):
                        continue  # the acknowledgment went out from here
                    self.show(event)
                    if event.status_update.status.state in _LAST_STATES:
                        break
            except StorageError as e:
                logger.warning(
                    "an answer stream could not be followed",
                    extra={"streamId": stream_id, "error": str(e)},
                )
                return
            self._end()

        self._following = asyncio.ensure_future(relay())

    async def followed(self, wait: float) -> bool:
        """Waits up to ``wait`` seconds for the followed stream's last frame; ``True`` once the
        response ended with it."""
        if self._following is not None:
            await asyncio.wait({self._following}, timeout=wait)
        return self._finished

    def _end(self) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(None)

    async def fail(self, error: BaseException) -> None:
        """The turn raised: close the stream, hand the error to the response, unless it ended
        already."""
        if self._finished:
            logger.warning("a turn failed after its response ended", extra={"error": str(error)})
            return
        self._finished = True
        if self._following is not None:
            self._following.cancel()
        await self.close(self._stream_id)
        self._queue.put_nowait(error)

    async def frames(self) -> AsyncIterator[StreamEvent | pb.StreamResponse]:
        """The response: every frame, then the turn's error when it had one. A client that goes
        away stops the relay, never the turn."""
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if self._following is not None:
                self._following.cancel()

    async def close(self, stream_id: str) -> None:
        """Closes a task's stream, which ends its readers."""
        try:
            await self._streams.close(stream_id)
        except Exception as e:
            logger.warning(
                "an answer stream could not be closed",
                extra={"streamId": stream_id, "error": str(e)},
            )


async def tail(streams: Streams, stream_id: str) -> str | None:
    """Where a task's stream ends now: what a continuation of the task writes comes after."""
    await streams.flush(stream_id)
    reader = streams.read(stream_id)
    while await reader.next() is not None:
        pass
    return await reader.next_offset()


@dataclass
class _Streamed:
    """The answer text one task streamed, under the artifact id it streamed it with."""

    artifact_id: str = field(default_factory=lambda: shortuuid("art_"))
    parts: list[str] = field(default_factory=list)


class WireHost(Generic[S]):
    """The host of a wire turn: the handler behind ``handle``, the frames of every task it runs
    appended to that task's stream, and the response, which follows its own task's stream from
    the acknowledgment."""

    def __init__(
        self,
        host: Callable[[UpdateHook, TokenHook], TurnHost[S]],
        pump: ResponsePump,
        *,
        task_id: str,
        context_id: str,
        exts: Extensions,
        continued_from: asyncio.Future[str | None] | None = None,
    ) -> None:
        self._host = host(self.on_update, self.on_token)
        self._pump = pump
        self._task_id = task_id
        self._context_id = context_id
        self._exts = exts
        self._continued_from = continued_from
        self._local = True
        self._answers: dict[str, _Streamed] = {}
        self._ended: set[str] = set()
        """The tasks whose end a turn of this host wrote."""
        self._started = time.monotonic()
        self._marks: dict[str, int] = {}

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        await self._ack(self._task_id, local=True)
        return await self._host.handle(ctx, boundary)

    async def on_update(self, task_id: str, text: str, verbatim: bool = False) -> None:
        event = working_event(task_id, self._context_id, text, exts=self._exts, verbatim=verbatim)
        self._pump.append(task_id, event)

    async def on_token(self, task_id: str, delta: str) -> None:
        if not delta:
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
        self._mark("firstTokenMs")
        self._pump.append(task_id, event)

    async def accepted(self, task_id: str, local: bool) -> None:
        """Rust calls it under ``live`` once the message is in the inbox as ``task_id``'s."""
        await self._ack(task_id, local=local)

    async def on_end(self, record: TaskRecord, output: JsonValue) -> None:
        """Rust calls it under ``live`` after each commit, for every task the turn carried."""
        self._ended.add(record.id)
        if record.status == "working":
            # the reply beside work in the background was not the answer, which streams anew
            self._answers.pop(record.id, None)
        verbatim = output is not None and self._host.verbatim
        if output is not None and record.data.artifacts:
            artifact_id = str(record.data.artifacts[0]["artifactId"])
            self._pump.append(record.id, self._answer(record.id, output, artifact_id, verbatim))
        self._pump.append(
            record.id, final_event(record, self._context_id, exts=self._exts, verbatim=verbatim)
        )
        if record.status == "input-required":
            self._pump.forget(record.id)
        elif record.status in TERMINAL_TASK_STATES:
            start(self._pump.close(record.id))

    async def cancel(self, task_id: str) -> None:
        """Rust calls it under ``live`` while holding the lane, for a canceled task whose work
        runs."""
        await self._host.cancel(task_id)

    async def background(self, wait: float) -> Background:
        """Rust calls it under ``live`` between turns, to learn what runs and what is ready."""
        return await self._host.background(wait)

    async def finish(self, outcome: TurnOutcome) -> None:
        """The request's turn returned: once its task's row has an end that no turn of this host
        wrote, that end is written from the row."""
        record = outcome.result.task
        if record.status not in TERMINAL_TASK_STATES and record.status != "input-required":
            return  # the response follows the stream until the task gets there
        # a resend's task, or a merged message's, which no turn of this host ran
        await self._ack(record.id, local=record.id == self._task_id)
        if record.id not in self._ended and (
            self._local or not await self._pump.followed(RELAY_DRAIN)
        ):
            await self.on_end(record, outcome.result.output if record.id == self._task_id else None)
        self._mark("doneMs")
        logger.debug("a turn's timing", extra={"taskId": self._task_id, **self._marks})

    async def _ack(self, task_id: str, *, local: bool) -> None:
        """The acknowledgment, once, and the response follows ``task_id``'s stream: from its
        start, or from where it ended for a task the message continues or a resend's."""
        if self._pump.acked is not None:
            return

        self._pump.acked = task_id
        self._local = local
        self._mark("ackMs")
        ack = submitted_task(task_id, self._context_id)
        new = task_id == self._task_id and self._continued_from is None
        if new:
            self._pump.append(task_id, ack)
        self._pump.show(ack)
        after = await self._continued_from if self._continued_from is not None else None
        self._pump.follow(task_id, after, local=local and new)

    def _mark(self, name: str) -> None:
        """Milliseconds from the host's creation to a point of the turn, the first time."""
        self._marks.setdefault(name, round((time.monotonic() - self._started) * 1000))

    def _answer(
        self, task_id: str, output: JsonValue, artifact_id: str, verbatim: bool
    ) -> pb.TaskArtifactUpdateEvent:
        """The answer artifact's last chunk: empty when the streamed tokens spelled the answer
        already, the whole answer under the row's artifact id otherwise, flagged when it
        reaches the person as written."""
        text = output if isinstance(output, str) else json.dumps(output, separators=(",", ":"))
        streamed = self._answers.pop(task_id, None)
        if streamed is not None and "".join(streamed.parts) == text and not verbatim:
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
            verbatim=verbatim,
        )


__all__ = ["ResponsePump", "WireHost", "running", "start", "tail"]
