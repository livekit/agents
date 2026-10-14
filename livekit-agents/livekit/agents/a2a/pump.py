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
from typing import TYPE_CHECKING, Generic, TypeVar

from .._core import Boundary, Streams
from .._generated.storage_turn import HandlerOutput, TurnContext
from ..log import logger
from ..storage.context import TokenHook, TurnHost, UpdateHook
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
T = TypeVar("T")

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
        self._queue: asyncio.Queue[StreamEvent | BaseException | None] = asyncio.Queue()
        self._outcome: T | None = None

    @property
    def outcome(self) -> T:
        """What the turn produced; read once ``frames`` ended without an error."""
        if self._outcome is None:
            raise RuntimeError("the turn has not ended")
        return self._outcome

    def emit(self, event: StreamEvent) -> None:
        """One frame: appended to the stream (queued, sent in order, a failure is logged and does
        not fail the turn), then relayed."""
        record = json.dumps(to_json(to_stream_response(event)), separators=(",", ":"))
        self._streams.append(self._stream_id, record.encode())
        self._queue.put_nowait(event)

    async def finish(self, outcome: T) -> None:
        """The terminal frame was emitted: close the stream, end the relay."""
        self._outcome = outcome
        await self._close()
        self._queue.put_nowait(None)

    async def fail(self, error: BaseException) -> None:
        """The turn raised: close the stream, hand the error to the relay."""
        await self._close()
        self._queue.put_nowait(error)

    async def frames(self) -> AsyncIterator[StreamEvent]:
        """The response: every frame, then the turn's error when it had one."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _close(self) -> None:
        try:
            await self._streams.close(self._stream_id)
        except Exception as e:
            logger.warning(
                "an answer stream could not be closed",
                extra={"streamId": self._stream_id, "error": str(e)},
            )


class WireHost(Generic[S]):
    """The host of a wire turn: the handler behind ``handle``, acknowledged on its first call
    (the ``submitted`` row is durable by then), and the live response's frames behind the
    hooks."""

    def __init__(
        self,
        host: Callable[[UpdateHook, TokenHook], TurnHost[S]],
        pump: ResponsePump[TurnOutcome],
        *,
        task_id: str,
        context_id: str,
        exts: Extensions,
    ) -> None:
        self._host = host(self.on_update, self.on_token)
        self._pump = pump
        self._task_id = task_id
        self._context_id = context_id
        self._exts = exts
        self._artifact_id = shortuuid("art_")
        self._streamed: list[str] = []
        self._acked = False
        self._started = time.monotonic()
        self._marks: dict[str, int] = {}

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        self._ack(self._task_id, self._context_id)
        return await self._host.handle(ctx, boundary)

    async def on_update(self, task_id: str, text: str) -> None:
        if task_id != self._task_id:
            return  # a batched turn's progress belongs to the task that answers (3.3.4)

        self._pump.emit(working_event(task_id, self._context_id, text, exts=self._exts))

    async def on_token(self, task_id: str, delta: str) -> None:
        if task_id != self._task_id or not delta:
            return
        self._mark("firstTokenMs")

        self._pump.emit(
            artifact_event(
                task_id,
                self._context_id,
                artifact_id=self._artifact_id,
                text=delta,
                append=bool(self._streamed),
                last_chunk=False,
                exts=self._exts,
            )
        )
        self._streamed.append(delta)

    def finish(self, outcome: TurnOutcome) -> None:
        """The frames after the commit: the acknowledgment when no row of this request's own
        was run here (a resend, a merge), the answer's last chunk, the final status."""
        record = outcome.result.task
        self._ack(record.id, outcome.context_id)
        artifacts = record.data.artifacts
        if record.id == self._task_id and artifacts:
            self._pump.emit(self._answer(outcome, str(artifacts[0]["artifactId"])))
        self._pump.emit(final_event(record, outcome.context_id, exts=self._exts))
        self._mark("doneMs")
        logger.debug("a turn's timing", extra={"taskId": self._task_id, **self._marks})

    def _ack(self, task_id: str, context_id: str) -> None:
        if self._acked:
            return

        self._acked = True
        self._mark("ackMs")
        self._pump.emit(submitted_task(task_id, context_id))

    def _mark(self, name: str) -> None:
        """Milliseconds from the host's creation to a point of the turn, the first time."""
        self._marks.setdefault(name, round((time.monotonic() - self._started) * 1000))

    def _answer(self, outcome: TurnOutcome, artifact_id: str) -> pb.TaskArtifactUpdateEvent:
        """The answer artifact's last chunk: empty when the streamed tokens spelled the answer
        already, else the whole answer, replacing what was streamed under the stream's artifact
        id, or under the row's when nothing was streamed."""
        output = outcome.result.output
        text = output if isinstance(output, str) else json.dumps(output, separators=(",", ":"))
        if self._streamed and "".join(self._streamed) == text:
            return artifact_event(
                self._task_id,
                outcome.context_id,
                artifact_id=self._artifact_id,
                text="",
                append=True,
                last_chunk=True,
                exts=self._exts,
            )

        return artifact_event(
            self._task_id,
            outcome.context_id,
            artifact_id=self._artifact_id if self._streamed else artifact_id,
            text=text,
            append=False,
            last_chunk=True,
            exts=self._exts,
        )


__all__ = ["ResponsePump", "WireHost", "running", "start"]
