"""JobContext accepts any async callable as a shutdown callback or participant entrypoint,
not only plain functions: ``functools.partial`` objects and callable instances included."""

from __future__ import annotations

import asyncio
import functools
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import JobContext
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent as agent_proto, models

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


def _job_ctx(remote_participants: dict[str, MagicMock] | None = None) -> JobContext:
    info = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="agent", identity="agent-1", metadata="{}"),
        job=agent_proto.Job(id="AJ_1", room=models.Room(sid="RM_1", name="room")),
        url="wss://example.livekit.cloud",
        token="tok",
        worker_id="W_1",
        fake_job=False,
    )
    room = MagicMock()
    room.connect = AsyncMock()
    room.isconnected.return_value = False
    room.remote_participants = remote_participants or {}
    return JobContext(
        proc=MagicMock(),
        info=info,
        room=room,
        on_connect=lambda: None,
        on_shutdown=lambda reason: None,
        inference_executor=MagicMock(),
    )


async def _run_shutdown_callbacks(ctx: JobContext, reason: str) -> None:
    for callback in ctx._shutdown_callbacks:
        await callback(reason)


async def test_shutdown_callback_accepts_partial() -> None:
    ctx = _job_ctx()
    calls: list[tuple[str, ...]] = []

    async def save_transcript(path: str) -> None:
        calls.append((path,))

    async def save_with_reason(path: str, reason: str) -> None:
        calls.append((path, reason))

    ctx.add_shutdown_callback(functools.partial(save_transcript, "a.json"))
    ctx.add_shutdown_callback(functools.partial(save_with_reason, "b.json"))
    await _run_shutdown_callbacks(ctx, "room deleted")

    # the reason is passed only to the partial with a free positional parameter left
    assert calls == [("a.json",), ("b.json", "room deleted")]


async def test_shutdown_callback_accepts_callable_object() -> None:
    ctx = _job_ctx()
    calls: list[str] = []

    class Flusher:
        async def __call__(self, reason: str) -> None:
            calls.append(reason)

    class Closer:
        async def __call__(self) -> None:
            calls.append("closed")

    ctx.add_shutdown_callback(Flusher())
    ctx.add_shutdown_callback(Closer())
    await _run_shutdown_callbacks(ctx, "room deleted")

    assert calls == ["room deleted", "closed"]


async def test_shutdown_callback_reason_detection_for_functions_and_methods() -> None:
    ctx = _job_ctx()
    calls: list[str] = []

    async def without_reason() -> None:
        calls.append("function")

    async def with_reason(reason: str) -> None:
        calls.append(f"function:{reason}")

    class Recorder:
        async def aclose(self) -> None:
            calls.append("method")

        async def on_shutdown(self, reason: str) -> None:
            calls.append(f"method:{reason}")

    recorder = Recorder()
    for callback in (without_reason, with_reason, recorder.aclose, recorder.on_shutdown):
        ctx.add_shutdown_callback(callback)
    await _run_shutdown_callbacks(ctx, "bye")

    assert calls == ["function", "function:bye", "method", "method:bye"]


async def test_participant_entrypoint_accepts_partial() -> None:
    participant = MagicMock()
    participant.identity = "caller"
    participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    ctx = _job_ctx({"caller": participant})
    greeted: list[tuple[str, str]] = []

    async def greet(ctx: JobContext, p: rtc.RemoteParticipant, greeting: str) -> None:
        greeted.append((greeting, p.identity))

    ctx.add_participant_entrypoint(functools.partial(greet, greeting="hello"))
    await ctx.connect()
    await asyncio.gather(*ctx._participant_tasks.values())

    assert greeted == [("hello", "caller")]
