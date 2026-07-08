"""Test/dev helpers for running an agent in-process without a worker or AgentServer."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit import rtc

    from .ipc.channel import Message
    from .ipc.inference_executor import InferenceExecutor
    from .job import JobContext

__all__ = ["fake_job_context"]


class _NoopInferenceExecutor:
    """Local inference is unavailable in a fake (worker-less) job; the agent under
    test should use a remote model (e.g. inference.LLM)."""

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        raise NotImplementedError("inference executor not available in a fake job")


class _NoopIPCClient:
    """A fake (worker-less) job has no main process to talk to; text-mode responses
    (the only user of the IPC client) aren't supported here."""

    async def send(self, msg: Message) -> None:
        raise NotImplementedError("IPC client not available in a fake job")


@contextlib.contextmanager
def fake_job_context(
    *,
    room: rtc.Room | None = None,
    job_id: str = "fake-job",
    job_metadata: str = "",
    agent_identity: str = "fake-agent",
    inference_executor: InferenceExecutor | None = None,
) -> Iterator[JobContext]:
    """Install a fake :class:`JobContext` as the current job context, for running an
    agent in-process (tests/dev) — no worker, no AgentServer.

    The context is a real ``fake_job`` :class:`JobContext` (so ``get_job_context()``
    and its access points behave normally). The :class:`JobContext` is yielded so
    callers can tweak it (set ``_simulation_end_fnc``, etc.). Wrap
    ``session.start(room=room)`` with it::

        async with rtc.Room() as room:
            await room.connect(url, token)
            with fake_job_context(room=room):
                await session.start(agent=MyAgent(), room=room)
    """
    from livekit import rtc as _rtc
    from livekit.protocol import agent as agent_proto, models

    from . import utils
    from .job import (
        JobAcceptArguments,
        JobContext,
        JobExecutorType,
        JobProcess,
        RunningJobInfo,
        _JobContextVar,
    )

    room = room if room is not None else _rtc.Room()
    job = agent_proto.Job(
        id=job_id,
        room=models.Room(name=room.name or "fake-room", sid=utils.shortuuid("RM_")),
        type=agent_proto.JobType.JT_ROOM,
        metadata=job_metadata,
    )
    info = RunningJobInfo(
        accept_arguments=JobAcceptArguments(identity=agent_identity, name="", metadata=""),
        job=job,
        url="",
        token="",
        worker_id="fake",
        fake_job=True,
    )
    proc = JobProcess(executor_type=JobExecutorType.THREAD, user_arguments=None, http_proxy=None)
    jc = JobContext(
        proc=proc,
        info=info,
        room=room,
        on_connect=lambda: None,
        on_shutdown=lambda _reason: None,
        inference_executor=inference_executor or _NoopInferenceExecutor(),
        ipc_client=_NoopIPCClient(),
    )
    # the caller owns the room connection (a fake job has no signal URL), so mark the
    # context connected to keep JobContext.connect() — invoked by session.start() — a no-op
    jc._connected = True

    token = _JobContextVar.set(jc)
    try:
        yield jc
    finally:
        _JobContextVar.reset(token)
