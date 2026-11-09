from __future__ import annotations

import asyncio
import threading
from typing import ClassVar

import pytest

from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc import channel, proto
from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
from livekit.agents.ipc.inference_proc_lazy_main import _InferenceProc
from livekit.agents.ipc.job_proc_lazy_main import _InfClient
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit


class _SendRecorder:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.sent = asyncio.Event()

    async def send(self, msg: object) -> None:
        self.messages.append(msg)
        self.sent.set()


class _SendBytesRecorder:
    def __init__(self) -> None:
        self.sent = asyncio.Event()

    async def send_bytes(self, data: bytes) -> None:
        self.sent.set()


async def test_job_inference_client_cleans_up_cancelled_request() -> None:
    client = _SendRecorder()
    inf_client = _InfClient(client)  # type: ignore[arg-type]

    task = asyncio.create_task(inf_client.do_inference("test", b"input"))
    await asyncio.wait_for(client.sent.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not inf_client._active_requests


async def test_worker_inference_client_cleans_up_cancelled_request() -> None:
    executor = InferenceProcExecutor.__new__(InferenceProcExecutor)
    executor._active_requests = {}
    duplex = _SendBytesRecorder()
    executor._pch = duplex  # type: ignore[assignment]
    executor._supervise_atask = asyncio.create_task(asyncio.sleep(0))

    task = asyncio.create_task(executor.do_inference("test", b"input"))
    await asyncio.wait_for(duplex.sent.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not executor._active_requests
    await executor._supervise_atask


async def test_inference_process_runs_requests_concurrently() -> None:
    both_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    num_started = 0

    class _BlockingRunner(_InferenceRunner):
        INFERENCE_METHOD: ClassVar[str] = "blocking"

        def initialize(self) -> None:
            pass

        def run(self, data: bytes) -> bytes:
            nonlocal num_started
            with lock:
                num_started += 1
                if num_started == 2:
                    both_started.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("test did not release inference")
            return data

    proc = _InferenceProc({"blocking": _BlockingRunner})
    client = _SendRecorder()
    proc._client = client  # type: ignore[assignment]
    requests: aio.Chan[channel.Message] = aio.Chan()
    entrypoint_task = asyncio.create_task(proc.entrypoint(requests))

    try:
        requests.send_nowait(proto.InferenceRequest(request_id="one", method="blocking", data=b"1"))
        requests.send_nowait(proto.InferenceRequest(request_id="two", method="blocking", data=b"2"))

        # The first request blocks in a worker thread. The second can start only if the
        # event loop continues dispatching instead of awaiting each request in turn.
        await asyncio.wait_for(asyncio.to_thread(both_started.wait), timeout=1.0)

        requests.send_nowait(proto.ShutdownRequest(reason="test"))
        release.set()
        await asyncio.wait_for(entrypoint_task, timeout=1.0)
    finally:
        release.set()
        requests.close()
        await aio.cancel_and_wait(entrypoint_task)
        proc._executor.shutdown(wait=True, cancel_futures=True)

    responses = [msg for msg in client.messages if isinstance(msg, proto.InferenceResponse)]
    assert {msg.request_id for msg in responses} == {"one", "two"}
    assert all(not msg.error for msg in responses)
    assert isinstance(client.messages[-1], proto.Exiting)
