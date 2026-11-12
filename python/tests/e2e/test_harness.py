"""Hermetic failure injection against actual subprocesses and local HTTP peers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

import httpx
import pytest
from a2a.types import a2a_pb2 as pb
from aiohttp import web

from .drivers import A2ACaller, Workers
from .resources import Child, child_process, finish_cleanup, recording_service

pytestmark = pytest.mark.unit


@pytest.fixture
def live_credentials(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    """Local startup/authentication tests never address the real storage service."""
    values = ("ws://127.0.0.1:1", "APItest", "test-secret-test-secret-test-secret")
    for name, value in zip(
        ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"), values, strict=True
    ):
        monkeypatch.setenv(name, value)
    return values


async def test_real_worker_processes_start_and_serve_their_text_doors(
    workers: Workers,
) -> None:
    """Each worker serves its routes and its text doors on its own listener. The workers run
    as the console does, so the listener vouches for every caller; how a door refuses one is
    pinned by its Rust tests (``lk-agents-runtime/src/a2a/door_tests.rs``)."""
    assert workers.a_process.process.pid != workers.b_process.process.pid
    for name, caller in (("worker-a", workers.a), ("worker-b", workers.b)):
        assert (await caller.http.get("/ready")).text == name
        async with httpx.AsyncClient(base_url=caller.http.base_url, trust_env=False) as http:
            response = await http.post("/desk/v1", json={})
            assert response.status_code == 200
            assert response.json()["error"]["code"] == -32600


async def started(child: Child) -> None:
    async with asyncio.timeout(5):
        while "started" not in child.log.read_text():
            if child.process.returncode is not None:
                raise RuntimeError(child.log.read_text())
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("exit_mode", ["failure", "timeout", "cancellation"])
async def test_owned_resources_close_and_children_are_reaped_after_aborted_scenario(
    tmp_path: Path, exit_mode: str
) -> None:
    child: Child | None = None
    service_url = ""
    entered = asyncio.Event()

    async def scenario() -> None:
        nonlocal child, service_url
        async with recording_service() as service:
            service_url = service.url
            async with child_process(
                sys.executable,
                "-u",
                "-c",
                "import time; print('started'); time.sleep(60)",
                log=tmp_path / "worker.log",
            ) as owned:
                child = owned
                await started(child)
                entered.set()
                if exit_mode == "failure":
                    raise ValueError("scenario failed")
                if exit_mode == "timeout":
                    async with asyncio.timeout(0.01):
                        await asyncio.Event().wait()
                await asyncio.Event().wait()

    task = asyncio.create_task(scenario())
    async with asyncio.timeout(5):
        await entered.wait()
    if exit_mode == "cancellation":
        task.cancel()
    expected = {
        "failure": ValueError,
        "timeout": TimeoutError,
        "cancellation": asyncio.CancelledError,
    }
    with pytest.raises(expected[exit_mode]):
        await task
    assert child is not None and child.reaped
    assert child.process.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(child.process.pid, os.WNOHANG)
    assert "started" in child.log.read_text()
    async with httpx.AsyncClient(trust_env=False) as http:
        with pytest.raises(httpx.ConnectError):
            await http.post(service_url + "/call", json={})


async def test_failed_worker_startup_is_an_error_with_retained_log_and_reaping(
    tmp_path: Path,
) -> None:
    async with child_process(
        sys.executable,
        "-c",
        "raise RuntimeError('injected startup failure')",
        log=tmp_path / "startup.log",
    ) as child:
        with pytest.raises(RuntimeError, match="injected startup failure"):
            await child.ready("http://127.0.0.1:1/ready")
    assert child.reaped and child.process.returncode != 0


async def test_cleanup_settles_when_teardown_is_cancelled_repeatedly() -> None:
    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup() -> None:
        entered.set()
        await release.wait()
        settled.set()

    task = asyncio.create_task(finish_cleanup(cleanup()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not settled.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"jsonrpc":"2.0","result":{}}',
        '{"jsonrpc":"2.0","error":{"code":-32603,"message":"injected"}}',
    ],
)
async def test_missing_or_invalid_response_artifacts_cannot_become_a_success(payload: str) -> None:
    async def broken(request: web.Request) -> web.Response:
        return web.Response(text=payload, content_type="application/json")

    app = web.Application()
    app.router.add_post("/desk/v1", broken)
    runner = web.AppRunner(app)
    await runner.setup()
    caller: A2ACaller | None = None
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0]
        caller = A2ACaller(f"http://{host}:{port}", "unused")
        if payload == '{"jsonrpc":"2.0","result":{}}':
            observed = await caller.send(pb.Message())
            assert observed.status.state != pb.TaskState.TASK_STATE_COMPLETED
            assert not observed.artifacts
        else:
            with pytest.raises((ValueError, RuntimeError)):
                await caller.send(pb.Message())
    finally:
        if caller is not None:
            await caller.close()
        await runner.cleanup()


async def test_recording_service_observes_exact_effect_and_releases_blocked_requests() -> None:
    async with recording_service() as service, httpx.AsyncClient(trust_env=False) as http:
        response = await http.post(
            service.url + "/call",
            json={"worker": "a", "thread": "t", "operation": "book", "terms": "slot-9 $100"},
        )
        assert response.text == "booked slot-9 $100"
        assert len(service.calls) == 1 and service.calls[0].terms == "slot-9 $100"
        pending = asyncio.create_task(
            http.post(
                service.url + "/call", json={"worker": "a", "thread": "t", "operation": "wait"}
            )
        )
        try:
            await service.wait_for_call("t", "wait")
            assert not pending.done()
            service.release("t")
            assert (await pending).text == "found"
        finally:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
