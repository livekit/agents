from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import AgentServer, JobContext, JobExecutorType
from livekit.agents.ipc.mock_room import create_mock_room
from livekit.agents.ipc.proc_pool import ProcPool
from livekit.agents.job import RunningJobInfo
from livekit.protocol import agent, models

pytestmark = pytest.mark.unit


async def _status_entrypoint(ctx: JobContext) -> None:
    async def disconnect(
        *, reason: rtc.DisconnectReason.ValueType = rtc.DisconnectReason.CLIENT_INITIATED
    ) -> None:
        async with aiohttp.ClientSession() as http:
            async with http.post(ctx.job.room.metadata, json={"reason": reason}) as response:
                response.raise_for_status()

    # The worker connection and executors are real; only RTC is mocked.
    room = create_mock_room()
    room.disconnect.side_effect = disconnect
    room.remote_participants = {}
    scenario = ctx.job.metadata
    if scenario == "raise_before_connect":
        raise RuntimeError("entrypoint failed before connect")

    await ctx.connect()
    if scenario == "raise_after_connect":
        raise RuntimeError("entrypoint failed after connect")

    ctx.shutdown()
    if scenario == "shutdown_then_raise":
        raise RuntimeError("entrypoint failed after shutdown")


@pytest.mark.parametrize("executor_type", [JobExecutorType.PROCESS, JobExecutorType.THREAD])
@pytest.mark.parametrize(
    ("scenario", "expected", "expected_disconnect_reason"),
    [
        ("raise_before_connect", agent.JobStatus.JS_FAILED, rtc.DisconnectReason.AGENT_ERROR),
        ("raise_after_connect", agent.JobStatus.JS_FAILED, rtc.DisconnectReason.AGENT_ERROR),
        ("shutdown_then_raise", agent.JobStatus.JS_SUCCESS, rtc.DisconnectReason.CLIENT_INITIATED),
        ("normal_shutdown", agent.JobStatus.JS_SUCCESS, rtc.DisconnectReason.CLIENT_INITIATED),
    ],
)
async def test_final_job_status_reaches_worker_websocket(
    executor_type: JobExecutorType,
    scenario: str,
    expected: agent.JobStatus,
    expected_disconnect_reason: rtc.DisconnectReason.ValueType,
) -> None:
    terminal: asyncio.Future[agent.UpdateJobStatus] = asyncio.get_running_loop().create_future()
    job = agent.Job(
        id="AJ_status_test",
        type=agent.JobType.JT_ROOM,
        room=models.Room(name="status_test"),
        metadata=scenario,
    )
    updates: list[agent.UpdateJobStatus] = []
    disconnect_reasons: list[int] = []

    async def rtc_disconnect(request: web.Request) -> web.Response:
        disconnect_reasons.append((await request.json())["reason"])
        return web.Response()

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            if packet.type != aiohttp.WSMsgType.BINARY:
                continue
            message = agent.WorkerMessage.FromString(packet.data)
            if message.HasField("register"):
                await ws.send_bytes(
                    agent.ServerMessage(
                        register=agent.RegisterWorkerResponse(worker_id="AW_status_test")
                    ).SerializeToString()
                )
                await ws.send_bytes(
                    agent.ServerMessage(
                        availability=agent.AvailabilityRequest(job=job)
                    ).SerializeToString()
                )
            elif message.HasField("availability"):
                assert message.availability.available
                await ws.send_bytes(
                    agent.ServerMessage(
                        assignment=agent.JobAssignment(job=job, token="fake_token")
                    ).SerializeToString()
                )
            elif message.HasField("update_job"):
                updates.append(message.update_job)
                if message.update_job.status in (
                    agent.JobStatus.JS_FAILED,
                    agent.JobStatus.JS_SUCCESS,
                ):
                    if not terminal.done():
                        terminal.set_result(message.update_job)
        return ws

    app = web.Application()
    app.router.add_get("/agent", agent_ws)
    app.router.add_post("/rtc_disconnect", rtc_disconnect)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()
    job.room.metadata = f"http://127.0.0.1:{port}/rtc_disconnect"

    server = AgentServer(
        ws_url=f"ws://127.0.0.1:{port}",
        api_key="devkey",
        api_secret="secret" * 6,
        job_executor_type=executor_type,
        num_idle_processes=0,
        multiprocessing_context="spawn",
        max_retry=0,
    )
    server.rtc_session()(_status_entrypoint)
    server._simulation = True  # skip the health HTTP server and load checks
    launch_job = ProcPool.launch_job

    async def launch_fake_job(pool: ProcPool, info: RunningJobInfo) -> None:
        info.fake_job = True
        await launch_job(pool, info)

    mock_room = create_mock_room()
    with (
        patch.object(ProcPool, "launch_job", launch_fake_job),
        patch.object(mock_room, "disconnect", new_callable=AsyncMock),
        patch.object(mock_room, "remote_participants", {}),
    ):
        task = asyncio.create_task(server.run(devmode=True))
        try:
            final = await asyncio.wait_for(terminal, timeout=20)
            assert final.job_id == job.id
            assert final.status == expected
            assert disconnect_reasons == [expected_disconnect_reason]
            assert updates[0].status == agent.JobStatus.JS_RUNNING
            if scenario != "raise_before_connect":
                assert any(update.join_timing.connected_us > 0 for update in updates)
        finally:
            await server.aclose()
            await task
            await runner.cleanup()
