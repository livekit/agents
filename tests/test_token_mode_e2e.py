"""Token mode, end to end, against a real LiveKit server.

The worker registers with a worker token and holds no API key or secret. Its job joins the
room and calls a server API; the token for that call comes from the server, over the worker
connection (CreateJobToken). The test skips on a server before agent protocol 2.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from unittest.mock import patch

import pytest

from livekit import api
from livekit.agents import AgentServer, JobContext

from .lk_server import LK_API_KEY, LK_API_SECRET, LK_URL, livekit_server  # noqa: F401

pytestmark = pytest.mark.unit

_AGENT_NAME = "token-mode-e2e"
_METADATA = "set by the job in token mode"


async def _entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    await ctx.api.room.update_room_metadata(
        api.UpdateRoomMetadataRequest(room=ctx.room.name, metadata=_METADATA)
    )


async def _room_metadata(lk: api.LiveKitAPI, room: str) -> str:
    rooms = (await lk.room.list_rooms(api.ListRoomsRequest(names=[room]))).rooms
    return rooms[0].metadata if rooms else ""


async def test_token_mode_job_calls_the_server_api(livekit_server: None) -> None:  # noqa: F811
    room = f"token-mode-{uuid.uuid4().hex[:8]}"
    worker_token = (
        api.AccessToken(LK_API_KEY, LK_API_SECRET).with_grants(api.VideoGrants(agent=True)).to_jwt()
    )
    env = {"LIVEKIT_API_KEY": "", "LIVEKIT_API_SECRET": "", "LIVEKIT_TOKEN": worker_token}

    with patch.dict(os.environ, env):
        server = AgentServer(ws_url=LK_URL, num_idle_processes=0, max_retry=3)
        assert server._token_mode
        server.rtc_session(agent_name=_AGENT_NAME)(_entrypoint)
        server._simulation = True  # skip the health HTTP server
        registered = asyncio.Event()
        server.on("worker_registered", lambda *_: registered.set())
        run = asyncio.create_task(server.run(devmode=True))
        try:
            wait = asyncio.create_task(registered.wait())
            await asyncio.wait({run, wait}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
            wait.cancel()
            if run.done() and "does not support token mode" in str(run.exception()):
                pytest.skip("the LiveKit server is before agent protocol 2")
            assert registered.is_set(), "the worker did not register"

            async with api.LiveKitAPI(LK_URL, LK_API_KEY, LK_API_SECRET) as lk:
                await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(agent_name=_AGENT_NAME, room=room)
                )
                async with asyncio.timeout(30):
                    while await _room_metadata(lk, room) != _METADATA:
                        await asyncio.sleep(0.2)
                await lk.room.delete_room(api.DeleteRoomRequest(room=room))
        finally:
            await server.aclose()
            with contextlib.suppress(RuntimeError):
                await run
