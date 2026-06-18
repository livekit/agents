from __future__ import annotations

import asyncio
import datetime
import logging
import os
import sys
import uuid

import aiohttp
import nest_asyncio  # type: ignore

from livekit import api
from livekit.rtc.jupyter import display_room

from .cli import cli, proto
from .job import JobExecutorType
from .worker import AgentServer, WorkerOptions


def run_app(server: AgentServer | WorkerOptions, *, jupyter_url: str | None = None) -> None:
    nest_asyncio.apply()

    if isinstance(server, WorkerOptions):
        IN_COLAB = "google.colab" in sys.modules

        if IN_COLAB:
            from google.colab import userdata  # type: ignore

            if not jupyter_url:
                server.ws_url = userdata.get("LIVEKIT_URL")
                server.api_key = userdata.get("LIVEKIT_API_KEY")
                server.api_secret = userdata.get("LIVEKIT_API_SECRET")
        else:
            server.ws_url = server.ws_url or os.environ.get("LIVEKIT_URL", "")
            server.api_key = server.api_key or os.environ.get("LIVEKIT_API_KEY", "")
            server.api_secret = server.api_secret or os.environ.get("LIVEKIT_API_SECRET", "")

        if not jupyter_url and (not server.ws_url or not server.api_key or not server.api_secret):
            raise ValueError(
                "Failed to get LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET from environment variables. "  # noqa: E501
                "Alternatively, you can use `jupyter_url`, which generates and uses join tokens for authentication."  # noqa: E501
            )

        server = AgentServer.from_server_options(server)

    server._job_executor_type = JobExecutorType.THREAD

    # create user and agent tokens
    if jupyter_url:

        async def fetch_join_tokens(url: str) -> tuple[str, str, str]:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    data = await response.json()
                    return data["livekit_url"], data["user_token"], data["agent_token"]

        try:
            ws_url, user_token, agent_token = asyncio.run(fetch_join_tokens(jupyter_url))
            claims = api.TokenVerifier().verify(agent_token, verify_signature=False)
            if claims.video and claims.video.room:
                room_name = claims.video.room
            else:
                room_name = f"jupyter-room-{uuid.uuid4()}"
        except Exception as e:
            raise ValueError(
                f"Failed to fetch join tokens via jupyter_url. Error: {e}\n"
                "You can still use your own LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from environment variables instead."  # noqa: E501
            ) from None

    else:
        ws_url = server._ws_url
        api_key = server._api_key
        api_secret = server._api_secret

        # manually create the user_token and agent_token using the provided api key and secret
        room_name = f"jupyter-room-{uuid.uuid4()}"
        user_token = (
            api.AccessToken(api_key, api_secret)
            .with_identity("user-jupyter")
            .with_grants(
                api.VideoGrants(
                    can_publish=True, can_subscribe=True, room_join=True, room=room_name
                )
            )
            .with_ttl(datetime.timedelta(minutes=1))
            .to_jwt()
        )

        agent_token = (
            api.AccessToken(api_key, api_secret)
            .with_identity("agent-jupyter")
            .with_kind("agent")
            .with_grants(
                api.VideoGrants(
                    agent=True,
                    can_publish=True,
                    can_subscribe=True,
                    room_join=True,
                    can_update_own_metadata=True,
                    room=room_name,
                )
            )
            .with_ttl(datetime.timedelta(minutes=1))
            .to_jwt()
        )

    display_room(ws_url, user_token)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root.removeHandler(handler)

    @server.once("worker_started")
    def _simulate_job() -> None:
        async def simulate_job() -> None:
            async with api.LiveKitAPI(ws_url, api_key, api_secret) as lk_api:
                room_info = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))

            await server.simulate_job(
                room_name,
                fake_job=False,
                room_info=room_info,
                token=agent_token,
            )

        asyncio.run_coroutine_threadsafe(simulate_job(), asyncio.get_event_loop())

    args = proto.CliArgs(
        log_level="DEBUG", devmode=True, url=ws_url, api_key=api_key, api_secret=api_secret
    )
    cli._run_worker(server, args, jupyter=True)
