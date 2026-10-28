from __future__ import annotations

import asyncio
import datetime
import logging
import sys
import uuid

import aiohttp
import nest_asyncio  # type: ignore

from livekit import api
from livekit.protocol import models
from livekit.rtc.jupyter import display_room

from . import cli
from .job import JobExecutorType
from .worker import AgentServer


def run_app(server: AgentServer, *, jupyter_url: str | None = None) -> None:
    nest_asyncio.apply()

    if not jupyter_url:
        if "google.colab" in sys.modules:
            from google.colab import userdata  # type: ignore

            server.update_options(
                ws_url=userdata.get("LIVEKIT_URL"),
                api_key=userdata.get("LIVEKIT_API_KEY"),
                api_secret=userdata.get("LIVEKIT_API_SECRET"),
            )
        config = server._config
        if not config.ws_url or not config.api_key or not config.api_secret:
            raise ValueError(
                "Failed to get LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET from environment variables. "  # noqa: E501
                "Alternatively, you can use `jupyter_url`, which generates and uses join tokens for authentication."  # noqa: E501
            )

    server.update_options(job_executor_type=JobExecutorType.THREAD)
    api_key = server._config.api_key
    api_secret = server._config.api_secret

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
        ws_url = server._config.ws_url

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
            if jupyter_url:
                # The join service owns the room; its tokens carry no admin credentials.
                room_info = models.Room(name=room_name)
            else:
                async with api.LiveKitAPI(ws_url, api_key, api_secret) as lk_api:
                    room_info = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))

            await server.simulate_job(
                room_name,
                fake_job=False,
                room_info=room_info,
                token=agent_token,
            )

        asyncio.run_coroutine_threadsafe(simulate_job(), asyncio.get_event_loop())

    options = {"ws_url": ws_url, "api_key": api_key, "api_secret": api_secret}
    cli._run_worker(
        server,
        {
            "log_level": "DEBUG",
            "log_style": "json",
            "dev": False,
            "cli_addr": None,
            "simulation": False,
            "options": {k: v for k, v in options.items() if v},
        },
        unregistered=True,
    )
