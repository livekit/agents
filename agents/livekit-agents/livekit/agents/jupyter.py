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

from .cli import _run, proto
from .job import JobExecutorType
from .types import NOT_GIVEN, NotGivenOr
from .worker import WorkerOptions


def run_app(
    opts: WorkerOptions,
    *,
    jupyter_url: NotGivenOr[str] = NOT_GIVEN,
) -> None:
    IN_COLAB = "google.colab" in sys.modules

    nest_asyncio.apply()

    if IN_COLAB:
        from google.colab import userdata  # type: ignore

        if not jupyter_url:
            opts.ws_url = userdata.get("LIVEKIT_URL")
            opts.api_key = userdata.get("LIVEKIT_API_KEY")
            opts.api_secret = userdata.get("LIVEKIT_API_SECRET")
    else:
        opts.ws_url = os.environ.get("LIVEKIT_URL", "")
        opts.api_key = os.environ.get("LIVEKIT_API_KEY", "")
        opts.api_secret = os.environ.get("LIVEKIT_API_SECRET", "")

    if not jupyter_url and (not opts.ws_url or not opts.api_key or not opts.api_secret):
        raise ValueError(
            "Failed to get LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET from environment variables. "  # noqa: E501
            "Alternatively, you can use `jupyter_url`, which generates and uses join tokens for authentication."  # noqa: E501
        )

    if jupyter_url:

        async def fetch_join_tokens(url: str) -> tuple[str, str, str]:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    data = await response.json()
                    return data["livekit_url"], data["user_token"], data["agent_token"]

        try:
            opts.ws_url, user_token, agent_token = asyncio.run(fetch_join_tokens(jupyter_url))
        except Exception as e:
            raise ValueError(
                f"Failed to fetch join tokens via jupyter_url. Error: {e}\n"
                "You can still use your own LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from environment variables instead."  # noqa: E501
            ) from None

        opts.api_key = "fake_jupyter_key"
        opts.api_secret = "fake_jupyter_secret"
    else:
        # manually create the user_token and agent_token using the provided api key and secret
        room_name = f"jupyter-room-{uuid.uuid4()}"
        user_token = (
            api.AccessToken(opts.api_key, opts.api_secret)
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
            api.AccessToken(opts.api_key, opts.api_secret)
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

    display_room(opts.ws_url, user_token)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root.removeHandler(handler)

    opts.job_executor_type = JobExecutorType.THREAD
    opts.drain_timeout = 0
    args = proto.CliArgs(
        opts=opts,
        log_level="DEBUG",
        devmode=True,
        asyncio_debug=False,
        watch=False,
        register=False,
        simulate_job=agent_token,
    )
    _run.run_worker(args, jupyter=True)
