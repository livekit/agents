"""A job reads the credentials given to AgentServer in code, and the worker's environment stays
untouched."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading

import pytest

from livekit.agents import AgentServer, JobContext
from livekit.agents.inference._utils import get_default_inference_url, resolve_credentials
from livekit.agents.job import JobExecutorType
from livekit.agents.telemetry._providers import _AuthHeaderProvider

pytestmark = pytest.mark.unit


async def test_a_thread_job_reads_credentials_from_server_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("URL", "API_KEY", "API_SECRET", "INFERENCE_URL", "INFERENCE_API_KEY"):
        monkeypatch.delenv(f"LIVEKIT_{name}", raising=False)
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
    seen: list[tuple[tuple[str, str], str, str, str | None]] = []
    ran = threading.Event()

    server = AgentServer(
        ws_url="wss://project.staging.livekit.cloud",
        api_key="option-key",
        api_secret="option-secret",
    )
    server._update_options(job_executor_type=JobExecutorType.THREAD, num_idle_processes=0)
    server._config.simulation = True

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        credentials = resolve_credentials(None, None)
        url = get_default_inference_url()
        token = _AuthHeaderProvider()()["Authorization"].removeprefix("Bearer ")
        issuer = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))["iss"]
        seen.append((credentials, url, issuer, os.environ.get("LIVEKIT_API_KEY")))
        ran.set()

    started = asyncio.Event()
    server._on_started.append(started.set)
    running = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await server._simulate_job("server-settings", fake_job=True)
        assert await asyncio.to_thread(ran.wait, 10)
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=10)

    staging = "https://agent-gateway.staging.livekit.cloud/v1"
    assert seen == [(("option-key", "option-secret"), staging, "option-key", None)]
    assert "LIVEKIT_API_KEY" not in os.environ
