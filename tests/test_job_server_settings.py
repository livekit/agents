"""A job reads the credentials given to AgentServer in code, and the worker's environment stays
untouched."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from livekit.agents import AgentServer, JobContext, inference
from livekit.agents.inference import get_default_inference_url, resolve_credentials
from livekit.agents.job import JobExecutorType
from livekit.agents.storage import SQLiteStorage
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit


async def test_local_console_job_needs_no_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_WORKER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    server = AgentServer(storage=SQLiteStorage(directory=tmp_path))
    server._update_options(job_executor_type=JobExecutorType.THREAD, num_idle_processes=0)
    started, ran = asyncio.Event(), threading.Event()
    server._on_started.append(started.set)

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        assert ctx._info.token == ""
        ran.set()
        ctx.shutdown("done")

    running = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert server.thread_service_url is None
        await server._simulate_job("console-without-keys", fake_job=True)
        assert await asyncio.to_thread(ran.wait, 5)
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=5)


async def test_a_thread_job_reads_credentials_from_server_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("URL", "API_KEY", "API_SECRET", "INFERENCE_URL", "INFERENCE_API_KEY"):
        monkeypatch.delenv(f"LIVEKIT_{name}", raising=False)
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
    seen: list[tuple[tuple[str, str], str, str | None]] = []
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
        inference.AdaptiveInterruptionDetector()  # resolves from the server's options too
        url = get_default_inference_url()
        seen.append((credentials, url, os.environ.get("LIVEKIT_API_KEY")))
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
    assert seen == [(("option-key", "option-secret"), staging, None)]
    assert "LIVEKIT_API_KEY" not in os.environ


def test_inference_credentials_read_the_inference_environment_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "server-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "server-secret")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "inference-key")
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "http://localhost:8080")
    creds = inference.Credentials.resolve(NOT_GIVEN, None, "mine")
    assert creds == inference.Credentials("http://localhost:8080", "inference-key", "mine")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "")
    with pytest.raises(ValueError, match="api_key is required"):
        resolve_credentials(None, None)


def test_inference_headers_outside_a_job_identify_only_the_sdk() -> None:
    headers = inference.get_inference_headers(inference_class="low")
    assert list(headers) == ["User-Agent", "X-LiveKit-Inference-Priority"]
    assert headers["User-Agent"].startswith("LiveKit Agents/")


def test_openai_realtime_resolves_its_key_and_url_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.llm._realtime import RealtimeModel

    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
        RealtimeModel()
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        RealtimeModel.with_azure(azure_deployment="d", api_key="k")
    with pytest.raises(ValueError, match="mutually exclusive"):
        RealtimeModel.with_azure(
            azure_deployment="d", api_key="k", base_url="https://b", azure_endpoint="https://e"
        )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert RealtimeModel().provider == "api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    assert RealtimeModel().provider == "proxy.example"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://az/")
    with pytest.raises(ValueError, match="Missing credentials"):
        RealtimeModel.with_azure(azure_deployment="d")
    assert RealtimeModel.with_azure(azure_deployment="d", api_key="k").model
