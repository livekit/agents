"""Integration tests for the Avaz TTS plugin (dashboard WebSocket)."""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.plugin("avaz")

_INTEGRATION_ENV = ("AVAZ_API_KEY", "AVAZ_BASE_URL", "AVAZ_AGENT_MODEL_ID")


def _integration_env_ready() -> bool:
    return all(os.environ.get(name, "").strip() for name in _INTEGRATION_ENV)


def _skip_unless_dashboard_env() -> None:
    if not _integration_env_ready():
        pytest.skip(
            "Set AVAZ_API_KEY, AVAZ_BASE_URL, and AVAZ_AGENT_MODEL_ID for dashboard integration tests"
        )


def _dashboard_ws_url() -> str:
    from livekit.plugins.avaz.tts import _derive_ws_url_from_base

    base = os.environ["AVAZ_BASE_URL"].rstrip("/")
    return _derive_ws_url_from_base(base)


def _ws_host_port(ws_url: str) -> tuple[str, int]:
    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    return host, port


def _server_reachable(ws_url: str, timeout: float = 3.0) -> bool:
    host, port = _ws_host_port(ws_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.asyncio
async def test_avaz_tts_plugin_stream() -> None:
    from livekit.plugins.avaz import TTS

    _skip_unless_dashboard_env()
    uri = _dashboard_ws_url()
    if not _server_reachable(uri):
        pytest.skip(f"Avaz dashboard not reachable at {uri}")

    engine = TTS(
        api_key=os.environ["AVAZ_API_KEY"],
        base_url=os.environ["AVAZ_BASE_URL"],
        model_id=os.environ["AVAZ_AGENT_MODEL_ID"],
    )
    stream = engine.stream()
    stream.push_text("Merhaba, bu bir plugin testidir.")
    stream.end_input()

    frames = 0
    total_pcm = 0
    async for ev in stream:
        frames += 1
        total_pcm += len(ev.frame.data)

    await stream.aclose()
    await engine.aclose()

    assert frames >= 1
    assert total_pcm > 0
