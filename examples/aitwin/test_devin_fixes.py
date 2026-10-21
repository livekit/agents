"""Smoke tests for Devin-review hardening (no LiveKit / AiTwin network)."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_align_error_omits_response_body():
    # Stub heavy deps before import.
    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock  # type: ignore[attr-defined]
    sys.modules.setdefault("openai", fake_openai)

    fake_auth = types.ModuleType("aitwin_auth")
    fake_auth.mint_aitwin_auth_token = MagicMock(return_value="tok")  # type: ignore[attr-defined]
    sys.modules["aitwin_auth"] = fake_auth

    speech = _load_module("aitwin_speech_synth_test", ROOT / "speech_synth.py")

    bad = MagicMock()
    bad.status_code = 500
    bad.ok = False
    bad.text = "echo: secret user transcript should not leak"
    bad.content = b"{}"

    with patch.object(speech.requests, "post", return_value=bad):
        with pytest.raises(RuntimeError) as exc:
            speech.align_visemes_on_aitwin(b"\x00\x00", sample_rate=24000, text="hello")
    msg = str(exc.value)
    assert "500" in msg
    assert "secret user transcript" not in msg


@pytest.mark.unit
def test_synthesize_speech_offloads_align_to_thread():
    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = MagicMock  # type: ignore[attr-defined]
    sys.modules.setdefault("openai", fake_openai)
    fake_auth = types.ModuleType("aitwin_auth")
    fake_auth.mint_aitwin_auth_token = MagicMock(return_value="tok")  # type: ignore[attr-defined]
    sys.modules["aitwin_auth"] = fake_auth

    speech = _load_module("aitwin_speech_synth_test2", ROOT / "speech_synth.py")

    async def _pcm(*_a, **_k):
        return b"\x00\x00", 24000

    async def _run() -> None:
        with (
            patch.object(speech, "synthesize_openai_pcm", side_effect=_pcm),
            patch.object(
                speech,
                "align_visemes_on_aitwin",
                return_value=([{"symbol": "sil", "start": 0, "duration": 0.1}], [], 24000),
            ) as align_mock,
            patch.object(speech.asyncio, "to_thread", wraps=speech.asyncio.to_thread) as to_thread,
            patch.dict("os.environ", {"HEADTTS_URL": ""}, clear=False),
        ):
            bundle = await speech.synthesize_speech("hi")
            assert bundle.visemes
            assert to_thread.called
            assert align_mock.called

    asyncio.run(_run())


@pytest.mark.unit
def test_token_server_requires_secret_and_ignores_client_room(monkeypatch):
    # Minimal stubs so token_server can import.
    fake_livekit = types.ModuleType("livekit")
    fake_api = types.ModuleType("livekit.api")

    class _AccessToken:
        def __init__(self, *_a, **_k):
            pass

        def with_identity(self, *_a, **_k):
            return self

        def with_name(self, *_a, **_k):
            return self

        def with_ttl(self, *_a, **_k):
            return self

        def with_grants(self, *_a, **_k):
            return self

        def to_jwt(self):
            return "jwt-test"

    class _VideoGrants:
        def __init__(self, **_k):
            pass

    class _LiveKitAPI:
        def __init__(self, *_a, **_k):
            self.room = MagicMock()
            self.agent_dispatch = MagicMock()

        async def aclose(self):
            return None

    fake_api.AccessToken = _AccessToken  # type: ignore[attr-defined]
    fake_api.VideoGrants = _VideoGrants  # type: ignore[attr-defined]
    fake_api.LiveKitAPI = _LiveKitAPI  # type: ignore[attr-defined]
    fake_api.CreateRoomRequest = MagicMock  # type: ignore[attr-defined]
    fake_api.CreateAgentDispatchRequest = MagicMock  # type: ignore[attr-defined]
    fake_livekit.api = fake_api  # type: ignore[attr-defined]
    sys.modules["livekit"] = fake_livekit
    sys.modules["livekit.api"] = fake_api

    monkeypatch.setenv("AGENT_NAME", "aitwin-voice-assistant")
    monkeypatch.setenv("AITWIN_AUTH_TOKEN_URL", "https://example.invalid/token")
    monkeypatch.setenv("AITWIN_TWIN_ID", "victoria")
    monkeypatch.setenv("AITWIN_API_BASE", "https://ai.aitwin.me")
    monkeypatch.setenv("TOKEN_SERVER_PORT", "8080")
    monkeypatch.setenv("TOKEN_SERVER_SECRET", "test-secret")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("AITWIN_CLIENT_ID", "cid")
    monkeypatch.setenv("AITWIN_CLIENT_SECRET", "csec")

    # Avoid dotenv overriding monkeypatched env from a real .env.
    with patch("dotenv.load_dotenv", lambda *a, **k: None):
        # Force fresh import
        sys.modules.pop("token_server", None)
        ts = _load_module("token_server", ROOT / "token_server.py")

    client = ts.app.test_client()

    denied = client.post("/token", json={"room": "stolen-room"})
    assert denied.status_code == 401

    with patch.object(ts, "create_room_and_dispatch_agent", return_value=None), patch.object(
        ts, "_mint_aitwin_auth_token", return_value="aitwin-jwt"
    ), patch.object(ts.asyncio, "run", side_effect=lambda coro: None):
        # asyncio.run expects a coroutine; stub create path differently
        async def _ok(_room: str) -> None:
            return None

        with patch.object(ts, "create_room_and_dispatch_agent", side_effect=_ok):
            ok = client.post(
                "/token",
                json={"room": "stolen-room", "identity": "browser-test"},
                headers={"X-Token-Server-Secret": "test-secret"},
            )

    assert ok.status_code == 200
    body = ok.get_json()
    assert body["roomName"].startswith("aitwin-demo-")
    assert body["roomName"] != "stolen-room"
    assert body["aitwinAuthToken"] == "aitwin-jwt"


@pytest.mark.unit
def test_agent_skips_interrupted_ast():
    """Static check: interrupted assistant items must not schedule speech."""
    src = (ROOT / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    text = ast.dump(tree)
    assert "interrupted" in text
    assert "speak_epoch" in src
    assert 'logger.exception("speech synth failed (%d chars)"' in src
    assert "text[:120]" not in src


@pytest.mark.unit
def test_frontend_has_connection_generation():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "connectionGenRef" in app
    assert "stillCurrent" in app
    assert "track.stop()" in app
    assert "X-Token-Server-Secret" in app
    assert "aitwin-demo-${Date.now()}" not in app
