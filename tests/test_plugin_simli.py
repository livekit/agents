"""Tests for the Simli plugin: starting a session.

Covers the retry behaviour (same shape as the Anam fix in #7314), that a failed
start logs and returns instead of crashing inside its own error handler, and the
body SimliConfig sends to POST /compose/token.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("simli")

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)
_TOKEN_BODY = json.dumps({"session_token": "simli-session-token"})


class _Response:
    """Usable both as `await session.post(...)` and `async with session.post(...)`."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.ok = status < 400
        self._body = body

    async def text(self) -> str:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=self.status
            )

    def __await__(self):
        async def _self() -> _Response:
            return self

        return _self().__await__()

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next step in `script`: a status code (body is the
    token response for 2xx), or an exception instance to raise. The last step repeats."""

    timeout = aiohttp.ClientTimeout(total=300)

    def __init__(self, script: list[int | BaseException], token_body: str = _TOKEN_BODY) -> None:
        self._script = list(script)
        self._token_body = token_body
        self.posts = 0
        self.timeouts: list[aiohttp.ClientTimeout] = []
        self.requests: list[tuple[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        self.requests.append((url, kwargs.get("json")))
        if "timeout" in kwargs:
            self.timeouts.append(kwargs["timeout"])
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step, self._token_body if step < 400 else f"status {step}")


def _avatar(monkeypatch: pytest.MonkeyPatch, session: _ScriptedSession, config: Any = None):
    from livekit.agents import utils
    from livekit.plugins.simli import AvatarSession, SimliConfig

    monkeypatch.setattr(utils.http_context, "http_session", lambda: session)
    return AvatarSession(
        simli_config=config or SimliConfig(api_key="test-key", face_id="face"),
        api_url="http://simli.test",
        conn_options=_OPTS,
    )


def _patch_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let AvatarSession.start() run without a job or a connected room."""
    from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
    from livekit.plugins.simli import avatar as simli_avatar

    async def _base_start(self, agent_session, room):
        return None

    monkeypatch.setattr(BaseAvatarSession, "start", _base_start)
    monkeypatch.setattr(
        simli_avatar,
        "get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent"),
    )


async def _start(avatar: Any, agent_session: Any) -> None:
    await avatar.start(
        agent_session,
        SimpleNamespace(name="room"),  # type: ignore[arg-type]
        livekit_url="wss://livekit.test",
        livekit_api_key="key",
        livekit_api_secret="secret" * 8,
    )


async def test_client_error_is_not_retried_and_keeps_its_status(monkeypatch):
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.posts == 1
    assert exc_info.value.status_code == 401


async def test_server_error_is_retried_until_success(monkeypatch):
    session = _ScriptedSession([503, 503, 200])

    body = await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert body == _TOKEN_BODY
    assert session.posts == 3


async def test_request_timeout_keeps_the_session_total_and_uses_connect_option(monkeypatch):
    session = _ScriptedSession([200])

    await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.timeouts[0].total == 300
    assert session.timeouts[0].sock_connect == _OPTS.timeout


async def test_network_error_is_retried_and_chained(monkeypatch):
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError) as exc_info:
        await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.posts == _OPTS.max_retry + 1
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)


async def test_failed_connect_request_logs_and_returns(monkeypatch, caplog):
    """If the second request fails before a response exists, the old error handler
    referenced the unassigned response and raised UnboundLocalError. start() should
    log the failure, leave the agent's audio output alone, and return."""
    _patch_start(monkeypatch)

    session = _ScriptedSession([200, aiohttp.ClientConnectionError("connection reset")])
    avatar = _avatar(monkeypatch, session)
    agent_session = MagicMock()

    with caplog.at_level(logging.ERROR, logger="livekit.plugins.simli"):
        await _start(avatar, agent_session)

    agent_session.output.replace_audio_tail.assert_not_called()
    assert any("simli" in r.getMessage() for r in caplog.records)
    assert not any("simli-session-token" in r.getMessage() for r in caplog.records)


def test_default_body_sends_the_plain_face_id_and_leaves_framing_to_simli():
    """Without an emotion the face ID goes as is and Simli applies the face's default
    emotion; no Trinity-2 field is sent, so Simli's defaults apply."""
    from livekit.plugins.simli import SimliConfig

    body = SimliConfig(api_key="test-key", face_id="face").create_json()

    assert body == {
        "faceId": "face",
        "handleSilence": True,
        "maxSessionLength": 600,
        "maxIdleTime": 30,
    }


def test_explicit_emotion_is_appended_to_the_face_id():
    from livekit.plugins.simli import SimliConfig

    keyword = SimliConfig(api_key="test-key", face_id="face", emotion_id="emotion")
    positional = SimliConfig("test-key", "face", "emotion")

    assert keyword.create_json()["faceId"] == "face/emotion"
    assert positional.create_json()["faceId"] == "face/emotion"
    assert SimliConfig("test-key", "face", "").create_json()["faceId"] == "face"


def test_trinity2_options_are_sent_when_set():
    from livekit.plugins.simli import CameraSetting, SimliConfig

    config = SimliConfig(
        api_key="test-key",
        face_id="face",
        max_session_length=120,
        max_idle_time=60,
        streaming_width=1366,
        streaming_height=768,
        avatar_position_x=0.3,
        avatar_position_y=1.0,
        camera=CameraSetting(zoom=1.2, orbit_y_off=15),
        background_id="6f1c2a9e-1b2c-4d3e-8f90-0123456789ab",
        foreground_id="0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
    )

    assert config.create_json() == {
        "faceId": "face",
        "handleSilence": True,
        "maxSessionLength": 120,
        "maxIdleTime": 60,
        "streaming_width": 1366,
        "streaming_height": 768,
        "avatar_position_x": 0.3,
        "avatar_position_y": 1.0,
        "camera": {"zoom": 1.2, "orbit_y_off": 15},
        "background_id": "6f1c2a9e-1b2c-4d3e-8f90-0123456789ab",
        "foreground_id": "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
    }


def test_unset_trinity2_options_are_left_out():
    from livekit.plugins.simli import CameraSetting, SimliConfig

    body = SimliConfig(
        api_key="test-key",
        face_id="face",
        streaming_width=1366,
        avatar_position_y=0.0,
        camera=CameraSetting(pos_x_off=0.0),
    ).create_json()

    assert body["streaming_width"] == 1366
    assert body["avatar_position_y"] == 0.0
    assert body["camera"] == {"pos_x_off": 0.0}
    for name in ("streaming_height", "avatar_position_x", "background_id", "foreground_id"):
        assert name not in body


def test_camera_can_be_a_dict():
    """As in Simli's API docs, and in create_json() overrides written before
    SimliConfig had a camera field."""
    from livekit.plugins.simli import SimliConfig

    body = SimliConfig(api_key="test-key", face_id="face", camera={"zoom": 1.2}).create_json()

    assert body["camera"] == {"zoom": 1.2}


async def test_start_posts_the_config_body_to_compose_token(monkeypatch):
    from livekit.plugins.simli import SimliConfig, avatar as simli_avatar

    _patch_start(monkeypatch)
    monkeypatch.setattr(simli_avatar, "DataStreamAudioOutput", MagicMock())
    config = SimliConfig(
        api_key="test-key", face_id="face", streaming_width=1366, streaming_height=768
    )
    session = _ScriptedSession([200])
    agent_session = MagicMock()

    await _start(_avatar(monkeypatch, session, config), agent_session)

    url, body = session.requests[0]
    assert url == "http://simli.test/compose/token"
    assert body == config.create_json()
    assert body["faceId"] == "face"
    assert session.requests[1][0] == "http://simli.test/integrations/livekit/agents"
    agent_session.output.replace_audio_tail.assert_called_once()


async def test_start_logs_the_token_warnings(monkeypatch, caplog):
    """Simli still mints the token when it drops an option, e.g. a background_id that
    isn't the account's, and says so in the response's warnings."""
    from livekit.plugins.simli import avatar as simli_avatar

    _patch_start(monkeypatch)
    monkeypatch.setattr(simli_avatar, "DataStreamAudioOutput", MagicMock())
    warning = "background_id 6f1c2a9e-1b2c-4d3e-8f90-0123456789ab is not one of yours"
    token_body = json.dumps({"session_token": "simli-session-token", "warnings": [warning]})
    session = _ScriptedSession([200], token_body=token_body)
    agent_session = MagicMock()

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.simli"):
        await _start(_avatar(monkeypatch, session), agent_session)

    assert [r.warnings for r in caplog.records if hasattr(r, "warnings")] == [[warning]]
    agent_session.output.replace_audio_tail.assert_called_once()
