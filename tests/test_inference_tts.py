from __future__ import annotations

import base64
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

import livekit.agents.inference.tts as inference_tts
from livekit.agents import APIConnectOptions
from livekit.agents.inference._utils import HEADER_SESSION_ID
from livekit.agents.inference.tts import (
    TTS,
    FallbackModel,
    _normalize_fallback,
    _parse_model_string,
)
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

pytestmark = pytest.mark.unit

_AUDIO = {"type": "output_audio", "audio": base64.b64encode(b"\0\0" * 2400).decode()}


def _make_tts(model: str = "cartesia/sonic-3", **kwargs: Any) -> TTS:
    return TTS(
        model=model,
        **{
            "api_key": "test-key",
            "api_secret": "test-secret-long-enough-for-hs256-signing",
            "base_url": "https://example.livekit.cloud",
            **kwargs,
        },
    )


@pytest.mark.parametrize(
    ("model_str", "expected_model", "expected_voice"),
    [
        ("cartesia", "cartesia", None),
        ("cartesia/sonic", "cartesia/sonic", None),
        ("cartesia:my-voice-id", "cartesia", "my-voice-id"),
        ("cartesia/sonic:my-voice-id", "cartesia/sonic", "my-voice-id"),
        ("rime/mistv2:narrator", "rime/mistv2", "narrator"),
        ("inworld/inworld-tts-1:character", "inworld/inworld-tts-1", "character"),
        # an empty string after the colon still counts as a voice
        ("cartesia/sonic:", "cartesia/sonic", ""),
    ],
)
def test_parse_model_string(
    model_str: str, expected_model: str, expected_voice: str | None
) -> None:
    assert _parse_model_string(model_str) == (expected_model, expected_voice)


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        ("cartesia/sonic", [{"model": "cartesia/sonic", "voice": ""}]),
        ("cartesia/sonic:my-voice", [{"model": "cartesia/sonic", "voice": "my-voice"}]),
        (
            FallbackModel(model="cartesia/sonic", voice="narrator"),
            [{"model": "cartesia/sonic", "voice": "narrator"}],
        ),
        ([], []),
        (
            [
                FallbackModel(model="cartesia/sonic", voice="v1", extra_kwargs={"speed": "slow"}),
                "deepgram:voice2",
                "rime/mist",
            ],
            [
                {"model": "cartesia/sonic", "voice": "v1", "extra_kwargs": {"speed": "slow"}},
                {"model": "deepgram", "voice": "voice2"},
                {"model": "rime/mist", "voice": ""},
            ],
        ),
    ],
)
def test_normalize_fallback(fallback: Any, expected: list[dict[str, Any]]) -> None:
    assert _normalize_fallback(fallback) == expected


@pytest.mark.parametrize(
    ("extra_kwargs", "expected"),
    [
        (None, True),
        ({"word_timestamps": True}, True),
        ({"word_timestamps": False}, False),
        ({"word_timestamps": "no"}, True),
    ],
)
def test_assemblyai_claims_aligned_transcript_unless_turned_off(
    extra_kwargs: dict[str, Any] | None, expected: bool
) -> None:
    # word_boundaries is on by default on the service, and the gateway forwards them
    kwargs = {"extra_kwargs": extra_kwargs} if extra_kwargs is not None else {}
    tts = _make_tts("assemblyai/assemblyai-tts-1", **kwargs)
    assert tts.capabilities.aligned_transcript is expected


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = deque(messages)
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self, *, timeout: float) -> SimpleNamespace:
        if not self._messages:
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT, data=json.dumps(self._messages.popleft())
        )


class _FakePool:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self._websocket = websocket
        self.last_acquire_time = 0.0
        self.last_connection_reused = False

    def connection(self, *, timeout: float):  # noqa: ANN201, ARG002
        websocket = self._websocket

        class _Context:
            async def __aenter__(self):  # noqa: ANN204
                return SimpleNamespace(ws=websocket, session_id="inference_connection")

            async def __aexit__(self, *_exc: object) -> bool:  # noqa: ANN204
                return False

        return _Context()


async def _synthesize(tts: TTS, messages: list[dict[str, Any]], text: str) -> list[Any]:
    websocket = _FakeWebSocket([{"type": "session.created", "session_id": "s"}, *messages])
    tts._pool = _FakePool(websocket)  # type: ignore[assignment]
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0)) as stream:
        stream.push_text(text)
        stream.end_input()
        return [event async for event in stream]


@pytest.mark.parametrize(
    ("model", "extra_kwargs", "words"),
    [
        # cartesia sends bare words and the SDK appends the separator
        ("cartesia/sonic-3", {"add_timestamps": True}, ["hello", "world."]),
        # the gateway already appends assemblyai's separator, so words pass through verbatim
        ("assemblyai/assemblyai-tts-1", {"word_timestamps": True}, ["hello ", "world. "]),
    ],
)
async def test_alignment_words_keep_one_separator(
    model: str, extra_kwargs: dict[str, Any], words: list[str]
) -> None:
    alignment = {
        "type": "output_alignment",
        "words": [
            {"word": words[0], "start": 0.0, "end": 0.2},
            {"word": words[1], "start": 0.2, "end": 0.5},
        ],
    }
    tts = _make_tts(model, extra_kwargs=extra_kwargs)

    events = await _synthesize(tts, [_AUDIO, alignment, {"type": "done"}], "hello world.")

    timed = [t for event in events for t in event.frame.userdata[USERDATA_TIMED_TRANSCRIPT]]
    assert all(event.request_id.startswith("inference_connection_tts_") for event in events)
    assert list(map(str, timed)) == ["hello ", "world. "]
    assert [(w.start_time, w.end_time) for w in timed] == [(0.0, 0.2), (0.2, 0.5)]


async def test_model_switch_mid_stream_keeps_the_stream_dialect() -> None:
    tts = _make_tts("fishaudio/s2.1-pro")
    tts._set_expressive(True)
    websocket = _FakeWebSocket(
        [{"type": "session.created", "session_id": "s"}, _AUDIO, {"type": "done"}]
    )
    tts._pool = _FakePool(websocket)  # type: ignore[assignment]

    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0)) as stream:
        # drama-3 lowers differently; this stream still sends s2.1-pro
        tts.update_options(model="fishaudio/drama-3")
        stream.push_text('<expr type="expression" label="sad"/> Sorry about that.')
        stream.end_input()
        _ = [event async for event in stream]

    packets = [packet for packet in websocket.sent if packet["type"] == "input_transcript"]
    assert [packet["transcript"] for packet in packets] == ["[very sad] Sorry about that. "]
    assert packets[0]["generation_config"]["model"] == "fishaudio/s2.1-pro"


async def test_dial_carries_model_and_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket([])

    class _FakeHTTPSession:
        url = ""
        headers: dict[str, str] = {}

        async def ws_connect(self, url: str, *, headers: dict[str, str]) -> _FakeWebSocket:
            self.url, self.headers = url, headers
            return websocket

    http_session = _FakeHTTPSession()
    monkeypatch.setattr(
        inference_tts, "get_inference_headers", lambda: {HEADER_SESSION_ID: "inference_connection"}
    )
    tts = _make_tts(http_session=http_session)

    connection = await tts._connect_ws(timeout=1.0)

    assert aiohttp.client_reqrep.URL(http_session.url).query["model"] == "cartesia/sonic-3"
    assert http_session.headers[HEADER_SESSION_ID] == "inference_connection"
    assert connection.session_id == "inference_connection"
    assert connection.ws is websocket
