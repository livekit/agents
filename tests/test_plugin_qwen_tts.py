"""Hermetic tests for the Qwen (Alibaba Cloud Model Studio) TTS plugin.

Driven against the in-process fake Model Studio server in ``tests/fake_qwen_realtime.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)
from livekit.plugins.qwen import TTS
from livekit.plugins.qwen.models import REALTIME_BASE_URLS

from .fake_qwen_realtime import FakeTTSServer, audio_delta, pcm

pytestmark = pytest.mark.unit

NO_RETRY = APIConnectOptions(max_retry=0, timeout=5.0)

# 200 ms of 24 kHz mono PCM, the emitter's own frame size.
FRAME_SAMPLES = 4800


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)


@pytest.fixture
async def server() -> AsyncIterator[FakeTTSServer]:
    srv = FakeTTSServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    http = aiohttp.ClientSession()
    yield http
    await http.close()


def make_tts(server: FakeTTSServer, session: aiohttp.ClientSession, **kwargs: Any) -> TTS:
    return TTS(api_key="sk-test", base_url=server.url, http_session=session, **kwargs)


async def speak(tts_: TTS, *tokens: str) -> bytes:
    """Push text through a stream and return the concatenated audio."""
    stream = tts_.stream(conn_options=NO_RETRY)
    for token in tokens:
        stream.push_text(token)
    stream.end_input()
    audio = bytearray()
    try:
        async for event in stream:
            audio.extend(event.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


def session_config(server: FakeTTSServer) -> dict[str, Any]:
    updates = server.events_of_type("session.update")
    assert len(updates) == 1
    config: dict[str, Any] = updates[0]["session"]
    return config


def error_event(**error: Any) -> dict[str, Any]:
    return {"event_id": "srv_e", "type": "error", "error": error}


# --- construction ------------------------------------------------------------------------


def test_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        TTS()


def test_reads_the_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    assert TTS()._api_key == "sk-env"


def test_region_selects_the_public_endpoint() -> None:
    assert TTS(api_key="sk-x")._base_url == REALTIME_BASE_URLS["intl"]
    assert TTS(api_key="sk-x", region="cn")._base_url == REALTIME_BASE_URLS["cn"]


def test_provider_model_and_audio_format_are_declared() -> None:
    tts_ = TTS(api_key="sk-x")
    assert tts_.provider == "Qwen"
    assert tts_.model == "qwen3-tts-flash-realtime"
    assert tts_.sample_rate == 24000
    assert tts_.num_channels == 1
    assert tts_.capabilities.streaming is True


# --- handshake ---------------------------------------------------------------------------


async def test_model_is_sent_in_the_query_string(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert server.query.get("model") == "qwen3-tts-flash-realtime"


async def test_api_key_is_sent_as_a_bearer_token(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert server.headers.get("Authorization") == "Bearer sk-test"


async def test_session_requests_raw_pcm_at_24k(server, session) -> None:
    # The emitter is told audio/pcm at 24 kHz, so the socket must deliver exactly that;
    # an mp3 default would be decoded as garbage.
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    config = session_config(server)
    assert config["response_format"] == "pcm"
    assert config["sample_rate"] == 24000


async def test_voice_defaults_to_a_bilingual_one(server, session) -> None:
    # Cherry speaks Mandarin and English; the dialect voices do not, and a conversation
    # can switch language mid-session.
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert session_config(server)["voice"] == "Cherry"


async def test_voice_is_configurable(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session, voice="Ethan"), "hello")
    assert session_config(server)["voice"] == "Ethan"


async def test_language_type_defaults_to_auto(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert session_config(server)["language_type"] == "Auto"


async def test_language_type_is_configurable(server, session) -> None:
    # Naming the language measurably improves quality on single-language text, per Model
    # Studio's own guidance.
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session, language_type="Chinese"), "你好")
    assert session_config(server)["language_type"] == "Chinese"


async def test_speech_rate_is_only_sent_when_set(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert "speech_rate" not in session_config(server)


async def test_speech_rate_is_forwarded_when_set(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session, speech_rate=1.2), "hello")
    assert session_config(server)["speech_rate"] == 1.2


async def test_server_commit_mode_lets_the_model_pick_sentence_ends(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert session_config(server)["mode"] == "server_commit"


async def test_update_options_applies_to_the_next_stream(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    tts_ = make_tts(server, session)
    tts_.update_options(voice="Serena", language_type="English", speech_rate=0.9)
    await speak(tts_, "hello")
    config = session_config(server)
    assert config["voice"] == "Serena"
    assert config["language_type"] == "English"
    assert config["speech_rate"] == 0.9


# --- text and audio flow -----------------------------------------------------------------


async def test_pushed_text_is_streamed_up_incrementally(server, session) -> None:
    # One append per token, not one buffered blob at the end: that is what lets the model
    # start speaking before the LLM has finished.
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "Tell me ", "about ", "your project.")
    appends = server.events_of_type("input_text_buffer.append")
    assert [e["text"] for e in appends] == ["Tell me ", "about ", "your project."]
    assert server.text == "Tell me about your project."


async def test_audio_deltas_become_playable_frames(server, session) -> None:
    payload = pcm(FRAME_SAMPLES)
    server.script(audio_delta(payload))
    audio = await speak(make_tts(server, session), "hello")
    assert audio == payload


async def test_audio_arriving_in_several_deltas_is_concatenated_in_order(server, session) -> None:
    first = pcm(FRAME_SAMPLES, value=1)
    second = pcm(FRAME_SAMPLES, value=2)
    server.script(audio_delta(first), audio_delta(second))
    audio = await speak(make_tts(server, session), "hello")
    assert audio == first + second


async def test_finish_is_sent_when_input_ends(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    await speak(make_tts(server, session), "hello")
    assert len(server.events_of_type("session.finish")) == 1


async def test_audio_flushed_after_finish_is_not_lost(session) -> None:
    # The documented behaviour is that the server drains remaining audio after
    # session.finish; closing the socket on send would clip the tail.
    tail = pcm(FRAME_SAMPLES, value=3)
    server = FakeTTSServer(audio_on_finish=[tail])
    await server.start()
    try:
        audio = await speak(make_tts(server, session), "hello")
    finally:
        await server.stop()
    assert audio == tail


# --- failures ----------------------------------------------------------------------------


async def test_server_error_event_surfaces_as_an_api_error(server, session) -> None:
    server.script(error_event(code="DataInspectionFailed", message="flagged text"))
    with pytest.raises(APIStatusError) as exc_info:
        await speak(make_tts(server, session), "hello")
    assert "flagged text" in str(exc_info.value)


async def test_a_request_error_is_not_retried(server, session) -> None:
    server.script(error_event(type="invalid_request_error", message="unknown voice"))
    with pytest.raises(APIStatusError) as exc_info:
        await speak(make_tts(server, session), "hello")
    assert exc_info.value.retryable is False


async def test_hangup_before_finish_surfaces_as_a_connection_error(server, session) -> None:
    server.close_after_script()
    with pytest.raises(APIConnectionError):
        await speak(make_tts(server, session), "hello")


async def test_unreachable_endpoint_surfaces_as_a_connection_error(session) -> None:
    tts_ = TTS(
        api_key="sk-test",
        base_url="ws://127.0.0.1:1/api-ws/v1/realtime",
        http_session=session,
    )
    with pytest.raises(APIConnectionError):
        await speak(tts_, "hello")


async def test_a_wedged_upstream_times_out_instead_of_hanging(server, session) -> None:
    # A server holding the socket open without answering session.finish would stall the
    # agent mid-turn; the connect timeout cannot see it.
    server.ignore_finish()
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(
            speak(make_tts(server, session, finish_timeout=0.3), "hello"), timeout=10
        )


# --- one-shot synthesize -----------------------------------------------------------------


async def test_synthesize_returns_audio_for_a_whole_string(server, session) -> None:
    payload = pcm(FRAME_SAMPLES)
    server.script(audio_delta(payload))
    audio = bytearray()
    stream = make_tts(server, session).synthesize("hello", conn_options=NO_RETRY)
    try:
        async for event in stream:
            audio.extend(event.frame.data.tobytes())
    finally:
        await stream.aclose()
    # The non-streaming emitter appends a short silent frame of its own to carry is_final,
    # so assert on the audio rather than the exact length.
    assert bytes(audio[: len(payload)]) == payload
    assert set(audio[len(payload) :]) <= {0}


async def test_synthesize_sends_the_text_once(server, session) -> None:
    server.script(audio_delta(pcm(FRAME_SAMPLES)))
    stream = make_tts(server, session).synthesize("hello", conn_options=NO_RETRY)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
    assert server.text == "hello"
