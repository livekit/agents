"""Unit tests for the Blaze plugin (config, init, utils, STT, TTS, LLM)."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from collections.abc import Callable
from unittest.mock import AsyncMock

import httpx
import pytest

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APITimeoutError,
)
from livekit.agents.llm import ChatContext, FunctionCallOutput, function_tool
from livekit.agents.stt import SpeechEventType
from livekit.plugins.blaze._config import BlazeConfig
from livekit.plugins.blaze._utils import (
    apply_normalization_rules,
    convert_pcm_to_wav,
    effective_connect_timeout,
)
from livekit.plugins.blaze.llm import LLM, LLMStream
from livekit.plugins.blaze.stt import STT
from livekit.plugins.blaze.tts import (
    TTS,
    _apply_pcm16_fade,
    _generate_silence,
    _normalize_text,
    _WSStreamGuard,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_blaze_public_exports() -> None:
    from livekit.plugins import blaze

    assert blaze.__version__
    assert blaze.BlazeConfig is not None
    assert blaze.STT is not None
    assert blaze.TTS is not None
    assert blaze.LLM is not None
    assert blaze.ChunkedStream is not None
    assert blaze.LLMStream is not None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_blaze_config_explicit_values() -> None:
    config = BlazeConfig(
        api_url="https://custom.example.com",
        api_token="secret",
        stt_timeout=12.0,
        tts_timeout=34.0,
        tts_stream_timeout=400.0,
        llm_timeout=56.0,
    )

    assert config.api_url == "https://custom.example.com"
    assert config.api_token == "secret"
    assert config.stt_timeout == 12.0
    assert config.tts_timeout == 34.0
    assert config.tts_stream_timeout == 400.0
    assert config.llm_timeout == 56.0


def test_blaze_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLAZE_API_URL", "https://env.example.com")
    monkeypatch.setenv("BLAZE_API_TOKEN", "env-token")
    monkeypatch.setenv("BLAZE_STT_TIMEOUT", "11")
    monkeypatch.setenv("BLAZE_TTS_TIMEOUT", "22")
    monkeypatch.setenv("BLAZE_TTS_STREAM_TIMEOUT", "333")
    monkeypatch.setenv("BLAZE_LLM_TIMEOUT", "44")

    config = BlazeConfig()

    assert config.api_url == "https://env.example.com"
    assert config.api_token == "env-token"
    assert config.stt_timeout == 11.0
    assert config.tts_timeout == 22.0
    assert config.tts_stream_timeout == 333.0
    assert config.llm_timeout == 44.0


def test_blaze_config_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "BLAZE_API_URL",
        "BLAZE_API_TOKEN",
        "BLAZE_STT_TIMEOUT",
        "BLAZE_TTS_TIMEOUT",
        "BLAZE_TTS_STREAM_TIMEOUT",
        "BLAZE_LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)

    config = BlazeConfig()

    assert config.api_url == "https://api.blaze.vn"
    assert config.api_token == ""
    assert config.stt_timeout == 30.0
    assert config.tts_timeout == 60.0
    assert config.tts_stream_timeout == 300.0
    assert config.llm_timeout == 60.0


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_convert_pcm_to_wav_produces_valid_header() -> None:
    pcm = b"\x01\x00" * 100
    wav = convert_pcm_to_wav(pcm, sample_rate=16000, num_channels=1)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert b"data" in wav
    assert len(wav) > len(pcm)


def test_convert_pcm_to_wav_embeds_sample_rate() -> None:
    pcm = b"\x00\x00" * 8
    wav = convert_pcm_to_wav(pcm, sample_rate=24000, num_channels=2)

    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    channels = struct.unpack_from("<H", wav, 22)[0]
    assert sample_rate == 24000
    assert channels == 2


def test_apply_normalization_rules_no_rules_returns_original() -> None:
    assert apply_normalization_rules("Hello API", None) == "Hello API"
    assert apply_normalization_rules("Hello API", {}) == "Hello API"


def test_apply_normalization_rules_applies_replacements() -> None:
    rules = {"API": "A P I", "USD": "đô la"}
    assert apply_normalization_rules("USD API", rules) == "đô la A P I"


def test_apply_normalization_rules_prefers_longer_patterns_first() -> None:
    rules = {"$": "dollar", "USD": "đô la Mỹ"}
    assert apply_normalization_rules("USD price is $5", rules) == "đô la Mỹ price is dollar5"


def test_apply_normalization_rules_skips_empty_patterns() -> None:
    rules = {"": "ignored", "API": "A P I"}
    assert apply_normalization_rules("API", rules) == "A P I"


def test_effective_connect_timeout_uses_plugin_default_for_framework_default() -> None:
    assert effective_connect_timeout(DEFAULT_API_CONNECT_OPTIONS, 60.0) == 60.0


def test_effective_connect_timeout_honors_explicit_override() -> None:
    assert effective_connect_timeout(APIConnectOptions(timeout=5.0), 60.0) == 5.0


def test_effective_connect_timeout_zero_falls_back_to_plugin_default() -> None:
    assert effective_connect_timeout(APIConnectOptions(timeout=0), 30.0) == 30.0


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


def _pcm_frame(
    *,
    samples: int = 1600,
    sample_rate: int = 16000,
    byte_val: int = 0x11,
) -> rtc.AudioFrame:
    data = bytes([byte_val, byte_val]) * samples
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _QueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._responder(request)


def _make_stt(
    responder: Callable[[httpx.Request], httpx.Response],
    *,
    normalization_rules: dict[str, str] | None = None,
) -> STT:
    config = BlazeConfig(api_url="https://api.example.com", api_token="test-token")
    stt = STT(config=config, normalization_rules=normalization_rules)
    stt._client = httpx.AsyncClient(transport=_QueueTransport(responder))
    return stt


@pytest.mark.asyncio
async def test_stt_provider_and_sample_rate() -> None:
    stt = _make_stt(lambda _req: httpx.Response(200, json={"transcription": "ok"}))
    assert stt.provider == "Blaze"
    assert stt.sample_rate == 16000
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_empty_buffer_returns_empty_event() -> None:
    stt = _make_stt(lambda _req: httpx.Response(200, json={"transcription": "unused"}))

    event = await stt._recognize_impl([], conn_options=APIConnectOptions(max_retry=0))

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives == []
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_returns_transcription_with_normalization() -> None:
    stt = _make_stt(
        lambda _req: httpx.Response(200, json={"transcription": "API", "confidence": 0.88}),
        normalization_rules={"API": "A P I"},
    )

    event = await stt._recognize_impl(_pcm_frame(), conn_options=APIConnectOptions(max_retry=0))

    assert event.alternatives[0].text == "A P I"
    assert event.alternatives[0].confidence == 0.88
    assert stt._pending_pcm == b""
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_buffers_empty_transcription_for_next_call() -> None:
    request_count = 0

    def responder(_req: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, json={"transcription": "", "confidence": 0.0})
        return httpx.Response(200, json={"transcription": "xin chào", "confidence": 0.9})

    stt = _make_stt(responder)
    frame_a = _pcm_frame(samples=800, byte_val=0x01)
    frame_b = _pcm_frame(samples=800, byte_val=0x02)

    first = await stt._recognize_impl(frame_a, conn_options=APIConnectOptions(max_retry=0))
    assert first.alternatives[0].text == ""
    assert stt._pending_pcm != b""
    assert stt._pending_empty_count == 1

    second = await stt._recognize_impl(frame_b, conn_options=APIConnectOptions(max_retry=0))
    assert second.alternatives[0].text == "xin chào"
    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    assert request_count == 2
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_discards_pending_after_max_empty_segments() -> None:
    def responder(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcription": "", "confidence": 0.0})

    stt = _make_stt(responder)
    stt._max_pending_segments = 2

    for _ in range(3):
        await stt._recognize_impl(
            _pcm_frame(samples=400), conn_options=APIConnectOptions(max_retry=0)
        )

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_discards_pending_when_duration_limit_exceeded() -> None:
    def responder(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcription": "", "confidence": 0.0})

    stt = _make_stt(responder)
    stt._max_pending_duration = 0.05

    # ~0.1s of audio at 16kHz mono PCM16
    large_frame = _pcm_frame(samples=1600)

    await stt._recognize_impl(large_frame, conn_options=APIConnectOptions(max_retry=0))

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_clears_stale_pending_buffer_after_idle_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt = _make_stt(
        lambda _req: httpx.Response(200, json={"transcription": "ok", "confidence": 1.0})
    )
    stt._pending_pcm = b"\x01\x00" * 100
    stt._pending_empty_count = 1
    stt._last_recognize_time = time.monotonic() - 20.0
    stt._pending_idle_timeout = 10.0

    await stt._recognize_impl(_pcm_frame(samples=200), conn_options=APIConnectOptions(max_retry=0))

    assert stt._pending_pcm == b""
    assert stt._pending_empty_count == 0
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_raises_api_status_error_on_http_failure() -> None:
    from livekit.agents import APIStatusError

    stt = _make_stt(lambda _req: httpx.Response(500, text="server error"))

    with pytest.raises(APIStatusError, match="STT service error 500"):
        await stt._recognize_impl(_pcm_frame(), conn_options=APIConnectOptions(max_retry=0))

    await stt.aclose()


def test_stt_with_streaming_requires_vad_instance() -> None:
    stt = STT(config=BlazeConfig(api_url="https://api.example.com"))

    with pytest.raises(TypeError, match="Expected a VAD instance"):
        stt.with_streaming(object())  # type: ignore[arg-type]


def test_stt_default_models() -> None:
    stt = STT(config=BlazeConfig(api_url="http://localhost", api_token="tok"))
    assert stt.model == "v2.0"
    assert stt.stream_model == "stt-stream-1.5"
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is True
    assert stt._ws_url == "ws://localhost/v1/stt/realtime"


def test_stt_update_models() -> None:
    stt = STT(config=BlazeConfig(api_url="http://localhost", api_token="tok"))
    stt.update_options(model="v1.0", stream_model="stt-stream-1.5")
    assert stt.model == "v1.0"
    assert stt.stream_model == "stt-stream-1.5"


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def test_normalize_text_collapses_whitespace() -> None:
    assert _normalize_text("Hello   world\n\n\nagain") == "Hello world\nagain"


def test_generate_silence_pcm16_length() -> None:
    silence = _generate_silence(24000, 150)
    assert len(silence) == int(24000 * 150 / 1000) * 2
    assert silence == b"\x00\x00" * int(24000 * 150 / 1000)


def test_apply_pcm16_fade_noop_without_flags() -> None:
    pcm = b"\x10\x00" * 8
    assert _apply_pcm16_fade(pcm, fade_samples=4) == pcm


def test_apply_pcm16_fade_in_reduces_leading_samples() -> None:
    pcm = b"\xff\x7f" * 8
    faded = _apply_pcm16_fade(pcm, fade_samples=4, fade_in=True)
    samples = struct.unpack("<8h", faded)

    assert samples[0] == 0
    assert abs(samples[3]) < abs(struct.unpack("<8h", pcm)[3])


def test_apply_pcm16_fade_out_reduces_trailing_samples() -> None:
    pcm = b"\xff\x7f" * 8
    faded = _apply_pcm16_fade(pcm, fade_samples=4, fade_out=True)
    original = struct.unpack("<8h", pcm)
    result = struct.unpack("<8h", faded)

    assert abs(result[-1]) < abs(original[-1])
    assert abs(result[-4]) == abs(original[-4])


@pytest.mark.asyncio
async def test_ws_stream_guard_idle_timeout() -> None:
    async def slow_recv() -> None:
        await asyncio.sleep(0.2)

    ws = AsyncMock()
    ws.recv = slow_recv

    guard = _WSStreamGuard(
        idle_timeout=0.05,
        session_deadline=time.monotonic() + 5.0,
        request_id="test",
    )

    with pytest.raises(APITimeoutError, match="idle timeout"):
        await guard.recv(ws)


@pytest.mark.asyncio
async def test_ws_stream_guard_session_deadline() -> None:
    ws = AsyncMock()
    guard = _WSStreamGuard(
        idle_timeout=5.0,
        session_deadline=time.monotonic() - 0.01,
        request_id="test",
    )

    with pytest.raises(APITimeoutError, match="max session duration"):
        await guard.recv(ws)


def test_tts_default_model_is_2_0_realtime() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"))
    assert tts.model == "2.0-realtime"


def test_tts_provider_model_and_ws_url() -> None:
    config = BlazeConfig(api_url="https://api.example.com", api_token="tok")
    tts = TTS(config=config, speaker_id="voice-1", model="v2_pro")

    assert tts.provider == "Blaze"
    assert tts.model == "v2_pro"
    assert tts._ws_url == "wss://api.example.com/v1/tts/realtime"
    assert tts._speaker_id == "voice-1"


def test_tts_invalid_audio_format_falls_back_to_pcm() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"), audio_format="ogg")
    assert tts._audio_format == "pcm"


def test_tts_uses_framework_sentence_tokenizer() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"))
    assert tts._sentence_tokenizer is not None
    assert not hasattr(tts, "_batch_min_chars")
    assert not hasattr(tts, "_batch_max_chars")


def test_tts_update_options() -> None:
    tts = TTS(config=BlazeConfig(api_url="http://localhost"))
    tts.update_options(
        speaker_id="new-speaker",
        model="v1_5_pro",
        audio_format="wav",
        audio_quality=64,
        language="en",
        auth_token="new-token",
        normalization_rules={"API": "A P I"},
    )

    assert tts._speaker_id == "new-speaker"
    assert tts._model == "v1_5_pro"
    assert tts._audio_format == "wav"
    assert tts._audio_quality == 64
    assert tts._language == "en"
    assert tts._auth_token == "new-token"
    assert tts._normalization_rules == {"API": "A P I"}


def test_tts_speech_start_params_include_optional_fields() -> None:
    from livekit.plugins.blaze.tts import _TTSSynthesizeStream

    config = BlazeConfig(api_url="http://localhost")
    tts = TTS(
        config=config,
        speaker_id="spk",
        model="v1_5_pro",
        audio_speed="1.2",
        audio_quality=48,
        language="vi",
    )
    stream = object.__new__(_TTSSynthesizeStream)
    stream._blaze_tts = tts
    params = stream._speech_start_params()

    assert params["event"] == "speech-start"
    assert params["speaker_id"] == "spk"
    assert params["model"] == "v1_5_pro"
    assert params["audio_speed"] == "1.2"
    assert params["audio_quality"] == 48
    assert params["language"] == "vi"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _make_llm(**kwargs: object) -> LLM:
    config = BlazeConfig(api_url="https://api.example.com", api_token="test-token")
    return LLM(bot_id="bot-123", config=config, **kwargs)


def _make_stream(chat_ctx: ChatContext, *, tools: list | None = None) -> LLMStream:
    llm = _make_llm(enable_tools=True)
    stream = object.__new__(LLMStream)
    stream._blaze_llm = llm
    stream._chat_ctx = chat_ctx
    stream._tools = tools or []
    stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
    return stream


def test_llm_provider_and_model() -> None:
    llm = _make_llm()
    assert llm.provider == "Blaze"
    assert llm.model == "bot-123"
    assert llm.bot_id == "bot-123"


def test_llm_update_options() -> None:
    llm = _make_llm()
    llm.update_options(
        deep_search=True,
        agentic_search=True,
        enable_tools=True,
        demographics={"gender": "female", "age": 30},
        auth_token="new-token",
    )

    assert llm._deep_search is True
    assert llm._agentic_search is True
    assert llm._enable_tools is True
    assert llm._demographics == {"gender": "female", "age": 30}
    assert llm._auth_token == "new-token"


def test_convert_messages_skips_system_and_developer() -> None:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="You are helpful.")
    chat_ctx.add_message(role="developer", content="Hidden developer prompt.")
    chat_ctx.add_message(role="user", content="Xin chào")

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "Xin chào"}]


def test_convert_messages_maps_assistant_and_strips_img_tags() -> None:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content="Hello <img>chart</img> world.")

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "assistant", "content": "Hello  world."}]


def test_convert_messages_includes_function_output_as_user() -> None:
    chat_ctx = ChatContext()
    chat_ctx.items.append(
        FunctionCallOutput(
            call_id="call_1",
            name="lookup",
            output="result payload",
            is_error=False,
        )
    )

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "result payload"}]


def test_convert_messages_wraps_tool_errors() -> None:
    chat_ctx = ChatContext()
    chat_ctx.items.append(
        FunctionCallOutput(
            call_id="call_1",
            name="lookup",
            output="timeout",
            is_error=True,
        )
    )

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "[Tool Error]: timeout"}]


@function_tool
async def sample_tool(query: str) -> str:
    """Look up information."""
    return query


def test_build_tools_param_serializes_function_tools() -> None:
    chat_ctx = ChatContext()
    stream = _make_stream(chat_ctx, tools=[sample_tool])

    tools_param = stream._build_tools_param()

    assert tools_param is not None
    assert len(tools_param) == 1
    assert tools_param[0]["type"] == "function"
    assert tools_param[0]["function"]["name"] == "sample_tool"


def test_build_tools_param_returns_none_without_tools() -> None:
    chat_ctx = ChatContext()
    assert _make_stream(chat_ctx)._build_tools_param() is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"content": "hello"}, "hello"),
        ({"text": "world"}, "world"),
        ({"delta": {"text": "delta-text"}}, "delta-text"),
        ({"delta": {"content": "delta-content"}}, "delta-content"),
        ({"choices": []}, None),
    ],
)
def test_extract_content(payload: dict, expected: str | None) -> None:
    stream = _make_stream(ChatContext())
    assert stream._extract_content(payload) == expected


def test_extract_tool_calls_openai_format() -> None:
    stream = _make_stream(ChatContext())
    data = {
        "tool_calls": [
            {
                "id": "call_abc",
                "function": {"name": "sample_tool", "arguments": '{"query":"hi"}'},
            }
        ]
    }

    calls = stream._extract_tool_calls(data)

    assert len(calls) == 1
    assert calls[0].name == "sample_tool"
    assert calls[0].call_id == "call_abc"
    assert json.loads(calls[0].arguments) == {"query": "hi"}


def test_extract_tool_calls_delta_format() -> None:
    stream = _make_stream(ChatContext())
    data = {
        "delta": {
            "tool_calls": [
                {
                    "id": "call_delta",
                    "function": {"name": "sample_tool", "arguments": {}},
                }
            ]
        }
    }

    calls = stream._extract_tool_calls(data)

    assert len(calls) == 1
    assert calls[0].name == "sample_tool"
    assert calls[0].arguments == "{}"
