"""Tests for Gnani Vachana STT plugin configuration and behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt

pytestmark = pytest.mark.plugin("gnani")


def test_stt_requires_api_key():
    """STT constructor raises when no API key is provided."""
    from livekit.plugins.gnani import STT

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            STT(api_key=None)


def test_stt_accepts_api_key_directly():
    """STT constructor accepts api_key argument."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.api_key == "test-key"


def test_stt_accepts_api_key_from_env():
    """STT constructor reads GNANI_API_KEY from environment."""
    from livekit.plugins.gnani import STT

    with patch.dict("os.environ", {"GNANI_API_KEY": "env-key"}):
        stt = STT()
        assert stt._opts.api_key == "env-key"


def test_stt_default_language():
    """STT defaults to en-IN."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.language == "en-IN"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", language="hi-IN")
    assert stt._opts.language == "hi-IN"


def test_stt_default_sample_rate():
    """STT defaults to 16000 Hz sample rate."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.sample_rate == 16000


def test_stt_accepts_8k_sample_rate():
    """STT accepts 8000 Hz sample rate."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", sample_rate=8000)
    assert stt._opts.sample_rate == 8000


def test_stt_rejects_invalid_sample_rate():
    """STT rejects sample rates outside STREAM_SUPPORTED_SAMPLE_RATES."""
    from livekit.plugins.gnani import STT

    with pytest.raises(ValueError, match="sample_rate"):
        STT(api_key="test-key", sample_rate=12345)


def test_stt_accepts_44k_and_48k_sample_rates():
    """STT accepts 44100 and 48000 Hz sample rates."""
    from livekit.plugins.gnani import STT

    for rate in (44100, 48000):
        stt = STT(api_key="test-key", sample_rate=rate)
        assert stt._opts.sample_rate == rate


def test_stt_capabilities():
    """STT reports streaming=True, interim_results=False."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is False


def test_stt_model_property():
    """The model property is the real Prisma model, not the URL version ("vachana-stt-v3").

    With no ``model`` the API uses prisma-v2.5, and the property says so.
    """
    from livekit.plugins.gnani import STT

    assert STT(api_key="test-key").model == "prisma-v2.5"
    assert STT(api_key="test-key", model="prisma-v3.0").model == "prisma-v3.0"


def test_stt_provider_property():
    """STT provider property returns 'Gnani'."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt.provider == "Gnani"


def test_stt_base_url_default():
    """STT defaults to Vachana API base URL."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.base_url == "https://api.vachana.ai"


def test_stt_custom_base_url():
    """STT accepts custom base URL."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", base_url="https://custom.api.com")
    assert stt._opts.base_url == "https://custom.api.com"


def test_stt_only_api_key_auth():
    """STT options only contain api_key for authentication (no organization_id or user_id)."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert not hasattr(stt._opts, "organization_id")
    assert not hasattr(stt._opts, "user_id")


def test_speech_stream_ws_url_https():
    """SpeechStream builds wss:// URL from https:// base."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.stt import STT, GnaniSTTOptions, SpeechStream

    stt = STT(api_key="test-key")
    opts = GnaniSTTOptions(
        api_key="test-key",
        language="en-IN",
        base_url="https://api.vachana.ai",
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    assert stream._build_ws_url() == "wss://api.vachana.ai/stt/v3/stream"


def test_speech_stream_ws_url_http():
    """SpeechStream builds ws:// URL from http:// base."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.stt import STT, GnaniSTTOptions, SpeechStream

    stt = STT(api_key="test-key")
    opts = GnaniSTTOptions(
        api_key="test-key",
        language="en-IN",
        base_url="http://localhost:8080",
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    assert stream._build_ws_url() == "ws://localhost:8080/stt/v3/stream"


def test_stt_default_format():
    """STT defaults to 'verbatim' format."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.format == "verbatim"


def test_stt_transcribe_format():
    """STT accepts 'transcribe' format for ITN."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", format="transcribe")
    assert stt._opts.format == "transcribe"


def test_stt_default_preferred_language():
    """STT defaults preferred_language to None."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.preferred_language is None


def test_stt_custom_preferred_language():
    """STT accepts custom preferred_language."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", preferred_language="hi-IN")
    assert stt._opts.preferred_language == "hi-IN"


def test_stt_default_itn_native_numerals():
    """STT defaults itn_native_numerals to False."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key")
    assert stt._opts.itn_native_numerals is False


def test_stt_itn_native_numerals_enabled():
    """STT accepts itn_native_numerals=True."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", format="transcribe", itn_native_numerals=True)
    assert stt._opts.itn_native_numerals is True


def test_stt_deprecated_kwargs_warning(caplog):
    """STT warns about deprecated kwargs without raising."""
    import logging

    from livekit.plugins.gnani import STT

    with caplog.at_level(logging.WARNING):
        stt = STT(api_key="test-key", organization_id="old", user_id="old")
    assert "deprecated" in caplog.text.lower()
    assert stt._opts.api_key == "test-key"


def test_stt_stream_returns_speech_stream():
    """STT.stream() returns a SpeechStream instance."""
    from livekit.plugins.gnani import STT
    from livekit.plugins.gnani.stt import SpeechStream

    stt = STT(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()
    assert isinstance(stream, SpeechStream)


def test_stt_stream_inherits_language():
    """STT.stream() uses the configured language."""
    from livekit.plugins.gnani import STT

    stt = STT(api_key="test-key", language="hi-IN")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()
    assert stream._opts.language == "hi-IN"


@pytest.mark.parametrize(
    "message_type, expected_retries",
    [
        ("connected", 2),
        ("pong", 2),
        ("processing", 0),
        ("transcript", 0),
        ("vad_start", 0),
        ("vad_end", 0),
    ],
)
async def test_provider_progress_resets_retries_without_transcript(message_type, expected_retries):
    import json
    from types import SimpleNamespace

    from livekit.plugins.gnani.stt import SpeechStream

    class Socket:
        def __aiter__(self):
            return self.messages()

        async def messages(self):
            yield json.dumps({"type": message_type, "text": ""})

    stream = SpeechStream.__new__(SpeechStream)
    stream._event_ch = SimpleNamespace(send_nowait=lambda event: None)
    stream._on_provider_progress = None
    stream._num_retries = 2
    await stream._recv_messages(Socket())
    assert stream._num_retries == expected_retries


# ---------------------------------------------------------------------------
# Prisma v3.0: model selection, languages, formats, context, request building
# ---------------------------------------------------------------------------


def _stt(**kwargs):
    from livekit.plugins.gnani import STT

    return STT(api_key="k", **kwargs)


def test_stt_model_is_unset_by_default_and_not_sent():
    """No model keeps today's behaviour: the server default (prisma-v2.5) applies."""
    from livekit.plugins.gnani.stt import _build_rest_fields, _build_ws_headers

    stt = _stt(language="hi-IN")
    assert stt._opts.model is None
    assert "model" not in _build_ws_headers(stt._opts)
    assert "model" not in dict(_build_rest_fields(stt._opts, "hi-IN"))


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
def test_stt_model_is_sent_on_both_transports(model):
    from livekit.plugins.gnani.stt import _build_rest_fields, _build_ws_headers

    stt = _stt(language="hi-IN", model=model)
    assert _build_ws_headers(stt._opts)["model"] == model
    assert dict(_build_rest_fields(stt._opts, "hi-IN"))["model"] == model


def test_stt_unknown_model_is_rejected():
    """The API silently falls back to the default model for unknown names."""
    with pytest.raises(ValueError, match="model must be one of"):
        _stt(model="prisma-v9")


def test_stt_language_forms_are_normalized():
    assert _stt(language="hi-IN")._opts.language == "hi-IN"
    assert _stt(language="indic-auto", model="prisma-v3.0")._opts.language == "indic-auto"
    assert _stt(language=["hi-IN", "en-IN"], model="prisma-v3.0")._opts.language == "hi-IN,en-IN"
    assert _stt(language="hi-IN, en-IN", model="prisma-v3.0")._opts.language == "hi-IN,en-IN"
    assert _stt(language=["hi-IN", "hi-IN"], model="prisma-v3.0")._opts.language == "hi-IN"


@pytest.mark.parametrize("language", ["indic-auto", "hi-IN,en-IN", ["hi-IN", "en-IN"]])
@pytest.mark.parametrize("model", [None, "prisma-v2.5"])
def test_stt_multi_language_requires_prisma_v3(language, model):
    """On the default model these fail at the API (400 on REST, an error mid-stream on
    WebSocket). Fail at construction with the fix in the message instead."""
    with pytest.raises(ValueError, match="prisma-v3.0"):
        _stt(language=language, model=model)


def test_stt_language_validation_limits():
    with pytest.raises(ValueError, match="cannot be combined"):
        _stt(language=["indic-auto", "hi-IN"], model="prisma-v3.0")
    with pytest.raises(ValueError, match="at most 4"):
        _stt(language=["hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN"], model="prisma-v3.0")
    with pytest.raises(ValueError, match="empty"):
        _stt(language=[])


def test_ws_headers_single_language():
    from livekit.plugins.gnani.stt import _build_ws_headers

    h = _build_ws_headers(_stt(language="ta-IN")._opts)
    assert h["language_code"] == "ta-IN"
    assert "x-language-candidates" not in h
    assert h["x-api-key-id"] == "k"
    assert h["x-sample-rate"] == "16000"


def test_ws_headers_language_list_uses_indic_auto_with_candidates():
    """The streaming endpoint rejects a comma-separated language_code (1008) and wants
    exactly one locale: a list is indic-auto restricted to the candidates."""
    from livekit.plugins.gnani.stt import _build_ws_headers

    h = _build_ws_headers(_stt(language=["hi-IN", "en-IN"], model="prisma-v3.0")._opts)
    assert h["language_code"] == "indic-auto"
    assert h["x-language-candidates"] == "hi-IN,en-IN"

    h = _build_ws_headers(_stt(language="indic-auto", model="prisma-v3.0")._opts)
    assert h["language_code"] == "indic-auto"
    assert "x-language-candidates" not in h


def test_rest_fields_send_the_language_list_as_given():
    from livekit.plugins.gnani.stt import _build_rest_fields

    stt = _stt(language=["hi-IN", "en-IN"], model="prisma-v3.0")
    assert dict(_build_rest_fields(stt._opts, stt._opts.language))["language_code"] == "hi-IN,en-IN"


def test_format_is_always_sent_explicitly():
    """Omitting it changes the output on prisma-v3.0 ("transcribe" for single hi-IN/en-IN)
    versus prisma-v2.5 ("verbatim"), so be explicit on both transports."""
    from livekit.plugins.gnani.stt import _build_rest_fields, _build_ws_headers

    opts = _stt(language="hi-IN")._opts
    assert _build_ws_headers(opts)["format"] == "verbatim"
    assert dict(_build_rest_fields(opts, "hi-IN"))["format"] == "verbatim"


@pytest.mark.parametrize("fmt", ["codemix", "transliterate"])
def test_v3_only_formats(fmt):
    assert _stt(model="prisma-v3.0", format=fmt)._opts.format == fmt
    for model in (None, "prisma-v2.5"):
        with pytest.raises(ValueError, match="prisma-v3.0"):
            _stt(model=model, format=fmt)


def test_invalid_format_is_rejected():
    with pytest.raises(ValueError, match="format must be one of"):
        _stt(format="loud")


def test_v3_context_is_rejected_on_the_default_model():
    """The API ignores it silently; surface it instead."""
    for kwargs in (
        {"keyterms": ["Gnani"]},
        {"topic": "banking"},
        {"previous_text": "hello"},
        {"speaker_names": ["Asha"]},
    ):
        with pytest.raises(ValueError, match="prisma-v3.0"):
            _stt(**kwargs)
        assert _stt(model="prisma-v3.0", **kwargs)


def test_ws_headers_carry_v3_context_ascii_safe():
    """Lists are JSON-escaped, as the API documents, because headers are bytes."""
    import json

    from livekit.plugins.gnani.stt import _build_ws_headers

    stt = _stt(
        language="hi-IN",
        model="prisma-v3.0",
        keyterms=["खाता", "HDFC Bank", "खाता", " "],
        topic="बैंक\nखाता  recharge",
        previous_text="x" * 700,
        speaker_names=["Asha", "Ravi"],
    )
    h = _build_ws_headers(stt._opts)

    assert h["keyterms"].isascii()
    assert json.loads(h["keyterms"]) == ["खाता", "HDFC Bank"]  # de-duplicated, blanks dropped
    assert h["topic"] == "बैंक खाता recharge"  # one line
    assert h["previous_text"] == "x" * 600  # the API keeps the last 600 characters
    assert json.loads(h["speaker_names"]) == ["Asha", "Ravi"]


def test_context_limits():
    with pytest.raises(ValueError, match="at most 200"):
        _stt(model="prisma-v3.0", keyterms=[f"w{i}" for i in range(201)])
    assert _stt(model="prisma-v3.0", keyterms=[f"w{i}" for i in range(200)])
    with pytest.raises(ValueError, match="1600"):
        _stt(model="prisma-v3.0", topic="x" * 1601)
    with pytest.raises(ValueError, match="empty"):
        _stt(model="prisma-v3.0", topic="  ")
    with pytest.raises(ValueError, match="1-4"):
        _stt(model="prisma-v3.0", speaker_names=list("abcde"))
    with pytest.raises(ValueError, match=r"<\|"):
        _stt(model="prisma-v3.0", keyterms=["bad<|term"])
    with pytest.raises(TypeError, match="list"):
        _stt(model="prisma-v3.0", keyterms="Gnani")


def test_session_tuning_is_sent_and_validated():
    from livekit.plugins.gnani.stt import _build_ws_headers

    h = _build_ws_headers(
        _stt(
            idle_timeout_sec=120,
            vad_threshold=0.5,
            min_silence_ms=300,
            min_speech_ms=100,
            min_segment_ms=150,
            max_speech_s=30,
        )._opts
    )
    assert h["x-idle-timeout-sec"] == "120"
    assert h["x-vad-threshold"] == "0.5"
    assert h["x-min-silence-ms"] == "300"
    assert h["x-min-speech-ms"] == "100"
    assert h["x-min-segment-ms"] == "150"
    assert h["x-max-speech-s"] == "30"

    unset = _build_ws_headers(_stt()._opts)
    assert not [
        k for k in unset if k.startswith("x-") and k not in ("x-api-key-id", "x-sample-rate")
    ]

    # the server clamps silently, so out-of-range values are rejected here
    for kwargs in (
        {"idle_timeout_sec": 4},
        {"idle_timeout_sec": 901},
        {"vad_threshold": 1.5},
        {"min_silence_ms": 50},
        {"min_silence_ms": 5000},
        {"max_speech_s": 1.0},
    ):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            _stt(**kwargs)


def test_capabilities_follow_the_model():
    streaming = _stt()
    assert streaming.capabilities.manual_flush is True
    assert streaming.capabilities.interim_results is False
    assert streaming.capabilities.keyterms is False  # prisma-v2.5 ignores keyterms
    assert _stt(model="prisma-v3.0").capabilities.keyterms is True

    rest = _stt(use_streaming=False)
    assert rest.capabilities.streaming is False
    assert rest.capabilities.manual_flush is False


def test_update_options_validates_atomically():
    stt = _stt(model="prisma-v3.0", language="hi-IN", format="transcribe", keyterms=["a"])
    stt.update_options(language=["hi-IN", "en-IN"], format="codemix", keyterms=["b", "c"])
    assert stt._opts.language == "hi-IN,en-IN"
    assert stt._opts.format == "codemix"
    assert stt._opts.keyterms == ("b", "c")

    with pytest.raises(ValueError, match="format must be one of"):
        stt.update_options(language="ta-IN", format="loud")
    assert stt._opts.language == "hi-IN,en-IN"  # unchanged by the rejected call

    v25 = _stt(language="hi-IN")
    with pytest.raises(ValueError, match="prisma-v3.0"):
        v25.update_options(language="indic-auto")
    with pytest.raises(ValueError, match="prisma-v3.0"):
        v25.update_options(format="codemix")
    assert v25._opts.language == "hi-IN"


def test_session_keyterms_are_merged_with_user_keyterms():
    """LiveKit's stt_context_options keyterms are applied alongside the user's."""
    stt = _stt(model="prisma-v3.0", keyterms=["user1", "shared"])
    stt._update_session_keyterms(["shared", "auto1"])
    assert stt._opts.keyterms == ("user1", "shared", "auto1")

    # a user update must not drop the framework-managed ones
    stt.update_options(keyterms=["user2"])
    assert stt._opts.keyterms == ("user2", "shared", "auto1")


def test_session_keyterms_are_ignored_when_the_model_does_not_support_them():
    stt = _stt(language="hi-IN")
    stt._update_session_keyterms(["auto1"])
    assert stt._opts.keyterms == ()


def test_session_keyterms_are_capped_at_the_api_limit():
    stt = _stt(model="prisma-v3.0", keyterms=["user"])
    stt._update_session_keyterms([f"auto{i}" for i in range(300)])
    assert len(stt._opts.keyterms) == 200
    assert stt._opts.keyterms[0] == "user"


# ================================================================================================
# STT behaviour against a local fake of the Vachana API
#
# Gnani STT streaming/REST behaviour against a local fake of the Vachana API.
#
# The fake reproduces behaviour recorded from the live API (2026-10-30):
#
# * ``connected`` first, then ``speech_start`` when audio starts and one final
#   ``transcript`` per utterance. There is no ``speech_end`` message, and no interim results.
# * A ``commit`` control frame closes the open utterance immediately.
# * Idle expiry is announced as an ``error`` carrying ``timeout_reason``, then ``session_end``,
#   then a normal 1001 close.
# * Rejected handshakes send an ``error`` message and close with 1008.
# * Utterance-level failures ("Transcription failed or audio unclear") are ``error`` messages
#   that leave the connection open.
# ================================================================================================


OPTS = APIConnectOptions(max_retry=0, timeout=5, retry_interval=0.05)
# 100 ms of a loud tone (speech stand-in) and 100 ms of digital silence
FRAME = rtc.AudioFrame(
    (b"\xe8\x03\x18\xfc") * 800,  # +1000, -1000, ...
    16000,
    1,
    1600,
)
SILENCE = rtc.AudioFrame(bytes(3200), 16000, 1, 1600)


class FakeVachanaSTT:
    def __init__(self) -> None:
        self.sessions: list[SimpleNamespace] = []
        self.text = "नमस्ते, आप कैसे हैं?"
        self.detected: str | None = "hi-IN"
        # per-session behaviour by index: "idle" | "reject" | "quick_close"
        self.behaviours: dict[int, str] = {}
        self.utterance_errors_left = 0
        # seconds the server takes before it answers the audio it received (speech_start
        # and the transcript both come late), and a switch for a server that never answers
        self.reply_delay = 0.0
        self.silent = False
        # seconds before the transcript answering a `commit`
        self.commit_delay = 0.0
        # finalise an utterance on its own when silence follows speech, like the real VAD
        self.auto_finalize = False
        self.rest_calls: list[dict[str, Any]] = []
        self.rest_status = 200

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        index = len(self.sessions)
        s = SimpleNamespace(headers=dict(request.headers), audio=0, controls=[], speaking=False)
        self.sessions.append(s)
        mode = self.behaviours.get(index)

        if mode == "reject":
            await ws.send_json({"type": "error", "message": "Invalid x-format. Supported values"})
            await ws.close(code=1008, message=b"policy violation")
            return ws

        await ws.send_json({"type": "connected", "config": {}})
        if mode == "quick_close":
            await ws.send_json({"type": "session_end", "reason": "idle_timeout"})
            await ws.close(code=1001, message=b"idle_timeout")
            return ws

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                s.audio += len(msg.data)
                if mode == "idle":
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Session closed: no speech segment received for 5 seconds.",
                            "timeout_reason": "idle",
                        }
                    )
                    await ws.send_json({"type": "session_end", "reason": "idle_timeout"})
                    await ws.close(code=1001, message=b"idle_timeout")
                    break
                if self.auto_finalize and s.speaking and not any(msg.data):
                    s.speaking = False
                    await ws.send_json(self._transcript(index))
                    continue
                if not s.speaking and not self.silent and any(msg.data):
                    s.speaking = True
                    if self.reply_delay:
                        await asyncio.sleep(self.reply_delay)
                    await ws.send_json({"type": "speech_start", "utterance_id": "u1"})
            elif msg.type == aiohttp.WSMsgType.TEXT:
                body = json.loads(msg.data)
                s.controls.append(body)
                if body["type"] == "commit" and s.speaking:
                    s.speaking = False
                    if self.utterance_errors_left > 0:
                        self.utterance_errors_left -= 1
                        await ws.send_json(
                            {"type": "error", "message": "Transcription failed or audio unclear"}
                        )
                        continue
                    if self.commit_delay:
                        await asyncio.sleep(self.commit_delay)
                    await ws.send_json(self._transcript(index))
                elif body["type"] == "update_settings":
                    await ws.send_json({"type": "settings_updated", "config": {}})
        return ws

    def _transcript(self, index: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "transcript",
            "text": self.text,
            "is_final": True,
            "utterance_id": "u1",
            "segment_id": f"seg-{index}",
        }
        if self.detected:
            payload["detected_language_code"] = self.detected
        return payload

    async def rest(self, request: web.Request) -> web.Response:
        form = await request.post()
        self.rest_calls.append(
            {k: v for k, v in form.items() if k != "audio_file"}
            | {"_headers": dict(request.headers)}
        )
        if self.rest_status != 200:
            return web.json_response(
                {"success": False, "error": {"type": "INVALID_LANGUAGE_CODE", "message": "nope"}},
                status=self.rest_status,
            )
        body: dict[str, Any] = {
            "success": True,
            "request_id": "req-1",
            "transcript": self.text,
            "model": form.get("model", "prisma-v2.5"),
        }
        if self.detected:
            body["detected_language_code"] = self.detected
        return web.json_response(body)


@pytest.fixture
async def server():
    fake = FakeVachanaSTT()
    app = web.Application()
    app.router.add_get("/stt/v3/stream", fake.ws)
    app.router.add_post("/stt/v3", fake.rest)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    fake.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"  # type: ignore[attr-defined]
    fake.session = aiohttp.ClientSession()  # type: ignore[attr-defined]
    try:
        yield fake
    finally:
        await fake.session.close()  # type: ignore[attr-defined]
        await runner.cleanup()


def _make_stt(server: Any, **kwargs: Any):
    from livekit.plugins.gnani import STT

    stt_ = STT(api_key="test-key", base_url=server.base_url, **kwargs)
    stt_._session = server.session
    return stt_


async def _utterance(stream, *, flush: bool = True) -> None:
    stream.push_frame(FRAME)
    if flush:
        stream.flush()


async def _collect(stream) -> list[stt.SpeechEvent]:
    events = []
    async for ev in stream:
        events.append(ev)
    return events


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [e.type for e in events]


SOS = stt.SpeechEventType.START_OF_SPEECH
EOS = stt.SpeechEventType.END_OF_SPEECH
FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


async def test_stream_sends_the_expected_handshake(server):
    stt_ = _make_stt(
        server,
        model="prisma-v3.0",
        language=["hi-IN", "en-IN"],
        format="codemix",
        keyterms=["Gnani"],
        topic="recharge",
        speaker_names=["Asha"],
        idle_timeout_sec=120,
        min_silence_ms=300,
    )
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    await _collect(stream)

    h = server.sessions[0].headers
    assert h["x-api-key-id"] == "test-key"
    assert h["model"] == "prisma-v3.0"
    assert h["language_code"] == "indic-auto"
    assert h["x-language-candidates"] == "hi-IN,en-IN"
    assert h["format"] == "codemix"
    assert json.loads(h["keyterms"]) == ["Gnani"]
    assert h["topic"] == "recharge"
    assert json.loads(h["speaker_names"]) == ["Asha"]
    assert h["x-idle-timeout-sec"] == "120"
    assert h["x-min-silence-ms"] == "300"
    assert h["x-sample-rate"] == "16000"


async def test_utterance_produces_start_final_end_in_order(server):
    """The server sends no end-of-speech message. STT-based turn detection needs an
    END_OF_SPEECH, so it follows the final transcript."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()

    events = await _collect(stream)

    assert _types(events) == [SOS, FINAL, EOS]
    final = events[1]
    assert final.alternatives[0].text == server.text
    assert final.alternatives[0].language == "hi-IN"
    assert final.request_id == "seg-0"


async def test_detected_language_is_reported_not_the_configured_one(server):
    server.detected = "en-IN"
    stt_ = _make_stt(server, model="prisma-v3.0", language="indic-auto")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()

    final = next(e for e in await _collect(stream) if e.type == FINAL)

    assert final.alternatives[0].language == "en-IN"


async def test_language_falls_back_to_the_configured_one_when_not_detected(server):
    server.detected = None  # prisma-v2.5 and format="transliterate" do not return it
    stt_ = _make_stt(server, language="ta-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    final = next(e for e in await _collect(stream) if e.type == FINAL)
    assert final.alternatives[0].language == "ta-IN"

    stt_ = _make_stt(server, model="prisma-v3.0", language="indic-auto", format="transliterate")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    final = next(e for e in await _collect(stream) if e.type == FINAL)
    assert final.alternatives[0].language == "multi"  # unknown, not the literal "indic-auto"


async def test_flush_sends_a_commit_frame(server):
    """manual_flush: LiveKit's flush() closes the utterance without waiting for the
    server's trailing-silence window."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=True)

    await _wait_for(lambda: server.sessions and {"type": "commit"} in server.sessions[0].controls)
    async for ev in stream:
        if ev.type == FINAL:
            break
    stream.end_input()
    await _collect(stream)

    assert {"type": "commit"} in server.sessions[0].controls


async def test_end_input_commits_so_the_last_utterance_is_not_lost(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    stream.end_input()  # no explicit flush from the caller

    events = await _collect(stream)

    assert FINAL in _types(events)


async def test_end_input_waits_for_a_late_transcript_of_the_final_utterance(server):
    """The server may take far longer than a fixed grace period to answer a short clip:
    speech_start and the transcript both arrive after end_input(). They must not be lost."""
    server.reply_delay = 1.2
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    stream.end_input()

    events = await _collect(stream)

    assert _types(events) == [SOS, FINAL, EOS]
    assert events[1].alternatives[0].text == server.text


async def test_end_input_returns_once_the_final_utterance_failed(server):
    """An utterance error answers the commit too, so there is nothing left to wait for."""
    server.reply_delay = 0.3
    server.utterance_errors_left = 1
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.push_frame(FRAME)
    stream.end_input()

    started = asyncio.get_running_loop().time()
    events = await _collect(stream)

    assert FINAL not in _types(events)
    assert asyncio.get_running_loop().time() - started < 5  # well below the 30 s timeout


async def test_end_input_gives_up_at_the_connection_timeout_when_never_answered(server):
    """A final utterance without speech is never answered: wait for the configured
    timeout (not forever) and finish quietly."""
    server.silent = True
    errors: list[stt.STTError] = []
    stt_ = _make_stt(server, language="hi-IN")
    stt_.on("error", errors.append)
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1))
    stream.push_frame(FRAME)
    stream.end_input()

    started = asyncio.get_running_loop().time()
    events = await _collect(stream)
    elapsed = asyncio.get_running_loop().time() - started

    assert events == []
    assert not errors
    assert 0.9 < elapsed < 4


async def test_end_input_does_not_wait_when_nothing_was_sent(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.end_input()

    started = asyncio.get_running_loop().time()
    assert await _collect(stream) == []
    assert asyncio.get_running_loop().time() - started < 5


async def test_end_input_after_silence_only_does_not_wait(server):
    """Audio without speech is never answered, so there is nothing to wait for."""
    server.silent = True
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.push_frame(SILENCE)
    stream.end_input()

    started = asyncio.get_running_loop().time()
    assert await _collect(stream) == []
    assert asyncio.get_running_loop().time() - started < 5


async def test_end_input_waits_for_every_queued_commit(server):
    """Two utterances committed before their replies arrive: the first reply must not end
    the wait for the second."""
    server.commit_delay = 0.3
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    stream.flush()  # utterance A
    stream.push_frame(FRAME)  # utterance B, committed by end_input
    stream.end_input()

    events = await _collect(stream)

    assert _types(events).count(FINAL) == 2


async def test_end_input_does_not_wait_for_audio_the_server_already_finalized(server):
    """Speech followed by silence is finalised by the server's own VAD. Ending the input
    afterwards must not wait for a reply that will never come."""
    server.auto_finalize = True
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.push_frame(FRAME)
    stream.push_frame(SILENCE)

    async for ev in stream:  # the naturally finalised transcript
        if ev.type == FINAL:
            break
    stream.push_frame(SILENCE)  # trailing silence while the caller winds down
    stream.end_input()

    started = asyncio.get_running_loop().time()
    await _collect(stream)
    assert asyncio.get_running_loop().time() - started < 5  # not the 30 s timeout


async def test_partial_frames_are_chunked_to_1024_bytes(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    for _ in range(5):
        stream.push_frame(FRAME)  # 5 x 3200 B = 16000 B -> 15 full chunks + a remainder
    stream.end_input()
    await _collect(stream)

    assert server.sessions[0].audio == 16000


async def test_idle_timeout_reconnects_transparently(server):
    """The server ends idle sessions (60 s by default) with an `error`, `session_end` and a
    1001 close. That is routine: it must not surface as an error, and the stream carries on
    with a new session (an error here counts toward LiveKit's unrecoverable-error limit)."""
    server.behaviours[0] = "idle"
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    errors: list[Any] = []
    stt_.on("error", errors.append)

    stream.push_frame(FRAME)  # session 0 answers this with the idle timeout
    await _wait_for(lambda: len(server.sessions) == 2)
    await _utterance(stream)
    stream.end_input()
    events = await _collect(stream)

    assert FINAL in _types(events)
    assert errors == []
    assert server.sessions[1].headers["model"] == "prisma-v3.0"
    assert server.sessions[1].audio > 0


async def test_reconnect_uses_options_updated_in_the_meantime(server):
    server.behaviours[0] = "idle"
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", format="verbatim")
    stream = stt_.stream(conn_options=OPTS)
    stt_.update_options(format="transcribe")

    stream.push_frame(FRAME)
    await _wait_for(lambda: len(server.sessions) == 2)
    stream.end_input()
    await _collect(stream)

    assert server.sessions[1].headers["format"] == "transcribe"


async def test_session_that_keeps_closing_immediately_gives_up(server):
    for i in range(10):
        server.behaviours[i] = "quick_close"
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    with pytest.raises(APIConnectionError, match="keeps closing"):
        await _collect(stream)

    assert len(server.sessions) == 3


async def test_utterance_error_does_not_kill_the_stream(server):
    """ "Transcription failed or audio unclear" leaves the connection open; a noisy moment
    must not end the stream. The utterance it belonged to is closed."""
    server.utterance_errors_left = 1
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    await _utterance(stream)  # fails
    first = []
    async for ev in stream:
        first.append(ev)
        if ev.type == EOS:
            break
    assert _types(first) == [SOS, EOS]

    await _utterance(stream)  # next utterance on the same connection works
    stream.end_input()
    rest = await _collect(stream)

    assert FINAL in _types(rest)
    assert len(server.sessions) == 1


async def test_rejected_handshake_is_a_clear_non_retryable_error(server):
    server.behaviours[0] = "reject"
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    with pytest.raises(APIStatusError) as exc:
        await _collect(stream)

    assert exc.value.status_code == 400
    assert "Invalid x-format" in str(exc.value)
    assert exc.value.retryable is False
    assert len(server.sessions) == 1


async def test_update_options_applies_live_via_update_settings(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", format="verbatim")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language="en-IN", format="transcribe", keyterms=["Gnani", "Vachana"])

    await _wait_for(
        lambda: any(c["type"] == "update_settings" for c in server.sessions[0].controls)
    )
    stream.end_input()
    await _collect(stream)

    update = next(c for c in server.sessions[0].controls if c["type"] == "update_settings")
    assert update == {
        "type": "update_settings",
        "language_code": "en-IN",
        "format": "transcribe",
        "bias": {"list": ["Gnani", "Vachana"]},
    }


def _language_updates(session: Any) -> list[dict[str, Any]]:
    return [c for c in session.controls if c["type"] == "update_settings" and "language_code" in c]


@pytest.mark.parametrize(
    ("start", "update", "headers"),
    [
        # single -> list: the new session needs the candidate restriction
        (
            "hi-IN",
            ["hi-IN", "en-IN"],
            {"language_code": "indic-auto", "x-language-candidates": "hi-IN,en-IN"},
        ),
        # list -> a different list: the old restriction would stay in force otherwise
        (
            ["hi-IN", "en-IN"],
            ["ta-IN", "te-IN"],
            {"language_code": "indic-auto", "x-language-candidates": "ta-IN,te-IN"},
        ),
        # list -> single: the candidate header must go away
        (["hi-IN", "en-IN"], "ta-IN", {"language_code": "ta-IN"}),
    ],
)
async def test_language_list_changes_reconnect_with_the_new_handshake(
    server, start, update, headers
):
    """update_settings can only change `language_code`; the candidate list lives in the
    handshake. Sending `indic-auto` alone would enable unrestricted detection (or keep the
    old restriction), so a change involving a list opens a new session instead."""
    stt_ = _make_stt(server, model="prisma-v3.0", language=start)
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language=update)

    await _wait_for(lambda: len(server.sessions) >= 2)
    stream.end_input()
    await _collect(stream)

    assert _language_updates(server.sessions[0]) == []  # never a partial live update
    new = server.sessions[1].headers
    for key, value in headers.items():
        assert new[key] == value
    if "x-language-candidates" not in headers:
        assert "x-language-candidates" not in new


async def test_single_language_change_still_updates_live(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language="en-IN")

    await _wait_for(lambda: _language_updates(server.sessions[0]))
    stream.end_input()
    await _collect(stream)

    assert _language_updates(server.sessions[0]) == [
        {"type": "update_settings", "language_code": "en-IN"}
    ]
    assert len(server.sessions) == 1  # no reconnect


async def test_language_list_change_keeps_the_utterance_in_progress(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)  # the server answers with speech_start
    first = await stream.__anext__()
    assert first.type == SOS

    stt_.update_options(language=["hi-IN", "en-IN"], format="transcribe")

    await _wait_for(lambda: len(server.sessions) >= 2)
    stream.end_input()
    events = [first, *await _collect(stream)]

    assert {"type": "commit"} in server.sessions[0].controls  # committed before closing
    assert FINAL in _types(events)
    assert server.sessions[1].headers["format"] == "transcribe"  # other options carried over


async def test_repeated_language_changes_are_not_mistaken_for_a_broken_session(server):
    """Reconnecting on request must not trip the guard against sessions that keep closing
    immediately."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    for n, language in enumerate(
        [["hi-IN", "en-IN"], ["ta-IN", "te-IN"], "hi-IN", ["en-IN", "hi-IN"]]
    ):
        await _wait_for(lambda n=n: len(server.sessions) == n + 1 and server.sessions[n].audio > 0)
        stt_.update_options(language=language)
        await _wait_for(lambda n=n: len(server.sessions) == n + 2)
        stream.push_frame(FRAME)

    stream.end_input()
    await _collect(stream)  # would raise APIConnectionError if throttled as quick closes
    assert len(server.sessions) == 5


async def test_session_keyterms_reach_a_running_stream(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", keyterms=["user"])
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_._update_session_keyterms(["auto"])

    await _wait_for(
        lambda: any(c["type"] == "update_settings" for c in server.sessions[0].controls)
    )
    stream.end_input()
    await _collect(stream)

    update = next(c for c in server.sessions[0].controls if c["type"] == "update_settings")
    assert update["bias"] == {"list": ["user", "auto"]}


async def test_stream_language_override_is_validated(server):
    stt_ = _make_stt(server, language="hi-IN")  # default model: one language only
    with pytest.raises(ValueError, match="prisma-v3.0"):
        stt_.stream(language=["hi-IN", "en-IN"], conn_options=OPTS)

    v3 = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = v3.stream(language=["hi-IN", "en-IN"], conn_options=OPTS)
    stream.end_input()
    await _collect(stream)
    assert server.sessions[-1].headers["x-language-candidates"] == "hi-IN,en-IN"


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


async def test_recognize_sends_model_aware_form_and_reports_detected_language(server):
    server.detected = "en-IN"
    stt_ = _make_stt(
        server,
        model="prisma-v3.0",
        language=["hi-IN", "en-IN"],
        format="codemix",
        keyterms=["Gnani"],
        topic="recharge",
    )

    event = await stt_.recognize(FRAME, conn_options=OPTS)

    call = server.rest_calls[0]
    assert call["language_code"] == "hi-IN,en-IN"
    assert call["model"] == "prisma-v3.0"
    assert call["format"] == "codemix"
    assert json.loads(call["keyterms"]) == ["Gnani"]
    assert call["topic"] == "recharge"
    assert call["_headers"]["X-API-Key-ID"] == "test-key"
    assert event.alternatives[0].text == server.text
    assert event.alternatives[0].language == "en-IN"
    assert event.request_id == "req-1"


async def test_recognize_default_model_sends_no_model(server):
    server.detected = None
    stt_ = _make_stt(server, language="hi-IN")

    event = await stt_.recognize(FRAME, conn_options=OPTS)

    assert "model" not in server.rest_calls[0]
    assert server.rest_calls[0]["language_code"] == "hi-IN"
    assert event.alternatives[0].language == "hi-IN"


async def test_recognize_language_override_is_validated(server):
    stt_ = _make_stt(server, language="hi-IN")
    with pytest.raises(ValueError, match="prisma-v3.0"):
        await stt_.recognize(FRAME, language="indic-auto", conn_options=OPTS)
    assert server.rest_calls == []


async def test_recognize_api_error_is_surfaced(server):
    server.rest_status = 400
    stt_ = _make_stt(server, language="hi-IN")

    with pytest.raises(APIStatusError) as exc:
        await stt_.recognize(FRAME, conn_options=OPTS)

    assert exc.value.status_code == 400
    assert "INVALID_LANGUAGE_CODE" in str(exc.value)
