"""Tests for Gnani Vachana STT plugin configuration and behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
