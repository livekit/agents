"""Tests for AssemblyAI STT plugin configuration options."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.stt import SpeechEventType
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.plugin("assemblyai")


# ---------------------------------------------------------------------------
# SpeechStarted → speech_start_time conversion
#
# The plugin anchors the stream's wall-clock via the base-class `start_time`
# property (which it overrides in send_task on the first ws.send_bytes). The
# server emits SpeechStarted with a stream-relative `timestamp` in ms, which
# the plugin converts to wall-clock by `self.start_time + timestamp_ms/1000`
# and surfaces via `SpeechEvent.speech_start_time`.
# ---------------------------------------------------------------------------


def _make_stream_for_unit_test(stt=None):
    """Construct a SpeechStream without triggering the _main_task WebSocket
    loop. Patches asyncio.create_task during __init__ so the stream doesn't
    try to open a real connection; also closes the coroutines that would
    otherwise be scheduled, to avoid un-awaited coroutine warnings."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.assemblyai import STT
    from livekit.plugins.assemblyai.stt import SpeechStream

    if stt is None:
        stt = STT(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        # Close the coroutine so we don't get RuntimeWarning about it never
        # being awaited. Return a benign mock so callers that chain
        # add_done_callback / cancel don't break.
        coro.close()
        task = MagicMock()
        return task

    with patch("asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=stt._opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            http_session=MagicMock(),
            base_url="wss://streaming.assemblyai.com",
        )
    return stream


async def test_speech_started_uses_start_time_anchor():
    """The SpeechStarted handler converts server timestamp_ms to wall-clock
    using self.start_time, and emits a SpeechEvent with the correct
    speech_start_time."""
    stream = _make_stream_for_unit_test()

    # Override the stream anchor to a known value — simulates what send_task
    # would do on the first ws.send_bytes.
    anchor = 1_700_000_000.0
    stream.start_time = anchor

    # Simulate the server sending a SpeechStarted message 500ms into the stream.
    stream._process_stream_event({"type": "SpeechStarted", "timestamp": 500})

    ev = stream._event_ch.recv_nowait()
    assert ev.type == SpeechEventType.START_OF_SPEECH
    assert ev.speech_start_time == anchor + 0.5


async def test_speech_started_timestamp_zero_still_anchored():
    """A timestamp of 0 is a valid onset at stream start (not treated as
    'missing field'). Tests the earlier fix for the `is not None` check."""
    stream = _make_stream_for_unit_test()

    anchor = 1_700_000_000.0
    stream.start_time = anchor
    stream._process_stream_event({"type": "SpeechStarted", "timestamp": 0})

    ev = stream._event_ch.recv_nowait()
    assert ev.type == SpeechEventType.START_OF_SPEECH
    assert ev.speech_start_time == anchor


async def test_speech_started_without_timestamp_leaves_field_none():
    """If the server omits `timestamp` entirely, the plugin should emit
    START_OF_SPEECH with speech_start_time=None so the framework falls back
    to message-arrival time (pre-PR behavior)."""
    stream = _make_stream_for_unit_test()

    # Server sends SpeechStarted with no timestamp field.
    stream._process_stream_event({"type": "SpeechStarted"})

    ev = stream._event_ch.recv_nowait()
    assert ev.type == SpeechEventType.START_OF_SPEECH
    assert ev.speech_start_time is None


async def test_continuous_partials_requires_u3_rt_pro():
    """Test continuous_partials raises ValueError when used with a non-u3-rt-pro model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="continuous_partials"):
        STT(api_key="test-key", model="universal-streaming-english", continuous_partials=True)


async def test_interruption_delay_requires_u3_rt_pro():
    """Test interruption_delay raises ValueError when used with a non-u3-rt-pro model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="interruption_delay"):
        STT(api_key="test-key", model="universal-streaming-english", interruption_delay=200)


# ---------------------------------------------------------------------------
# agent_context
#
# agent_context carries "what the agent said" so the model can use it to bias
# transcription of the user's reply. It is threaded through STTOptions, the
# constructor, and both update_options paths, and is sent over the live
# websocket as an UpdateConfiguration message.
# ---------------------------------------------------------------------------


async def test_agent_context_stream_sends_update_configuration():
    """SpeechStream.update_options enqueues an UpdateConfiguration message
    containing agent_context, even when it's the only field updated (the
    len(config_msg) > 1 guard)."""
    stream = _make_stream_for_unit_test()

    stream.update_options(agent_context="The agent confirmed the order.")

    assert stream._opts.agent_context == "The agent confirmed the order."
    msg = stream._config_update_queue.get_nowait()
    assert msg["type"] == "UpdateConfiguration"
    assert msg["agent_context"] == "The agent confirmed the order."


async def test_agent_context_propagates_from_stt_to_active_stream():
    """STT.update_options(agent_context=...) propagates to an already-active
    stream and sends it over that stream's websocket queue."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()

    stt.update_options(agent_context="The agent greeted the caller.")

    assert stt._opts.agent_context == "The agent greeted the caller."
    assert stream._opts.agent_context == "The agent greeted the caller."
    msg = stream._config_update_queue.get_nowait()
    assert msg["type"] == "UpdateConfiguration"
    assert msg["agent_context"] == "The agent greeted the caller."


# ---------------------------------------------------------------------------
# u3-rt-pro-beta-1 model + u3-pro param family
#
# u3-rt-pro-beta-1 shares all u3-rt-pro behavior, so the u3-pro-gated params
# (prompt, agent_context, previous_context_n_turns, continuous_partials,
# interruption_delay) are accepted with it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# agent_context is u3-rt-pro-only
# ---------------------------------------------------------------------------


async def test_agent_context_requires_u3_rt_pro():
    """agent_context in the constructor raises for non-u3-rt-pro models."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="agent_context"):
        STT(api_key="test-key", model="universal-streaming-english", agent_context="hello")


# ---------------------------------------------------------------------------
# previous_context_n_turns (u3-rt-pro only, connect-only)
# ---------------------------------------------------------------------------


async def test_previous_context_n_turns_requires_u3_rt_pro():
    """previous_context_n_turns raises for non-u3-rt-pro models."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="previous_context_n_turns"):
        STT(api_key="test-key", model="universal-streaming-english", previous_context_n_turns=5)


async def test_previous_context_n_turns_zero_is_forwarded():
    """0 is a meaningful value (disable carryover), distinct from unset, and must
    be sent in the connect config rather than dropped."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="u3-rt-pro", previous_context_n_turns=0)
    assert stt._opts.previous_context_n_turns == 0

    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["previous_context_n_turns"] == ["0"]


# ---------------------------------------------------------------------------
# universal-3-5-pro: u3-rt-pro parameter family
#
# universal-3-5-pro belongs to the u3-rt-pro parameter family, so it accepts
# the u3-pro-gated params (prompt, agent_context, previous_context_n_turns,
# continuous_partials, interruption_delay) and inherits the family's
# connect-time defaults.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# universal-3-6-pro: default model + u3-rt-pro parameter family
#
# universal-3-6-pro is the plugin's default model and the next U3 Pro release
# after universal-3-5-pro: server-side it has the same parameter support as
# universal-3-5-pro and differs only in which ASR deployment serves the
# session. So it accepts the u3-pro-gated params and inherits the family's
# connect-time defaults.
# ---------------------------------------------------------------------------


async def test_universal_3_6_pro_accepts_u3_pro_params():
    """universal-3-6-pro shares the u3-rt-pro parameter family."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="universal-3-6-pro",
        prompt="medical dictation",
        agent_context="The agent asked for the patient's name.",
        previous_context_n_turns=10,
        interruption_delay=300,
        voice_focus="near-field",
        mode="max_accuracy",
        language_codes=["en", "es"],
    )
    assert stt._opts.speech_model == "universal-3-6-pro"
    assert stt._opts.prompt == "medical dictation"
    assert stt._opts.agent_context == "The agent asked for the patient's name."
    assert stt._opts.previous_context_n_turns == 10
    assert stt._opts.interruption_delay == 300
    assert stt._opts.voice_focus == "near-field"
    assert stt._opts.mode == "max_accuracy"
    assert stt._opts.language_codes == ["en", "es"]


async def test_universal_3_6_pro_connect_config_uses_u3_pro_defaults():
    """The connect query names universal-3-6-pro and applies the U3 Pro family's
    connect-time defaults (100ms min/max turn silence, language detection on)."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-6-pro")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["speech_model"] == ["universal-3-6-pro"]
    assert query["min_turn_silence"] == ["100"]
    assert query["max_turn_silence"] == ["100"]
    assert query["language_detection"] == ["true"]


# ---------------------------------------------------------------------------
# voice_focus / voice_focus_threshold
#
# Voice Focus isolates the primary voice and suppresses background noise.
# voice_focus is a string enum ("near-field" / "far-field"); voice_focus_threshold
# is a float in [0, 1]. Both are u3-rt-pro-family-only and connect-time only
# (not exposed via update_options).
# ---------------------------------------------------------------------------


async def test_voice_focus_requires_u3_pro_family():
    """voice_focus raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="voice_focus"):
        STT(api_key="test-key", model="universal-streaming-english", voice_focus="near-field")


async def test_voice_focus_threshold_requires_u3_pro_family():
    """voice_focus_threshold raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="voice_focus_threshold"):
        STT(
            api_key="test-key",
            model="universal-streaming-english",
            voice_focus_threshold=0.5,
        )


async def test_voice_focus_in_connect_config():
    """voice_focus and voice_focus_threshold are sent in the connect config query."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(
        api_key="test-key",
        model="universal-3-5-pro",
        voice_focus="far-field",
        voice_focus_threshold=0.8,
    )
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["voice_focus"] == ["far-field"]
    assert query["voice_focus_threshold"] == ["0.8"]


async def test_voice_focus_absent_from_connect_config_when_unset():
    """voice_focus keys are omitted from the connect config when not set."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "voice_focus" not in query
    assert "voice_focus_threshold" not in query


async def test_voice_focus_threshold_zero_is_forwarded():
    """0.0 is a meaningful threshold (minimum suppression), distinct from unset, and must
    be sent in the connect config rather than dropped by a truthiness filter."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", voice_focus="near-field", voice_focus_threshold=0.0)
    assert stt._opts.voice_focus_threshold == 0.0

    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["voice_focus_threshold"] == ["0.0"]


async def test_voice_focus_allowed_for_all_u3_pro_family_models():
    """voice_focus is accepted for every u3-rt-pro-family model, not just the default."""
    from livekit.plugins.assemblyai import STT

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro", "universal-3-6-pro"):
        stt = STT(api_key="test-key", model=model, voice_focus="far-field")
        assert stt._opts.voice_focus == "far-field"


# ---------------------------------------------------------------------------
# mode (latency/accuracy preset)
#
# `mode` selects the u3-pro accuracy/latency tradeoff: "min_latency",
# "balanced" (server default), or "max_accuracy". It is forwarded to the
# u3-pro ASR server, which applies its own per-mode tuning, so it is
# u3-rt-pro-family-only and connect-time only (not exposed via update_options,
# matching the official AssemblyAI SDK, where `mode` lives on the connect-time
# parameters and not on UpdateConfiguration).
# ---------------------------------------------------------------------------


async def test_mode_requires_u3_pro_family():
    """mode raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="mode"):
        STT(api_key="test-key", model="universal-streaming-english", mode="max_accuracy")


async def test_mode_in_connect_config():
    """mode is sent in the connect config query."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro", mode="max_accuracy")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["mode"] == ["max_accuracy"]


async def test_mode_absent_from_connect_config_when_unset():
    """mode key is omitted from the connect config when not set."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "mode" not in query


async def test_mode_omits_silence_defaults_when_unset():
    """When `mode` is set but min/max turn silence aren't explicitly provided,
    the plugin must NOT inject its default 100ms windows. Sending explicit
    silence values would override the mode preset's own silence tuning
    server-side, defeating the purpose of selecting a mode."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    for mode in ("min_latency", "balanced", "max_accuracy"):
        stt = STT(api_key="test-key", model="universal-3-5-pro", mode=mode)
        stream = _make_stream_for_unit_test(stt)
        stream._session.ws_connect = _fake_ws_connect
        await stream._connect_ws()

        query = parse_qs(urlparse(captured["url"]).query)
        assert query["mode"] == [mode]
        assert "min_turn_silence" not in query
        assert "max_turn_silence" not in query


async def test_mode_with_explicit_silence_still_sent():
    """Explicit min/max turn silence override the mode preset and are sent even
    when `mode` is set."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(
        api_key="test-key",
        model="universal-3-5-pro",
        mode="max_accuracy",
        min_turn_silence=400,
        max_turn_silence=2000,
    )
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["mode"] == ["max_accuracy"]
    assert query["min_turn_silence"] == ["400"]
    assert query["max_turn_silence"] == ["2000"]


async def test_silence_defaults_injected_without_mode():
    """Without `mode`, the plugin still injects its latency-optimized 100ms
    min/max turn silence defaults (the LiveKit default behavior)."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "mode" not in query
    assert query["min_turn_silence"] == ["100"]
    assert query["max_turn_silence"] == ["100"]


# ---------------------------------------------------------------------------
# language_code / language_codes (language steering)
#
# `language_codes` biases transcription toward one or more expected languages
# (e.g. ["en", "es"]) instead of automatic detection/code-switching across all
# supported languages. `language_code` is shorthand for a one-element list;
# both normalize into `_opts.language_codes`. Steering is only applied by the
# u3-pro ASR, so — like `mode` and the context/voice-focus params — both are
# u3-rt-pro-family-only at construction. `language_codes` (unlike the
# singular) can also be re-steered mid-session via update_options, matching
# the AssemblyAI streaming API's UpdateConfiguration; an empty list clears
# steering back to the model default.
# ---------------------------------------------------------------------------


async def test_single_language_code_normalized_to_iso_639_1():
    """A single language_codes entry is normalized to a bare ISO 639-1 code regardless of input format."""
    from livekit.plugins.assemblyai import STT

    for raw, expected in (
        ("es", "es"),
        ("es-ES", "es"),
        ("Spanish", "es"),
        ("en-US", "en"),
        ("english", "en"),
        ("pt-BR", "pt"),
    ):
        stt = STT(api_key="test-key", model="u3-rt-pro", language_codes=raw)
        assert stt._opts.language_codes == [expected]


async def test_language_codes_normalized_and_deduped():
    """Each entry is normalized; duplicates after normalization are dropped,
    preserving first-seen order."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="u3-rt-pro",
        language_codes=["en-US", "en-GB", "Spanish", "es"],
    )
    assert stt._opts.language_codes == ["en", "es"]


async def test_language_codes_max_10():
    """More than 10 codes after normalization/dedup raises (server limit)."""
    from livekit.plugins.assemblyai import STT

    codes = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko"]
    with pytest.raises(ValueError, match="at most 10"):
        STT(api_key="test-key", model="u3-rt-pro", language_codes=codes)


async def test_language_codes_max_10_applies_after_dedup():
    """11 raw entries that dedup to 10 codes are accepted — the client is never
    stricter than what it actually sends to the server."""
    from livekit.plugins.assemblyai import STT

    codes = ["en-US", "en-GB", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja"]
    stt = STT(api_key="test-key", model="u3-rt-pro", language_codes=codes)
    assert stt._opts.language_codes == [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "pl",
        "ru",
        "ja",
    ]


async def test_language_codes_multi_cannot_be_combined():
    """'multi' routes to the unsteered multilingual model and must be sent alone."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="multi"):
        STT(api_key="test-key", model="u3-rt-pro", language_codes=["multi", "en"])


async def test_language_codes_requires_u3_pro_family():
    """language_codes raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="language_codes"):
        STT(
            api_key="test-key",
            model="universal-streaming-multilingual",
            language_codes=["es"],
        )


async def test_language_codes_in_connect_config():
    """language_codes is sent as a JSON array in the connect config query."""
    import json
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro", language_codes=["en", "es"])
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert json.loads(query["language_codes"][0]) == ["en", "es"]


async def test_language_code_singular_in_connect_config():
    """A singular language_code is sent as a one-element language_codes JSON
    array — the server aliases both names to the same field."""
    import json
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro", language_codes="es")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert json.loads(query["language_codes"][0]) == ["es"]
    assert "language_code" not in query


async def test_language_codes_absent_from_connect_config_when_unset():
    """language_codes key is omitted from the connect config when not set."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "language_codes" not in query
    assert "language_code" not in query


async def test_language_codes_stream_sends_update_configuration():
    """SpeechStream.update_options enqueues an UpdateConfiguration message
    containing language_codes."""
    stream = _make_stream_for_unit_test()

    stream.update_options(language_codes=["en", "es"])

    assert stream._opts.language_codes == ["en", "es"]
    msg = stream._config_update_queue.get_nowait()
    assert msg["type"] == "UpdateConfiguration"
    assert msg["language_codes"] == ["en", "es"]


async def test_language_codes_update_normalizes_input():
    """Mid-session updates get the same normalization/dedup as the constructor."""
    stream = _make_stream_for_unit_test()

    stream.update_options(language_codes=["en-US", "Spanish"])

    msg = stream._config_update_queue.get_nowait()
    assert msg["language_codes"] == ["en", "es"]


async def test_language_codes_update_validates_input():
    """Mid-session updates get the same fail-fast validation as the constructor."""
    stream = _make_stream_for_unit_test()

    with pytest.raises(ValueError, match="multi"):
        stream.update_options(language_codes=["multi", "en"])


async def test_language_codes_empty_list_clears_steering():
    """An explicit empty list is forwarded (the server's documented 'clear
    steering back to model default' form), not dropped by a truthiness check."""
    stream = _make_stream_for_unit_test()

    stream.update_options(language_codes=[])

    assert stream._opts.language_codes == []
    msg = stream._config_update_queue.get_nowait()
    assert msg["type"] == "UpdateConfiguration"
    assert msg["language_codes"] == []


async def test_language_codes_propagates_from_stt_to_active_stream():
    """STT.update_options(language_codes=...) propagates to an already-active
    stream and sends it over that stream's websocket queue."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()

    stt.update_options(language_codes=["en", "es"])

    assert stt._opts.language_codes == ["en", "es"]
    assert stream._opts.language_codes == ["en", "es"]
    msg = stream._config_update_queue.get_nowait()
    assert msg["type"] == "UpdateConfiguration"
    assert msg["language_codes"] == ["en", "es"]


async def test_cleared_language_codes_omitted_on_reconnect():
    """After clearing ([]), a (re)connect omits the language_codes param —
    consistent with the cleared state."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro", language_codes=["en", "es"])
    stream = _make_stream_for_unit_test(stt)
    stream.update_options(language_codes=[])
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "language_codes" not in query


async def test_language_codes_update_requires_u3_pro_family():
    """update_options mirrors the constructor's family gate for language_codes."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="universal-streaming-english")
    with pytest.raises(ValueError, match="language_codes"):
        stt.update_options(language_codes=["es"])

    stream = _make_stream_for_unit_test(stt)
    with pytest.raises(ValueError, match="language_codes"):
        stream.update_options(language_codes=["es"])
    assert stream._config_update_queue.empty()


async def test_language_codes_empty_list_at_construction_is_unset():
    """An explicit [] at construction equals unset — even on non-U3-Pro models."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="universal-streaming-english", language_codes=[])
    assert stt._opts.language_codes is NOT_GIVEN

    stt_pro = STT(api_key="test-key", model="u3-rt-pro", language_codes=[])
    assert stt_pro._opts.language_codes is NOT_GIVEN


async def test_language_codes_accepts_bare_string():
    """language_codes accepts a single code directly, normalized like a list entry."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", language_codes="es")
    assert stt._opts.language_codes == ["es"]

    stt_norm = STT(api_key="test-key", model="u3-rt-pro", language_codes="Spanish")
    assert stt_norm._opts.language_codes == ["es"]


# ---------------------------------------------------------------------------
# agent_context server cap (1750 chars) and chat-context carryover
#
# The server rejects `agent_context` longer than 1750 chars, and an invalid
# UpdateConfiguration cancels the whole streaming session — so the plugin
# enforces the cap client-side. The auto-carryover path truncates (keeping the
# tail: the end of the agent's reply is what the user responds to); explicit
# developer input fails fast with ValueError instead of being silently
# mangled.
# ---------------------------------------------------------------------------


def _assistant_item_event(text: str):
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent

    return ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[text]))


async def test_agent_context_at_cap_accepted():
    """agent_context of exactly 1750 chars is accepted everywhere."""
    from livekit.plugins.assemblyai import STT

    text = "x" * 1750
    stt = STT(api_key="test-key", agent_context=text)
    assert stt._opts.agent_context == text

    stt.update_options(agent_context=text)
    assert stt._opts.agent_context == text


async def test_agent_context_over_cap_raises_in_constructor():
    """Explicit oversize agent_context fails fast instead of killing the session later."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="agent_context"):
        STT(api_key="test-key", agent_context="x" * 1751)


async def test_agent_context_over_cap_raises_in_stt_update_options():
    """STT.update_options rejects oversize agent_context without mutating any option."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", vad_threshold=0.5)
    with pytest.raises(ValueError, match="agent_context"):
        stt.update_options(vad_threshold=0.9, agent_context="x" * 1751)

    # the same call must not have applied its other updates (no partial update)
    assert stt._opts.vad_threshold == 0.5
    assert stt._opts.agent_context is NOT_GIVEN


async def test_agent_context_over_cap_raises_in_stream_update_options():
    """SpeechStream.update_options rejects oversize agent_context without
    mutating options or enqueuing an UpdateConfiguration."""
    stream = _make_stream_for_unit_test()

    with pytest.raises(ValueError, match="agent_context"):
        stream.update_options(vad_threshold=0.9, agent_context="x" * 1751)

    assert stream._opts.agent_context is NOT_GIVEN
    assert stream._opts.vad_threshold is NOT_GIVEN
    assert stream._config_update_queue.empty()


async def test_carryover_forwards_short_reply_verbatim():
    """_push_conversation_item forwards assistant text within the cap unchanged."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    stt._push_conversation_item("assistant", "Your room is booked for Tuesday.")
    assert stt._opts.agent_context == "Your room is booked for Tuesday."


async def test_carryover_truncates_oversize_reply_keeping_tail():
    """_push_conversation_item truncates oversize replies to the last 1750 chars."""
    from livekit.plugins.assemblyai import STT

    text = "a" * 2000 + "b" * 1750
    stt = STT(api_key="test-key")
    stt._push_conversation_item("assistant", text)
    assert stt._opts.agent_context == "b" * 1750


async def test_carryover_ignores_non_assistant_items():
    """_push_conversation_item ignores user messages and textless assistant items
    — the shapes it receives when carryover is enabled."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")

    stt._push_conversation_item("user", "hi there")
    assert stt._opts.agent_context is NOT_GIVEN

    stt._push_conversation_item("assistant", "")
    assert stt._opts.agent_context is NOT_GIVEN


async def test_carryover_ignores_agent_handoff_items():
    """_push_conversation_item ignores non-message items (e.g. AgentHandoff),
    which have `.type != "message"` and no `text_content`."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")

    stt._push_conversation_item("agent_handoff", "")
    assert stt._opts.agent_context is NOT_GIVEN


async def test_carryover_on_by_default_for_u3_pro_family():
    """chat_context carryover is on by default on models that support it."""
    from livekit.plugins.assemblyai import STT

    for model in (
        "u3-rt-pro",
        "u3-rt-pro-beta-1",
        "universal-3-5-pro",
        "universal-3-6-pro",
    ):
        stt = STT(api_key="test-key", model=model)
        assert stt.capabilities.chat_context is True


async def test_carryover_off_for_unsupported_models():
    """Models outside the U3-Pro family do not advertise chat-context carryover."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="universal-streaming-english")
    assert stt.capabilities.chat_context is False


# ---------------------------------------------------------------------------
# end_of_turn_confidence surfaced on SpeechData.metadata
#
# The server reports a per-Turn `end_of_turn_confidence`. On Universal-3.5 Pro
# (and later) this rises from 0 toward 1 across the partials emitted while a
# turn is held open between min_turn_silence and max_turn_silence, and is 1.0 on
# the final. The plugin surfaces it on SpeechData.metadata so callers can
# threshold it (e.g. to trigger preemptive/eager LLM generation) from a
# SpeechEvent / stt_node / UserInputTranscribedEvent without subclassing the
# stream. It's only attached when the message carries the field, so models that
# never emit it are unaffected.
# ---------------------------------------------------------------------------


def _drain_events(stream):
    from livekit.agents.utils.aio import ChanEmpty

    events = []
    while True:
        try:
            events.append(stream._event_ch.recv_nowait())
        except ChanEmpty:
            break
    return events


def _turn_message(**overrides):
    """A minimal server Turn message with a single word (so an interim fires)."""
    msg = {
        "type": "Turn",
        "words": [{"text": "hello", "start": 0, "end": 480, "confidence": 0.9}],
        "end_of_turn": False,
        "transcript": "",
    }
    msg.update(overrides)
    return msg


async def test_end_of_turn_confidence_surfaced_on_interim_metadata():
    """A held-turn partial carries its end_of_turn_confidence on interim metadata."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(_turn_message(end_of_turn_confidence=0.55))

    interim = [e for e in _drain_events(stream) if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
    assert interim
    assert interim[0].alternatives[0].metadata == {"end_of_turn_confidence": 0.55}


async def test_end_of_turn_confidence_surfaced_on_final_metadata():
    """The final transcript carries end_of_turn_confidence (1.0) on its metadata."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(
        _turn_message(end_of_turn=True, transcript="hello", end_of_turn_confidence=1.0)
    )

    final = [e for e in _drain_events(stream) if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert final
    assert final[0].alternatives[0].metadata == {"end_of_turn_confidence": 1.0}


async def test_end_of_turn_confidence_zero_is_surfaced():
    """0.0 on an early partial is a real value (turn just started), not 'absent',
    so it must still surface — mirrors the value seen before the ramp climbs."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(_turn_message(end_of_turn_confidence=0.0))

    interim = [e for e in _drain_events(stream) if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
    assert interim
    assert interim[0].alternatives[0].metadata == {"end_of_turn_confidence": 0.0}


async def test_end_of_turn_confidence_absent_leaves_metadata_none():
    """A model/message that doesn't include end_of_turn_confidence leaves metadata
    unset, so existing consumers are unaffected."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(_turn_message())  # no end_of_turn_confidence key

    interim = [e for e in _drain_events(stream) if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
    assert interim
    assert interim[0].alternatives[0].metadata is None


# ---------------------------------------------------------------------------
# language_confidence surfaced on SpeechData.metadata
#
# When `language_detection` is enabled (the plugin's default for U3 Pro and
# multilingual models) Turn messages carry `language_code` alongside a
# `language_confidence` between 0 and 1 scoring it. Both are populated only when
# an utterance is complete or the turn is final, so a plain interim carries
# neither. The plugin passes the score through on SpeechData.metadata untouched.
# ---------------------------------------------------------------------------


async def test_language_confidence_surfaced_on_final_metadata():
    """A final turn carries the detected language's confidence on its metadata."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(
        _turn_message(end_of_turn=True, transcript="hola", language_confidence=0.92)
    )

    final = [e for e in _drain_events(stream) if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert final
    assert final[0].alternatives[0].metadata == {"language_confidence": 0.92}


async def test_language_confidence_surfaced_on_preflight_metadata():
    """An utterance-carrying turn surfaces the confidence on its preflight event."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(_turn_message(utterance="hola", language_confidence=0.81))

    preflight = [e for e in _drain_events(stream) if e.type == SpeechEventType.PREFLIGHT_TRANSCRIPT]
    assert preflight
    assert preflight[0].alternatives[0].metadata == {"language_confidence": 0.81}


async def test_language_confidence_absent_on_plain_interim():
    """The server omits language_confidence on a plain partial, so metadata stays
    unset rather than carrying a fabricated score."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(_turn_message())  # no language_confidence key

    interim = [e for e in _drain_events(stream) if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
    assert interim
    assert interim[0].alternatives[0].metadata is None


async def test_language_confidence_zero_is_surfaced():
    """0 is the bottom of the documented 0-to-1 range, so it is a real score
    rather than an absent one and must not be dropped as falsy -- a truthiness
    check here would silently discard the least confident detections."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(
        _turn_message(end_of_turn=True, transcript="hello", language_confidence=0.0)
    )

    final = [e for e in _drain_events(stream) if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert final
    assert final[0].alternatives[0].metadata == {"language_confidence": 0.0}


async def test_language_confidence_coexists_with_end_of_turn_confidence():
    """Both scores ride the same metadata dict when the turn carries both."""
    stream = _make_stream_for_unit_test()
    stream._process_stream_event(
        _turn_message(
            end_of_turn=True,
            transcript="hola",
            end_of_turn_confidence=1.0,
            language_confidence=0.92,
        )
    )

    final = [e for e in _drain_events(stream) if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert final
    assert final[0].alternatives[0].metadata == {
        "end_of_turn_confidence": 1.0,
        "language_confidence": 0.92,
    }
