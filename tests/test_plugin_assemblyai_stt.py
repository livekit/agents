"""Tests for AssemblyAI STT plugin configuration options."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.stt import SpeechEventType
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.plugin("assemblyai")


async def test_vad_threshold_default():
    """Test vad_threshold is not set by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.vad_threshold is NOT_GIVEN


async def test_vad_threshold_set():
    """Test vad_threshold can be set in constructor."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", vad_threshold=0.3)
    assert stt._opts.vad_threshold == 0.3


async def test_vad_threshold_boundary_values():
    """Test vad_threshold accepts boundary values (0 and 1)."""
    from livekit.plugins.assemblyai import STT

    stt_low = STT(api_key="test-key", vad_threshold=0.0)
    assert stt_low._opts.vad_threshold == 0.0

    stt_high = STT(api_key="test-key", vad_threshold=1.0)
    assert stt_high._opts.vad_threshold == 1.0


async def test_vad_threshold_update():
    """Test vad_threshold can be updated dynamically."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", vad_threshold=0.5)
    stt.update_options(vad_threshold=0.7)
    assert stt._opts.vad_threshold == 0.7


async def test_vad_threshold_update_from_default():
    """Test vad_threshold can be set via update_options when not initially set."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.vad_threshold is NOT_GIVEN

    stt.update_options(vad_threshold=0.4)
    assert stt._opts.vad_threshold == 0.4


async def test_vad_threshold_with_other_options():
    """Test vad_threshold works alongside other options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        vad_threshold=0.6,
        end_of_turn_confidence_threshold=0.8,
        max_turn_silence=1000,
    )
    assert stt._opts.vad_threshold == 0.6
    assert stt._opts.end_of_turn_confidence_threshold == 0.8
    assert stt._opts.max_turn_silence == 1000


async def test_vad_threshold_partial_update():
    """Test updating vad_threshold doesn't affect other options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        vad_threshold=0.5,
        max_turn_silence=500,
    )
    stt.update_options(vad_threshold=0.8)

    assert stt._opts.vad_threshold == 0.8
    assert stt._opts.max_turn_silence == 500


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

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
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


async def test_start_time_has_default_before_plugin_override():
    """Even without any plugin override, the base-class default
    (time.time() seeded in __init__) is available for the SpeechStarted
    conversion to use."""
    stream = _make_stream_for_unit_test()

    # start_time should already be a recent wall-clock value from the base
    # class __init__, without any explicit override.
    assert time.time() - stream.start_time < 5.0


async def test_continuous_partials_default():
    """Test continuous_partials is not set by default so AssemblyAI's server defaults
    apply (enabled, except when speaker_labels is on)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="universal-streaming-english")
    assert stt._opts.continuous_partials is NOT_GIVEN


async def test_continuous_partials_set():
    """Test continuous_partials can be set in constructor with u3-rt-pro."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", continuous_partials=True)
    assert stt._opts.continuous_partials is True


async def test_continuous_partials_requires_u3_rt_pro():
    """Test continuous_partials raises ValueError when used with a non-u3-rt-pro model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="continuous_partials"):
        STT(api_key="test-key", model="universal-streaming-english", continuous_partials=True)


async def test_continuous_partials_with_u3_pro_alias():
    """continuous_partials works with the deprecated 'u3-pro' alias (rewritten to
    universal-3-5-pro)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-pro", continuous_partials=True)
    assert stt._opts.continuous_partials is True
    assert stt._opts.speech_model == "universal-3-5-pro"


async def test_u3_pro_deprecated_rewrites_to_universal_3_5_pro():
    """The deprecated 'u3-pro' alias warns and is rewritten to the recommended
    default model 'universal-3-5-pro'."""
    from livekit.plugins.assemblyai import STT

    with patch("livekit.plugins.assemblyai.stt.logger") as mock_logger:
        stt = STT(api_key="test-key", model="u3-pro")

    assert stt._opts.speech_model == "universal-3-5-pro"
    mock_logger.warning.assert_called_once()
    assert "universal-3-5-pro" in mock_logger.warning.call_args.args[0]


async def test_continuous_partials_update():
    """Test continuous_partials can be updated dynamically via update_options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", continuous_partials=False)
    stt.update_options(continuous_partials=True)
    assert stt._opts.continuous_partials is True


async def test_continuous_partials_unset_by_default_for_u3_rt_pro():
    """continuous_partials is left unset for u3-rt-pro so AssemblyAI's server defaults
    apply (enabled, but disabled by the server when speaker_labels is on)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.continuous_partials is NOT_GIVEN


async def test_continuous_partials_explicit_false():
    """Test explicit continuous_partials=False is preserved."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", continuous_partials=False)
    assert stt._opts.continuous_partials is False


async def test_continuous_partials_update_from_default():
    """Test continuous_partials can be set via update_options when unset at construction."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.continuous_partials is NOT_GIVEN

    stt.update_options(continuous_partials=False)
    assert stt._opts.continuous_partials is False


async def test_interruption_delay_update():
    """Test interruption_delay can be updated dynamically via update_options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", interruption_delay=200)
    stt.update_options(interruption_delay=750)
    assert stt._opts.interruption_delay == 750


async def test_interruption_delay_update_from_default():
    """Test interruption_delay can be set via update_options when not initially set."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.interruption_delay is NOT_GIVEN

    stt.update_options(interruption_delay=300)
    assert stt._opts.interruption_delay == 300


async def test_interruption_delay_default():
    """Test interruption_delay is not set by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.interruption_delay is NOT_GIVEN


async def test_interruption_delay_set():
    """Test interruption_delay can be set in constructor with u3-rt-pro."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", interruption_delay=200)
    assert stt._opts.interruption_delay == 200


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


async def test_agent_context_default():
    """Test agent_context is not set by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.agent_context is NOT_GIVEN


async def test_agent_context_set():
    """Test agent_context can be set in the constructor (u3-rt-pro only)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="u3-rt-pro",
        agent_context="The agent asked for a booking date.",
    )
    assert stt._opts.agent_context == "The agent asked for a booking date."


async def test_agent_context_update():
    """Test agent_context can be updated dynamically via update_options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.agent_context is NOT_GIVEN

    stt.update_options(agent_context="What is your account number?")
    assert stt._opts.agent_context == "What is your account number?"

    # A subsequent update overwrites (most-recent-turn semantics).
    stt.update_options(agent_context="Thanks, and your zip code?")
    assert stt._opts.agent_context == "Thanks, and your zip code?"


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

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
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


async def test_u3_rt_pro_beta_1_accepted():
    """u3-rt-pro-beta-1 is a valid model and gets the u3-rt-pro defaults."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro-beta-1")
    assert stt._opts.speech_model == "u3-rt-pro-beta-1"
    # continuous_partials is left unset so AssemblyAI's server defaults apply
    assert stt._opts.continuous_partials is NOT_GIVEN


async def test_u3_rt_pro_beta_1_accepts_u3_pro_params():
    """The u3-pro-gated params are accepted with u3-rt-pro-beta-1."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="u3-rt-pro-beta-1",
        prompt="medical dictation",
        agent_context="The agent asked for the patient's name.",
        previous_context_n_turns=10,
        interruption_delay=300,
    )
    assert stt._opts.prompt == "medical dictation"
    assert stt._opts.agent_context == "The agent asked for the patient's name."
    assert stt._opts.previous_context_n_turns == 10
    assert stt._opts.interruption_delay == 300


# ---------------------------------------------------------------------------
# agent_context is u3-rt-pro-only
# ---------------------------------------------------------------------------


async def test_agent_context_requires_u3_rt_pro():
    """agent_context in the constructor raises for non-u3-rt-pro models."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="agent_context"):
        STT(api_key="test-key", model="universal-streaming-english", agent_context="hello")


async def test_agent_context_allowed_for_u3_rt_pro_models():
    """agent_context is accepted for both u3-rt-pro and u3-rt-pro-beta-1."""
    from livekit.plugins.assemblyai import STT

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1"):
        stt = STT(api_key="test-key", model=model, agent_context="ctx")
        assert stt._opts.agent_context == "ctx"


# ---------------------------------------------------------------------------
# previous_context_n_turns (u3-rt-pro only, connect-only)
# ---------------------------------------------------------------------------


async def test_previous_context_n_turns_set():
    """previous_context_n_turns can be set for u3-rt-pro models."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", previous_context_n_turns=5)
    assert stt._opts.previous_context_n_turns == 5


async def test_previous_context_n_turns_default_unset():
    """previous_context_n_turns is unset by default (server default applies)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.previous_context_n_turns is NOT_GIVEN


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
# universal-3-5-pro: default model + u3-rt-pro parameter family
#
# universal-3-5-pro is the plugin's default model and belongs to the u3-rt-pro
# parameter family, so it accepts the u3-pro-gated params (prompt,
# agent_context, previous_context_n_turns, continuous_partials,
# interruption_delay) and inherits the family's connect-time defaults.
# ---------------------------------------------------------------------------


async def test_default_model_is_universal_3_5_pro():
    """The plugin defaults to universal-3-5-pro."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt.model == "universal-3-5-pro"
    assert stt._opts.speech_model == "universal-3-5-pro"


async def test_universal_3_5_pro_accepts_u3_pro_params():
    """universal-3-5-pro shares the u3-rt-pro parameter family."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="universal-3-5-pro",
        prompt="medical dictation",
        agent_context="The agent asked for the patient's name.",
        previous_context_n_turns=10,
        interruption_delay=300,
    )
    assert stt._opts.speech_model == "universal-3-5-pro"
    assert stt._opts.prompt == "medical dictation"
    assert stt._opts.agent_context == "The agent asked for the patient's name."
    assert stt._opts.previous_context_n_turns == 10
    assert stt._opts.interruption_delay == 300


async def test_universal_3_5_pro_leaves_continuous_partials_unset():
    """continuous_partials is left unset for universal-3-5-pro so AssemblyAI's server
    defaults apply (enabled, but disabled by the server when speaker_labels is on)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="universal-3-5-pro")
    assert stt._opts.continuous_partials is NOT_GIVEN


# ---------------------------------------------------------------------------
# voice_focus / voice_focus_threshold
#
# Voice Focus isolates the primary voice and suppresses background noise.
# voice_focus is a string enum ("near-field" / "far-field"); voice_focus_threshold
# is a float in [0, 1]. Both are u3-rt-pro-family-only and connect-time only
# (not exposed via update_options).
# ---------------------------------------------------------------------------


async def test_voice_focus_default():
    """voice_focus and voice_focus_threshold are unset by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.voice_focus is NOT_GIVEN
    assert stt._opts.voice_focus_threshold is NOT_GIVEN


async def test_voice_focus_set():
    """voice_focus accepts the documented near-field / far-field values."""
    from livekit.plugins.assemblyai import STT

    near = STT(api_key="test-key", voice_focus="near-field")
    assert near._opts.voice_focus == "near-field"

    far = STT(api_key="test-key", voice_focus="far-field")
    assert far._opts.voice_focus == "far-field"


async def test_voice_focus_threshold_set():
    """voice_focus_threshold is stored on the options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", voice_focus="near-field", voice_focus_threshold=0.7)
    assert stt._opts.voice_focus_threshold == 0.7


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


async def test_voice_focus_connect_time_only():
    """voice_focus is connect-time only — not exposed via update_options."""
    import inspect

    from livekit.plugins.assemblyai import STT
    from livekit.plugins.assemblyai.stt import SpeechStream

    assert "voice_focus" not in inspect.signature(STT.update_options).parameters
    assert "voice_focus_threshold" not in inspect.signature(STT.update_options).parameters
    assert "voice_focus" not in inspect.signature(SpeechStream.update_options).parameters


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

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro"):
        stt = STT(api_key="test-key", model=model, voice_focus="far-field")
        assert stt._opts.voice_focus == "far-field"


async def test_default_model_leaves_continuous_partials_unset():
    """A bare STT() (relying on the default model) leaves continuous_partials unset so
    AssemblyAI's server defaults apply (enabled, but disabled by the server when
    speaker_labels is on)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.continuous_partials is NOT_GIVEN


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


async def test_mode_default():
    """mode is unset by default (server default of 'balanced' applies)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.mode is NOT_GIVEN


async def test_mode_set():
    """mode accepts each documented value."""
    from livekit.plugins.assemblyai import STT

    for value in ("min_latency", "balanced", "max_accuracy"):
        stt = STT(api_key="test-key", model="u3-rt-pro", mode=value)
        assert stt._opts.mode == value


async def test_mode_requires_u3_pro_family():
    """mode raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="mode"):
        STT(api_key="test-key", model="universal-streaming-english", mode="max_accuracy")


async def test_mode_allowed_for_all_u3_pro_family_models():
    """mode is accepted for every u3-rt-pro-family model, not just the default."""
    from livekit.plugins.assemblyai import STT

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro"):
        stt = STT(api_key="test-key", model=model, mode="min_latency")
        assert stt._opts.mode == "min_latency"


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


async def test_mode_connect_time_only():
    """mode is connect-time only — not exposed via update_options."""
    import inspect

    from livekit.plugins.assemblyai import STT
    from livekit.plugins.assemblyai.stt import SpeechStream

    assert "mode" not in inspect.signature(STT.update_options).parameters
    assert "mode" not in inspect.signature(SpeechStream.update_options).parameters


# ---------------------------------------------------------------------------
# language_code (language steering)
#
# `language_code` biases transcription toward a single language (e.g. "en",
# "es") instead of automatic detection/code-switching. Steering is only applied
# by the u3-pro ASR, so — like `mode` and the context/voice-focus params — it is
# u3-rt-pro-family-only and connect-time only (not exposed via update_options,
# matching the AssemblyAI streaming API, where `language_code` is a connect-time
# parameter and not part of UpdateConfiguration).
# ---------------------------------------------------------------------------


async def test_language_code_default():
    """language_code is unset by default (automatic detection applies)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
    assert stt._opts.language_code is NOT_GIVEN


async def test_language_code_set():
    """language_code can be set in the constructor."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", language_code="es")
    assert stt._opts.language_code == "es"


async def test_language_code_normalized_to_iso_639_1():
    """language_code is normalized to a bare ISO 639-1 code regardless of input format."""
    from livekit.plugins.assemblyai import STT

    for raw, expected in (
        ("es", "es"),
        ("es-ES", "es"),
        ("Spanish", "es"),
        ("en-US", "en"),
        ("english", "en"),
        ("pt-BR", "pt"),
    ):
        stt = STT(api_key="test-key", model="u3-rt-pro", language_code=raw)
        assert stt._opts.language_code == expected


async def test_language_code_requires_u3_pro_family():
    """language_code raises ValueError when used with a non-u3-rt-pro-family model."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="language_code"):
        STT(api_key="test-key", model="universal-streaming-multilingual", language_code="es")


async def test_language_code_allowed_for_all_u3_pro_family_models():
    """language_code is accepted for every u3-rt-pro-family model, not just the default."""
    from livekit.plugins.assemblyai import STT

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1", "universal-3-5-pro"):
        stt = STT(api_key="test-key", model=model, language_code="en")
        assert stt._opts.language_code == "en"


async def test_language_code_in_connect_config():
    """language_code is sent in the connect config query."""
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.assemblyai import STT

    captured: dict = {}

    async def _fake_ws_connect(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    stt = STT(api_key="test-key", model="universal-3-5-pro", language_code="es")
    stream = _make_stream_for_unit_test(stt)
    stream._session.ws_connect = _fake_ws_connect
    await stream._connect_ws()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["language_code"] == ["es"]


async def test_language_code_absent_from_connect_config_when_unset():
    """language_code key is omitted from the connect config when not set."""
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
    assert "language_code" not in query


async def test_language_code_connect_time_only():
    """language_code is connect-time only — not exposed via update_options."""
    import inspect

    from livekit.plugins.assemblyai import STT
    from livekit.plugins.assemblyai.stt import SpeechStream

    assert "language_code" not in inspect.signature(STT.update_options).parameters
    assert "language_code" not in inspect.signature(SpeechStream.update_options).parameters
