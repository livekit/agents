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


def _make_stream_for_unit_test():
    """Construct a SpeechStream without triggering the _main_task WebSocket
    loop. Patches asyncio.create_task during __init__ so the stream doesn't
    try to open a real connection; also closes the coroutines that would
    otherwise be scheduled, to avoid un-awaited coroutine warnings."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.assemblyai import STT
    from livekit.plugins.assemblyai.stt import SpeechStream

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
    """Test continuous_partials is not set by default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key")
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
        STT(api_key="test-key", continuous_partials=True)


async def test_continuous_partials_with_u3_pro_alias():
    """Test continuous_partials works with the deprecated 'u3-pro' alias (rewritten to u3-rt-pro)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-pro", continuous_partials=True)
    assert stt._opts.continuous_partials is True
    assert stt._opts.speech_model == "u3-rt-pro"


async def test_continuous_partials_update():
    """Test continuous_partials can be updated dynamically via update_options."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", continuous_partials=False)
    stt.update_options(continuous_partials=True)
    assert stt._opts.continuous_partials is True


async def test_continuous_partials_defaults_to_true_for_u3_rt_pro():
    """Test continuous_partials defaults to True when model is u3-rt-pro (LiveKit-only
    default; AssemblyAI server defaults to False)."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.continuous_partials is True


async def test_continuous_partials_explicit_false_overrides_livekit_default():
    """Test explicit continuous_partials=False overrides the LiveKit-only True default."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro", continuous_partials=False)
    assert stt._opts.continuous_partials is False


async def test_continuous_partials_update_from_default():
    """Test continuous_partials can be updated via update_options away from LiveKit default."""
    from livekit.plugins.assemblyai import STT

    # LiveKit defaults this to True for u3-rt-pro
    stt = STT(api_key="test-key", model="u3-rt-pro")
    assert stt._opts.continuous_partials is True

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
        STT(api_key="test-key", interruption_delay=200)


# ---------------------------------------------------------------------------
# agent_context
#
# agent_context carries "what the agent said" so the model can use it to bias
# transcription of the user's reply. It is threaded through STTOptions, the
# constructor, and both update_options paths, and is sent over the live
# websocket as an UpdateConfiguration message. `enable_agent_context` wires the
# agent's spoken turn into update_options automatically for an AgentSession.
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


async def test_enable_agent_context_pushes_assistant_turn():
    """enable_agent_context updates the STT's agent_context with the assistant's
    spoken text on conversation_item_added."""
    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="u3-rt-pro")
    session = AgentSession(stt=stt)
    enable_agent_context(session)

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(
            item=ChatMessage(role="assistant", content=["What date works for you?"])
        ),
    )

    assert stt._opts.agent_context == "What date works for you?"


async def test_enable_agent_context_ignores_user_turn():
    """User messages must not update agent_context."""
    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="u3-rt-pro")
    session = AgentSession(stt=stt)
    enable_agent_context(session)

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="user", content=["next Tuesday"])),
    )

    assert stt._opts.agent_context is NOT_GIVEN


async def test_enable_agent_context_ignores_empty_assistant_turn():
    """Assistant messages with no text content (e.g. tool-only) are skipped."""
    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="u3-rt-pro")
    session = AgentSession(stt=stt)
    enable_agent_context(session)

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[])),
    )

    assert stt._opts.agent_context is NOT_GIVEN


async def test_enable_agent_context_unsubscribe_stops_updates():
    """The returned callable unsubscribes the handler."""
    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="u3-rt-pro")
    session = AgentSession(stt=stt)
    unsubscribe = enable_agent_context(session)

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["first"])),
    )
    assert stt._opts.agent_context == "first"

    unsubscribe()

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["second"])),
    )
    # Still the pre-unsubscribe value.
    assert stt._opts.agent_context == "first"


async def test_enable_agent_context_no_op_for_non_assemblyai_stt(caplog):
    """When the session STT is not an AssemblyAI STT, the helper is a no-op and
    warns once rather than raising."""
    import logging

    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    session = AgentSession(stt=STT(api_key="test-key"))
    enable_agent_context(session)
    # Simulate a session whose STT is not an AssemblyAI STT.
    session._stt = None

    ev = ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["hi"]))
    with caplog.at_level(logging.WARNING):
        session.emit("conversation_item_added", ev)
        session.emit("conversation_item_added", ev)

    warnings = [r for r in caplog.records if "agent_context" in r.getMessage()]
    assert len(warnings) == 1


async def test_enable_agent_context_no_op_for_non_u3_pro_model(caplog):
    """When the STT is an AssemblyAI STT but not a u3-rt-pro model, the helper is
    a no-op and warns once (agent_context is u3-rt-pro only)."""
    import logging

    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="universal-streaming-english")
    session = AgentSession(stt=stt)
    enable_agent_context(session)

    ev = ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["hi"]))
    with caplog.at_level(logging.WARNING):
        session.emit("conversation_item_added", ev)
        session.emit("conversation_item_added", ev)

    assert stt._opts.agent_context is NOT_GIVEN
    warnings = [r for r in caplog.records if "u3-rt-pro" in r.getMessage()]
    assert len(warnings) == 1


async def test_enable_agent_context_truncates_long_text():
    """agent_context is truncated to AssemblyAI's 1500-character limit."""
    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.assemblyai import STT, enable_agent_context

    stt = STT(api_key="test-key", model="u3-rt-pro")
    session = AgentSession(stt=stt)
    enable_agent_context(session)

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["a" * 2000])),
    )

    assert stt._opts.agent_context == "a" * 1500


# ---------------------------------------------------------------------------
# u3-rt-pro-beta-1 model + u3-pro param family
#
# u3-rt-pro-beta-1 shares all u3-rt-pro behavior, so the u3-pro-gated params
# (prompt, agent_context, continuous_partials, interruption_delay) are accepted
# with it, and continuous_partials defaults to True.
# ---------------------------------------------------------------------------


async def test_u3_rt_pro_beta_1_accepted():
    """u3-rt-pro-beta-1 is a valid model and gets the u3-rt-pro defaults."""
    from livekit.plugins.assemblyai import STT

    stt = STT(api_key="test-key", model="u3-rt-pro-beta-1")
    assert stt._opts.speech_model == "u3-rt-pro-beta-1"
    # continuous_partials defaults to True for the u3-rt-pro family
    assert stt._opts.continuous_partials is True


async def test_u3_rt_pro_beta_1_accepts_u3_pro_params():
    """The u3-pro-gated params are accepted with u3-rt-pro-beta-1."""
    from livekit.plugins.assemblyai import STT

    stt = STT(
        api_key="test-key",
        model="u3-rt-pro-beta-1",
        prompt="medical dictation",
        agent_context="The agent asked for the patient's name.",
        interruption_delay=300,
    )
    assert stt._opts.prompt == "medical dictation"
    assert stt._opts.agent_context == "The agent asked for the patient's name."
    assert stt._opts.interruption_delay == 300


# ---------------------------------------------------------------------------
# agent_context is u3-rt-pro-only
# ---------------------------------------------------------------------------


async def test_agent_context_requires_u3_rt_pro():
    """agent_context in the constructor raises for non-u3-rt-pro models."""
    from livekit.plugins.assemblyai import STT

    with pytest.raises(ValueError, match="agent_context"):
        STT(api_key="test-key", agent_context="hello")


async def test_agent_context_allowed_for_u3_rt_pro_models():
    """agent_context is accepted for both u3-rt-pro and u3-rt-pro-beta-1."""
    from livekit.plugins.assemblyai import STT

    for model in ("u3-rt-pro", "u3-rt-pro-beta-1"):
        stt = STT(api_key="test-key", model=model, agent_context="ctx")
        assert stt._opts.agent_context == "ctx"
