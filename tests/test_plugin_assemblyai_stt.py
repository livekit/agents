"""Tests for AssemblyAI STT plugin configuration options."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from livekit.agents.stt import SpeechEventType
from livekit.agents.types import NOT_GIVEN


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
