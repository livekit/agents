from __future__ import annotations

import asyncio

import pytest
from fake_stv import make_audio_frame, make_video_frame
from ojin.stv import FrameType

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins.ojin.avatar import _FrameSink

# Hermetic: driven by a fake Ojin client, no network and no credentials.
pytestmark = pytest.mark.unit


def drain(sink: _FrameSink) -> list[object]:
    items = list(sink.pending)
    sink._deque.clear()
    sink._video_count = 0
    return items


def counts(items: list[object]) -> tuple[int, int, int]:
    return (
        sum(isinstance(i, rtc.AudioFrame) for i in items),
        sum(isinstance(i, rtc.VideoFrame) for i in items),
        sum(isinstance(i, AudioSegmentEnd) for i in items),
    )


# --- silence gate -----------------------------------------------------------


async def test_synthesized_silence_never_enqueued() -> None:
    sink = _FrameSink()

    for _ in range(5):
        await sink.write_audio(make_audio_frame(silent=True))
    # pre-turn fill is 16 kHz-shaped, and must be dropped just the same
    await sink.write_audio(make_audio_frame(sample_rate=16000, silent=True))

    assert counts(drain(sink))[0] == 0


async def test_real_audio_forwarded() -> None:
    sink = _FrameSink()

    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 1


async def test_held_tick_with_real_audio_loses_no_audio() -> None:
    """A held tick carries the default (idle) frame type but real audio."""
    sink = _FrameSink()

    for _ in range(3):
        await sink.write_video(make_video_frame(frame_type=FrameType.IDLE, fresh=False))
        await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 3


# --- segment protocol: both sides must be done ------------------------------


async def test_marker_when_stop_edge_follows_input_close() -> None:
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())

    sink.note_input_segment_end()
    assert counts(drain(sink))[2] == 0  # output still draining

    sink.on_bot_stopped_speaking()
    assert counts(drain(sink))[2] == 1


async def test_marker_when_input_close_follows_stop_edge() -> None:
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())

    sink.on_bot_stopped_speaking()  # drained first
    assert counts(drain(sink))[2] == 0  # input still open

    sink.note_input_segment_end()
    assert counts(drain(sink))[2] == 1


async def test_underrun_stop_edge_emits_nothing() -> None:
    """Slow TTS drains the buffer mid-utterance; that is not a turn end."""
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())

    sink.on_bot_stopped_speaking()
    sink.note_input_audio()  # refill arrives
    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[2] == 0

    sink.on_bot_stopped_speaking()
    sink.note_input_segment_end()
    assert counts(drain(sink))[2] == 1


async def test_final_chunk_and_close_during_underrun_defers_marker() -> None:
    """The final chunk and the input close arrive back to back, mid-underrun.

    The refilled tail has not played yet, so no marker may be emitted until it does.
    """
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())
    sink.on_bot_stopped_speaking()  # drained, input still open

    sink.note_input_audio()  # final chunk pushed (not yet echoed)
    sink.note_input_segment_end()

    assert counts(drain(sink))[2] == 0, "marker emitted with the tail unplayed"

    await sink.write_audio(make_audio_frame())  # tail plays
    sink.on_bot_stopped_speaking()
    assert counts(drain(sink))[2] == 1


async def test_all_zero_utterance_emits_marker_at_close() -> None:
    """Nothing will ever echo back, so the marker must come from the input side."""
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame(silent=True))

    sink.note_input_segment_end(had_real_audio=False)

    assert counts(drain(sink))[2] == 1


async def test_stop_edge_while_idle_emits_nothing() -> None:
    sink = _FrameSink()

    sink.on_bot_stopped_speaking()

    assert counts(drain(sink))[2] == 0


# --- barge-in: purge and mute ----------------------------------------------


async def test_begin_clear_purges_audio_and_markers_keeps_video() -> None:
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_video(make_video_frame())
    await sink.write_audio(make_audio_frame())
    sink.on_bot_stopped_speaking()
    sink.note_input_segment_end()

    sink.begin_clear()

    audio, video, markers = counts(drain(sink))
    assert (audio, markers) == (0, 0)
    assert video == 1


async def test_audio_dropped_while_muting() -> None:
    sink = _FrameSink()
    sink.begin_clear()

    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 0


async def test_stop_edge_one_tick_after_clear_does_not_unmute() -> None:
    """The stop edge fires one tick after interrupt(), while the fade still plays."""
    sink = _FrameSink()
    sink.begin_clear()

    sink.on_bot_stopped_speaking()
    for _ in range(18):  # ~0.75 s of fade chunks, all real audio
        await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 0, "audible fade leaked into the room"


async def test_start_of_speech_unmutes_before_that_ticks_audio() -> None:
    """Video precedes audio within a tick, so the new turn's first chunk survives."""
    sink = _FrameSink()
    sink.begin_clear()

    await sink.write_video(make_video_frame(frame_type=FrameType.START_OF_SPEECH))
    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 1


async def test_mute_timer_expiry_unmutes() -> None:
    """The fade window begin_clear() opens must close on its own.

    START_OF_SPEECH normally lifts the mute; when the replacement turn never
    arrives, only this timer stops the sink muting for the rest of the session.
    """
    sink = _FrameSink(fade_s=0.0)
    sink.begin_clear()

    # begin_clear() arms fade_s plus a guard band; sleeping past the whole
    # window is what proves the window is finite rather than a hand-set value.
    await asyncio.sleep(0.3)
    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 1


async def test_abort_clear_lifts_mute() -> None:
    """A failed interrupt() means no fade is coming; the mute is pure harm."""
    sink = _FrameSink()
    sink.begin_clear()

    sink.abort_clear()
    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 1


# --- geometry invariant -----------------------------------------------------


async def test_geometry_mismatch_dropped_stream_continues() -> None:
    sink = _FrameSink()
    await sink.write_video(make_video_frame(1024, 1024))

    await sink.write_video(make_video_frame(736, 1216))
    await sink.write_video(make_video_frame(1024, 1024))

    assert counts(drain(sink))[1] == 2
    assert sink.geometry_mismatches == 1


# --- queue policy -----------------------------------------------------------


async def test_audio_never_dropped_video_capped() -> None:
    sink = _FrameSink(video_queue_size=4)

    for _ in range(50):
        await sink.write_video(make_video_frame(2, 2))
        await sink.write_audio(make_audio_frame())

    audio, video, _ = counts(drain(sink))
    assert audio == 50
    assert video == 4
    assert sink.dropped_video_frames == 46


async def test_video_overflow_drops_oldest() -> None:
    sink = _FrameSink(video_queue_size=2)
    await sink.write_video(make_video_frame(2, 2, pts=1))
    await sink.write_video(make_video_frame(2, 2, pts=2))

    await sink.write_video(make_video_frame(2, 2, pts=3))

    kept = [i for i in drain(sink) if isinstance(i, rtc.VideoFrame)]
    assert len(kept) == 2


async def test_next_frame_preserves_order() -> None:
    sink = _FrameSink()
    sink.note_input_segment_open()
    await sink.write_video(make_video_frame())
    await sink.write_audio(make_audio_frame())
    sink.note_input_segment_end()
    sink.on_bot_stopped_speaking()

    kinds = [type(await sink.next_frame()) for _ in range(3)]

    assert kinds == [rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]


async def test_next_frame_waits_for_a_frame() -> None:
    sink = _FrameSink()
    task = asyncio.create_task(sink.next_frame())
    await asyncio.sleep(0)
    assert not task.done()

    await sink.write_audio(make_audio_frame())

    assert isinstance(await asyncio.wait_for(task, 1), rtc.AudioFrame)


# --- liveness ---------------------------------------------------------------


async def test_only_fresh_server_frames_advance_liveness() -> None:
    """Synthesized writes continue forever on a dead transport; they prove nothing."""
    sink = _FrameSink()
    await sink.write_video(make_video_frame(fresh=True))
    marker = sink.last_server_frame_time

    await asyncio.sleep(0.01)
    for _ in range(5):
        await sink.write_video(make_video_frame(fresh=False))
        await sink.write_audio(make_audio_frame(silent=True))

    assert sink.last_server_frame_time == marker

    await sink.write_video(make_video_frame(fresh=True))
    assert sink.last_server_frame_time > marker


async def test_first_video_frame_is_recorded_and_forwarded() -> None:
    sink = _FrameSink()

    await sink.write_video(make_video_frame(736, 1216))

    first = await asyncio.wait_for(sink.wait_for_first_video_frame(1), 1)
    assert (first.width, first.height) == (736, 1216)
    assert counts(drain(sink))[1] == 1, "the discovery frame must still reach the runner"


async def test_first_video_frame_times_out() -> None:
    sink = _FrameSink()

    with pytest.raises(asyncio.TimeoutError):
        await sink.wait_for_first_video_frame(0.01)


# --- turn-render liveness ---------------------------------------------------


async def test_render_deadline_armed_when_a_turn_never_renders() -> None:
    sink = _FrameSink(turn_render_timeout=0.01)
    sink.note_input_segment_open()
    sink.note_input_audio()

    sink.note_input_segment_end()

    await asyncio.sleep(0.02)
    assert sink.render_deadline_expired()


async def test_render_deadline_disarmed_once_output_opens() -> None:
    sink = _FrameSink(turn_render_timeout=0.01)
    sink.note_input_segment_open()
    sink.note_input_audio()
    await sink.write_audio(make_audio_frame())  # the turn is rendering
    sink.note_input_segment_end()

    await asyncio.sleep(0.02)
    assert not sink.render_deadline_expired()


async def test_force_segment_end_unwedges_a_stalled_turn() -> None:
    sink = _FrameSink(turn_render_timeout=0.01)
    sink.note_input_segment_open()
    sink.note_input_audio()
    sink.note_input_segment_end()

    sink.force_segment_end()

    assert counts(drain(sink))[2] == 1
    assert not sink.render_deadline_expired()


# --- late render after a forced end -----------------------------------------


def _stalled_turn() -> _FrameSink:
    """A turn fed, never rendered, and forced closed by the render deadline."""
    sink = _FrameSink(turn_render_timeout=0.0)
    sink.note_input_segment_open()
    sink.note_input_audio()
    sink.note_input_segment_end()
    assert sink.render_deadline_expired()
    sink.force_segment_end()
    drain(sink)  # the marker that let the session proceed
    return sink


async def test_late_render_after_a_forced_end_is_not_played() -> None:
    """The session already moved on; this audio would speak over the next turn."""
    sink = _stalled_turn()

    await sink.write_audio(make_audio_frame())

    assert counts(drain(sink))[0] == 0


async def test_stale_stop_edge_after_a_forced_end_ends_nothing() -> None:
    """The retired turn's stop edge must not close whatever came after it."""
    sink = _stalled_turn()

    await sink.write_audio(make_audio_frame())
    sink.on_bot_stopped_speaking()

    assert counts(drain(sink))[2] == 0


async def test_the_next_turn_plays_normally_after_a_forced_end() -> None:
    """Dropping late output must not outlive the turn it belonged to."""
    sink = _stalled_turn()

    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())
    sink.note_input_segment_end()
    sink.on_bot_stopped_speaking()

    audio, _, markers = counts(drain(sink))
    assert (audio, markers) == (1, 1)
