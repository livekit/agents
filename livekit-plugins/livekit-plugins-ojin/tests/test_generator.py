from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest
from fake_stv import FakeSTVClient, make_audio_frame, make_video_frame

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins.ojin.avatar import OjinVideoGenerator, _FrameSink

# Hermetic: driven by a fake Ojin client, no network and no credentials.
pytestmark = pytest.mark.unit


def chunk(ms: int = 40, sample_rate: int = 24000, num_channels: int = 1) -> rtc.AudioFrame:
    samples = int(sample_rate * ms / 1000)
    return rtc.AudioFrame(
        data=b"\x11\x22" * samples * num_channels,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples,
    )


def build() -> tuple[FakeSTVClient, _FrameSink, OjinVideoGenerator]:
    sink = _FrameSink()
    client = FakeSTVClient(output=sink)
    return client, sink, OjinVideoGenerator(client, sink)


# --- turn state machine -----------------------------------------------------


async def test_one_turn_across_many_chunks() -> None:
    client, _, gen = build()

    for _ in range(5):
        await gen.push_audio(chunk())

    assert client.turns == 1
    assert len(client.sent) == 5


async def test_new_turn_after_segment_end() -> None:
    client, _, gen = build()

    await gen.push_audio(chunk())
    await gen.push_audio(AudioSegmentEnd())
    await gen.push_audio(chunk())

    assert client.turns == 2


async def test_segment_end_does_not_touch_the_client() -> None:
    client, _, gen = build()
    await gen.push_audio(chunk())

    await gen.push_audio(AudioSegmentEnd())

    assert len(client.sent) == 1


async def test_input_side_reported_to_the_sink() -> None:
    client, sink, gen = build()

    await gen.push_audio(chunk())
    assert sink._segments.input_closed is False
    assert sink._segments.input_had_audio is True

    await gen.push_audio(AudioSegmentEnd())
    assert sink._segments.input_closed is True


async def test_turn_flag_set_before_the_awaited_start_turn() -> None:
    """start_turn suspends on a real send; concurrent readers must see the turn."""
    client, _, gen = build()
    client.start_turn_gate = asyncio.Event()

    first = asyncio.create_task(gen.push_audio(chunk()))
    await asyncio.sleep(0)
    second = asyncio.create_task(gen.push_audio(chunk()))
    await asyncio.sleep(0)

    client.start_turn_gate.set()
    await asyncio.gather(first, second)

    assert client.turns == 1, "the utterance was split across two turns"


async def test_failed_start_turn_rolls_back_and_does_not_raise() -> None:
    """An escape would kill the runner's _read_audio task permanently."""
    client, _, gen = build()
    client.start_turn_raises = RuntimeError("boom")

    await gen.push_audio(chunk())

    assert gen._turn_started is False
    assert client.sent == []


async def test_send_failure_is_contained() -> None:
    client, _, gen = build()

    async def boom(*args: object) -> None:
        raise RuntimeError("send failed")

    client.send_tts_audio = boom  # type: ignore[assignment]
    await gen.push_audio(chunk())  # must not raise


# --- audio shaping ----------------------------------------------------------


async def test_stereo_is_downmixed_to_mono() -> None:
    client, _, gen = build()
    stereo = np.array([[1000, 2000], [-400, -600]], dtype=np.int16).tobytes()
    frame = rtc.AudioFrame(data=stereo, sample_rate=24000, num_channels=2, samples_per_channel=2)

    await gen.push_audio(frame)

    pcm, rate, channels = client.sent[0]
    assert channels == 1
    assert rate == 24000
    assert np.frombuffer(pcm, dtype=np.int16).tolist() == [1500, -500]


async def test_zero_chunks_do_not_mark_real_audio() -> None:
    """The SDK may discard an all-zero payload, so its echo may never arrive."""
    client, sink, gen = build()
    silent = rtc.AudioFrame(
        data=bytes(1920), sample_rate=24000, num_channels=1, samples_per_channel=960
    )

    await gen.push_audio(silent)

    assert gen._turn_had_real_audio is False
    assert sink._segments.input_had_audio is False


# --- barge-in ---------------------------------------------------------------


async def test_clear_buffer_interrupts_and_mutes() -> None:
    client, sink, gen = build()
    await gen.push_audio(chunk())

    await gen.clear_buffer()

    assert client.interrupts == 1
    assert gen._turn_started is False
    await sink.write_audio(make_audio_frame())
    assert sink.pending == (), "room audio must stop immediately on barge-in"


async def test_failed_interrupt_lifts_the_mute_so_the_surviving_reply_is_heard() -> None:
    client, sink, gen = build()
    client.interrupt_result = False
    await gen.push_audio(chunk())

    await gen.clear_buffer()

    await sink.write_audio(make_audio_frame())
    assert len(sink.pending) == 1, "the reply that was not cancelled must still be heard"
    assert sink._segments.input_closed is True, "the uncancelled turn's input must be retired"


async def test_failed_interrupt_while_idle_lifts_mute_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, sink, gen = build()
    client.interrupt_result = False

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.ojin"):
        await gen.clear_buffer()  # no turn was open

    assert caplog.records == [], "an idle clear is not the uncancellable-turn case"
    await sink.write_audio(make_audio_frame())
    assert len(sink.pending) == 1


async def test_refire_during_an_in_flight_fade_keeps_the_mute() -> None:
    client, sink, gen = build()
    await gen.push_audio(chunk())
    await gen.clear_buffer()  # succeeds, fade now in flight

    client.interrupt_result = False
    await gen.push_audio(chunk())
    await gen.clear_buffer()  # re-fire

    await sink.write_audio(make_audio_frame())
    assert sink.pending == (), "the in-flight fade leaked into the room"


async def test_uncancellable_turn_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    client, _, gen = build()
    client.interrupt_result = False

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.ojin"):
        for _ in range(3):
            await gen.push_audio(chunk())
            await gen.clear_buffer()

    warnings = [r for r in caplog.records if "not" in r.getMessage()]
    assert len(warnings) == 1, "the SDK limitation must be logged once, not per turn"


async def test_raising_interrupt_never_escapes_and_releases_the_guard() -> None:
    """The runner swallows exceptions here and would skip its interrupted report."""
    client, sink, gen = build()
    client.interrupt_raises = RuntimeError("dead transport")
    sink.note_clear_pending()

    await gen.clear_buffer()

    assert sink.clear_pending is False
    await asyncio.wait_for(sink.wait_clear_done(), 1)


async def test_new_turn_waits_for_a_pending_clear() -> None:
    """Otherwise the interrupt lands after start_turn and clears the new buffers."""
    client, sink, gen = build()
    order: list[str] = []

    async def slow_interrupt() -> bool:
        await asyncio.sleep(0.05)
        order.append("interrupt")
        return True

    client.interrupt = slow_interrupt  # type: ignore[assignment]
    sink.note_clear_pending()

    clear = asyncio.create_task(gen.clear_buffer())
    await asyncio.sleep(0)

    async def push() -> None:
        await gen.push_audio(chunk())
        order.append("start_turn")

    await asyncio.gather(clear, push())

    assert order == ["interrupt", "start_turn"]


# --- stream -----------------------------------------------------------------


async def test_aiter_yields_sink_frames_in_order() -> None:
    _, sink, gen = build()
    await sink.write_video(make_video_frame())
    await sink.write_audio(make_audio_frame())

    stream = gen.__aiter__()
    first = await asyncio.wait_for(stream.__anext__(), 1)
    second = await asyncio.wait_for(stream.__anext__(), 1)

    assert isinstance(first, rtc.VideoFrame)
    assert isinstance(second, rtc.AudioFrame)


# --- output format safety ---------------------------------------------------


async def test_input_resampled_to_the_configured_rate() -> None:
    """Ojin echoes back whatever rate we send, and the track's rate is fixed.

    The framework only installs its own resampler on a segment's first frame, so
    after a barge-in a native-rate frame can reach us unresampled.
    """
    client, _, gen = build()
    off_rate = rtc.AudioFrame(
        data=b"\x11\x22" * 640, sample_rate=16000, num_channels=1, samples_per_channel=640
    )

    await gen.push_audio(off_rate)

    assert client.sent, "nothing was sent"
    _, rate, channels = client.sent[0]
    assert (rate, channels) == (24000, 1), "sent at a rate the output track cannot play"


async def test_mismatched_echo_is_dropped_not_forwarded() -> None:
    """A single frame the audio source rejects kills the runner's loop forever."""
    sink = _FrameSink()

    await sink.write_audio(make_audio_frame(sample_rate=16000))

    assert sink.pending == (), "a frame in the wrong format reached the room"
    assert sink.format_mismatches == 1


async def test_stream_survives_a_mismatched_echo() -> None:
    sink = _FrameSink()

    await sink.write_audio(make_audio_frame(sample_rate=16000))
    await sink.write_audio(make_audio_frame())

    assert len(sink.pending) == 1, "the stream did not recover after a bad frame"


async def test_audio_pushed_during_start_turn_survives_the_open() -> None:
    """Opening the input segment must not discard audio reported during the await.

    The opening chunk here is silence, so only the interleaved call reports real
    audio. If the open runs after the await it resets that away, and
    `note_input_segment_end` can no longer arm the render deadline - leaving a
    turn the server never renders with nothing to close it.
    """
    client, sink, gen = build()
    client.start_turn_gate = asyncio.Event()
    silent = rtc.AudioFrame(
        data=bytes(1920), sample_rate=24000, num_channels=1, samples_per_channel=960
    )

    opening = asyncio.create_task(gen.push_audio(silent))
    await asyncio.sleep(0)
    interleaved = asyncio.create_task(gen.push_audio(chunk()))
    await asyncio.sleep(0)
    client.start_turn_gate.set()
    await asyncio.gather(opening, interleaved)

    assert sink._segments.input_had_audio is True, "audio reported during the await was lost"

    sink.note_input_segment_end(had_real_audio=True)
    assert sink._segments.render_deadline_armed, "an unrendered turn has no deadline"
