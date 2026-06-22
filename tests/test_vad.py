import asyncio

import pytest

from livekit.agents import vad
from livekit.agents.inference import VAD as InferenceVAD

from . import utils

# loads the silero ONNX model at import
pytestmark = pytest.mark.plugin("silero")

SAMPLE_RATES = [16000, 44100]  # test multiple input sample rates


VAD = InferenceVAD(
    model="silero",
    min_speech_duration=0.5,
    min_silence_duration=0.75,
)


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_chunks_vad(sample_rate) -> None:
    frames, *_ = await utils.make_test_speech(chunk_duration_ms=10, sample_rate=sample_rate)
    assert len(frames) > 1, "frames aren't chunked"

    stream = VAD.stream()

    for frame in frames:
        stream.push_frame(frame)

    stream.end_input()

    start_of_speech_i = 0
    end_of_speech_i = 0

    inference_frames = []

    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            with open(
                f"test_vad.{sample_rate}.start_of_speech_frames_{start_of_speech_i}.wav",
                "wb",
            ) as f:
                f.write(utils.make_wav_file(ev.frames))

            start_of_speech_i += 1

        if ev.type == vad.VADEventType.INFERENCE_DONE:
            inference_frames.extend(ev.frames)

        if ev.type == vad.VADEventType.END_OF_SPEECH:
            with open(
                f"test_vad.{sample_rate}.end_of_speech_frames_{end_of_speech_i}.wav",
                "wb",
            ) as f:
                f.write(utils.make_wav_file(ev.frames))

            end_of_speech_i += 1

    assert start_of_speech_i > 0, "no start of speech detected"
    assert start_of_speech_i == end_of_speech_i, "start and end of speech mismatch"

    with open(f"test_vad.{sample_rate}.inference_frames.wav", "wb") as f:
        f.write(utils.make_wav_file(inference_frames))


async def _drain_speech_segment(
    stream: vad.VADStream, frames: list, *, timeout: float = 30.0
) -> tuple[vad.VADEvent, vad.VADEvent]:
    """Push *frames* until both START_OF_SPEECH and END_OF_SPEECH have fired."""

    done = asyncio.Event()

    async def _pump() -> None:
        for frame in frames:
            if done.is_set():
                return
            stream.push_frame(frame)
            await asyncio.sleep(0)

    async def _consume() -> tuple[vad.VADEvent, vad.VADEvent]:
        sos_event: vad.VADEvent | None = None
        async for ev in stream:
            if ev.type == vad.VADEventType.START_OF_SPEECH and sos_event is None:
                sos_event = ev
            elif ev.type == vad.VADEventType.END_OF_SPEECH and sos_event is not None:
                return sos_event, ev

        raise AssertionError("stream ended before END_OF_SPEECH")

    pump_task = asyncio.create_task(_pump())
    try:
        return await asyncio.wait_for(_consume(), timeout=timeout)
    finally:
        done.set()
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass


async def test_reset_recovers_full_speech_segment() -> None:
    """Real speech audio should still produce a complete SOS + EOS cycle after reset."""

    frames, *_ = await utils.make_test_speech(chunk_duration_ms=10, sample_rate=16000)
    assert len(frames) > 1, "frames aren't chunked"

    stream = VAD.stream()
    try:
        first_sos, first_eos = await _drain_speech_segment(stream, frames)
        assert first_sos.type == vad.VADEventType.START_OF_SPEECH
        assert first_eos.type == vad.VADEventType.END_OF_SPEECH

        stream.flush()

        second_sos, second_eos = await _drain_speech_segment(stream, frames)
        assert second_sos.type == vad.VADEventType.START_OF_SPEECH
        assert second_eos.type == vad.VADEventType.END_OF_SPEECH
    finally:
        await stream.aclose()


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_file_vad(sample_rate):
    frames, *_ = await utils.make_test_speech(sample_rate=sample_rate)
    assert len(frames) == 1, "one frame should be the whole audio"

    stream = VAD.stream()

    for frame in frames:
        stream.push_frame(frame)

    stream.end_input()

    start_of_speech_i = 0
    end_of_speech_i = 0
    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            start_of_speech_i += 1

        if ev.type == vad.VADEventType.END_OF_SPEECH:
            end_of_speech_i += 1

    assert start_of_speech_i > 0, "no start of speech detected"
    assert start_of_speech_i == end_of_speech_i, "start and end of speech mismatch"
