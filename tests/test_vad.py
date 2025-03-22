import pytest

from livekit.agents import vad
from livekit.plugins import silero

from . import utils

SAMPLE_RATES = [16000, 44100]  # test multiple input sample rates


VAD = silero.VAD.load(
    min_speech_duration=0.5,
    min_silence_duration=0.6,
)


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_chunks_vad(sample_rate) -> None:
    frames, _ = await utils.make_test_speech(
        chunk_duration_ms=10, sample_rate=sample_rate
    )
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

    with open("test_vad.{sample_rate}.inference_frames.wav", "wb") as f:
        f.write(utils.make_wav_file(inference_frames))


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_file_vad(sample_rate):
    frames, _ = await utils.make_test_speech(sample_rate=sample_rate)
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
