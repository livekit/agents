from livekit.agents import vad
from livekit.plugins import silero

from . import utils

VAD = silero.VAD.load(
    min_speech_duration=0.5, min_silence_duration=0.5, padding_duration=1.0
)


async def test_chunks_vad() -> None:
    frames, transcript = utils.make_test_audio(chunk_duration_ms=10)
    assert len(frames) > 1, "frames aren't chunked"

    stream = VAD.stream()

    for frame in frames:
        stream.push_frame(frame)

    stream.end_input()

    start_of_speech_i = 0
    end_of_speech_i = 0
    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            with open(
                f"test_vad.start_of_speech_frames_{start_of_speech_i}.wav", "wb"
            ) as f:
                f.write(utils.make_wav_file(ev.frames))

            start_of_speech_i += 1

        if ev.type == vad.VADEventType.END_OF_SPEECH:
            with open(
                f"test_vad.end_of_speech_frames_{end_of_speech_i}.wav", "wb"
            ) as f:
                f.write(utils.make_wav_file(ev.frames))

            end_of_speech_i += 1

    assert start_of_speech_i > 0, "no start of speech detected"
    assert start_of_speech_i == end_of_speech_i, "start and end of speech mismatch"


async def test_file_vad():
    frames, transcript = utils.make_test_audio()
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
