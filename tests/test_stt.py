import asyncio
import wave
import os
from typing import List
from livekit import rtc, agents
from livekit.plugins import deepgram, google, openai, silero
from difflib import SequenceMatcher

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.wav")
TEST_AUDIO_FILEPATH_2 = os.path.join(os.path.dirname(__file__), "long-stt-irl.mp3")
TEST_AUDIO_TRANSCRIPT = "the people who are crazy enough to think they can change the world are the ones who do"
TEST_AUDIO_TRANSCRIPT_2 = "My girlfriend is asleep so I can't talk loud, but \
that's probably pretty good for this test. This is a long test for \
speech-to-text, it has some pauses in it, it might have some background noise \
in it, we'll see"


async def read_mp3_file(filename: str) -> List[rtc.AudioFrame]:
    mp3 = agents.codecs.Mp3StreamDecoder()
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            mp3.push_chunk(chunk)

    mp3.close()

    frames: List[rtc.AudioFrame] = []
    async for data in mp3:
        frames.append(data)

    return agents.utils.merge_frames(frames)


def read_wav_file(filename: str) -> rtc.AudioFrame:
    with wave.open(filename, "rb") as file:
        frames = file.getnframes()

        if file.getsampwidth() != 2:
            raise ValueError("Require 16-bit WAV files")

        return rtc.AudioFrame(
            data=file.readframes(frames),
            num_channels=file.getnchannels(),
            samples_per_channel=frames // file.getnchannels(),
            sample_rate=file.getframerate(),
        )


async def test_recognize():
    # stts = [deepgram.STT(), google.STT(), openai.STT()]
    stts = [deepgram.STT(), openai.STT()]
    inputs = [
        (read_wav_file(TEST_AUDIO_FILEPATH), TEST_AUDIO_TRANSCRIPT),
        (await read_mp3_file(TEST_AUDIO_FILEPATH_2), TEST_AUDIO_TRANSCRIPT_2),
    ]

    async def recognize(stt: agents.stt.STT, frame: rtc.AudioFrame, expected: str):
        event = await stt.recognize(buffer=frame)
        text = event.alternatives[0].text
        assert SequenceMatcher(None, text, expected).ratio() > 0.9
        assert event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT

    async with asyncio.TaskGroup() as group:
        for input, expected in inputs:
            for stt in stts:
                group.create_task(recognize(stt, input, expected))


# async def test_stream():
#     silero_vad = silero.VAD()
#     stts = [
#         deepgram.STT(min_silence_duration=1000),
#         # google.STT(),
#         agents.stt.StreamAdapter(openai.STT(), silero_vad.stream()),
#     ]
#     frame = read_wav_file(TEST_AUDIO_FILEPATH)

#     # divide data into chunks of 10ms
#     chunk_size = frame.sample_rate // 100
#     frames = []
#     for i in range(0, len(frame.data), chunk_size):
#         data = frame.data[i : i + chunk_size]
#         frames.append(
#             rtc.AudioFrame(
#                 data=data.tobytes() + b"\0\0" * (chunk_size - len(data)),
#                 num_channels=frame.num_channels,
#                 samples_per_channel=chunk_size,
#                 sample_rate=frame.sample_rate,
#             )
#         )

#     async def stream(stt: agents.stt.STT):
#         stream = stt.stream()
#         for frame in frames:
#             stream.push_frame(frame)
#             await asyncio.sleep(0.01)  # one frame is 10ms

#         # STT Should start with a START_OF_SPEECH event
#         start_event = await anext(stream)
#         assert start_event.type == agents.stt.SpeechEventType.START_OF_SPEECH

#         async for event in stream:
#             if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
#                 text = event.alternatives[0].text
#                 assert SequenceMatcher(None, text, TEST_AUDIO_TRANSCRIPT).ratio() > 0.8

#                 await stream.aclose()
#                 break

#     async with asyncio.TaskGroup() as group:
#         for stt in stts:
#             group.create_task(stream(stt))
