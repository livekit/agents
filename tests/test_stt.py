import asyncio
import os
import wave
from typing import List

import pytest
from livekit import agents, rtc
from livekit.plugins import deepgram, google, openai, silero
from utils import compare_word_counts

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.wav")
TEST_AUDIO_FILEPATH_2 = os.path.join(os.path.dirname(__file__), "long.mp3")
TEST_AUDIO_TRANSCRIPT = "the people who are crazy enough to think they can change the world are the ones who do"
TEST_AUDIO_TRANSCRIPT_2 = "It could not have been ten seconds, and yet it seemed a long time that their hands were clasped together.\
He had time to learn every detail of her hand.\
He explored the long fingers, the shapely nails, the work-hardened palm with its row of callouses, the smooth flesh under the wrist.\
Merely from feeling it he would have known it by sight.\
In the same instant it occurred to him that he did not know what colour the girl's eyes were.\
They were probably brown, but people with dark hair sometimes had blue eyes.\
To turn his head and look at her would have been inconceivable folly.\
With hands locked together, invisible among the press of bodies,\
they stared steadily in front of them, and instead of the eyes of the girl, the eyes of the aged prisoner gazed mournfully at Winston out of nests of hair."


def read_mp3_file(filename: str) -> List[rtc.AudioFrame]:
    async def decode():
        mp3 = agents.codecs.Mp3StreamDecoder()
        with open(filename, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                mp3.push_chunk(chunk)

        mp3.close()

        frames: List[rtc.AudioFrame] = []
        async for data in mp3:
            frames.append(data)
        return agents.utils.merge_frames(frames)

    return asyncio.get_event_loop().run_until_complete(decode())


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


input = {
    "short": {
        "audio": read_wav_file(TEST_AUDIO_FILEPATH),
        "transcript": TEST_AUDIO_TRANSCRIPT,
    },
    "long": {
        "audio": read_mp3_file(TEST_AUDIO_FILEPATH_2),
        "transcript": TEST_AUDIO_TRANSCRIPT_2,
    },
}

STTFactoryStream = {
    "deepgram": lambda: deepgram.STT(min_silence_duration=100),
    "google": google.STT,
    "openai": lambda: agents.stt.StreamAdapter(
        openai.STT(),
        silero.VAD().stream(),
    ),
}

STTFactoryRecognize = {
    "deepgram": deepgram.STT,
    "google": google.STT,
    "openai": openai.STT,
}

cases = {
    "recognize": [
        ("deepgram", "short"),
        ("deepgram", "long"),
        ("google", "short"),  # skip long test for google because it google cuts it off
        ("openai", "short"),
        ("openai", "long"),
    ],
    "stream": [
        ("deepgram", "short"),
        ("deepgram", "long"),
        ("google", "short"),
        ("google", "long"),
        ("openai", "short"),
        ("openai", "long"),
    ],
}


@pytest.mark.parametrize(
    "provider, input_key",
    (pytest.param(p, i) for (p, i) in cases["recognize"]),
)
async def test_recognize(provider: str, input_key: str):
    frame = input[input_key]["audio"]
    expected = input[input_key]["transcript"]
    stt = STTFactoryRecognize[provider]()
    event = await stt.recognize(buffer=frame)
    text = event.alternatives[0].text
    assert compare_word_counts(text, expected) > 0.9
    assert event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT


@pytest.mark.parametrize(
    "provider, input_key",
    (pytest.param(p, i) for (p, i) in cases["stream"]),
)
async def test_stream(provider: str, input_key: str):
    frame = input[input_key]["audio"]
    expected = input[input_key]["transcript"]

    # divide data into chunks of 10ms
    chunk_size = frame.sample_rate // 100
    frames = []
    for i in range(0, len(frame.data), chunk_size):
        data = frame.data[i : i + chunk_size]
        frames.append(
            rtc.AudioFrame(
                data=data.tobytes() + b"\0\0" * (chunk_size - len(data)),
                num_channels=frame.num_channels,
                samples_per_channel=chunk_size,
                sample_rate=frame.sample_rate,
            )
        )

    stt = STTFactoryStream[provider]()
    stream = stt.stream()

    async def stream_input():
        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(0.01)  # one frame is 10ms

        await stream.aclose()

    async def stream_output():
        # STT Should start with a START_OF_SPEECH event
        start_event = await anext(stream)
        assert start_event.type == agents.stt.SpeechEventType.START_OF_SPEECH

        total_text = ""
        async for event in stream:
            if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                total_text = " ".join([total_text, event.alternatives[0].text])

        assert compare_word_counts(total_text, expected) > 0.8

    await asyncio.gather(stream_input(), stream_output())
