"""
Do speech recognition on a long audio file and compare the result with the expected transcript
"""

import asyncio
import os
import pathlib
import time

import pytest
from livekit import agents, rtc
from livekit.plugins import azure, deepgram, google, openai, silero

from .utils import wer

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "long.mp3")
TEST_AUDIO_TRANSCRIPT = pathlib.Path(
    os.path.dirname(__file__), "long_transcript.txt"
).read_text()


def read_mp3_file(filename: str) -> rtc.AudioFrame:
    mp3 = agents.utils.codecs.Mp3StreamDecoder()
    frames: list[rtc.AudioFrame] = []
    with open(filename, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            try:
                frames.extend(mp3.decode_chunk(chunk))
            except Exception as e:
                print(f"error decoding mp3 chunk: {e}", chunk)

    return agents.utils.merge_frames(frames)


RECOGNIZE_STT = [deepgram.STT(), google.STT(), openai.STT()]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt", RECOGNIZE_STT)
async def test_recognize(stt: agents.stt.STT):
    frame = read_mp3_file(TEST_AUDIO_FILEPATH)

    start_time = time.time()
    event = await stt.recognize(buffer=frame)
    text = event.alternatives[0].text
    dt = time.time() - start_time

    print(f"WER: {wer(text, TEST_AUDIO_TRANSCRIPT)} for {stt} in {dt:.2f}s")
    assert wer(text, TEST_AUDIO_TRANSCRIPT) <= 0.2
    assert event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT


STREAM_VAD = silero.VAD()
STREAM_STT = [
    deepgram.STT(),
    google.STT(),
    agents.stt.StreamAdapter(stt=openai.STT(), vad=STREAM_VAD),
    azure.STT(),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt", STREAM_STT)
async def test_stream(stt: agents.stt.STT):
    # divide data into chunks of 10ms
    frame = read_mp3_file(TEST_AUDIO_FILEPATH)
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

    stream = stt.stream()

    async def _stream_input():
        for frame in frames:
            stream.push_frame(frame)
            # audio are split in 10ms chunks but the whole file is 40s,
            # but we still wait less to make the tests faster
            await asyncio.sleep(0.001)

        await stream.aclose()

    async def _stream_output():
        text = ""
        # make sure the events are sent in the right order
        recv_start, recv_end = False, True
        start_time = time.time()

        async for event in stream:
            if event.type == agents.stt.SpeechEventType.START_OF_SPEECH:
                assert (
                    recv_end
                ), "START_OF_SPEECH recv but no END_OF_SPEECH has been sent before"
                assert not recv_start
                recv_end = False
                recv_start = True
                continue

            if event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                text += event.alternatives[0].text

            if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                recv_start = False
                recv_end = True

        dt = time.time() - start_time
        print(
            f"WER: {wer(text, TEST_AUDIO_TRANSCRIPT)} for streamed {stt} in {dt:.2f}s"
        )
        assert wer(text, TEST_AUDIO_TRANSCRIPT) <= 0.2

    await asyncio.wait_for(
        asyncio.gather(_stream_input(), _stream_output()), timeout=60
    )
