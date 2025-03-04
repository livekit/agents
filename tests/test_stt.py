"""
Do speech recognition on a long audio file and compare the result with the expected transcript
"""

import asyncio
import time
from typing import Callable

import pytest
from livekit import agents
from livekit.agents import stt
from livekit.plugins import (
    assemblyai,
    aws,
    azure,
    deepgram,
    fal,
    openai,
    silero,
    speechmatics,
)

from .utils import make_test_speech, wer

SAMPLE_RATES = [24000, 44100]  # test multiple input sample rates
WER_THRESHOLD = 0.25
RECOGNIZE_STT: list[Callable[[], stt.STT]] = [
    pytest.param(lambda: deepgram.STT(), id="deepgram"),
    # pytest.param(lambda: google.STT(), id="google"),
    # pytest.param(
    #     lambda: google.STT(
    #         languages=["en-AU"],
    #         model="chirp_2",
    #         spoken_punctuation=False,
    #         location="us-central1",
    #     ),
    #     id="google.chirp_2",
    # ),
    pytest.param(lambda: openai.STT(), id="openai"),
    pytest.param(lambda: fal.WizperSTT(), id="fal"),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt_factory", RECOGNIZE_STT)
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_recognize(stt_factory, sample_rate):
    async with stt_factory() as stt:
        frames, transcript = await make_test_speech(sample_rate=sample_rate)

        start_time = time.time()
        event = await stt.recognize(buffer=frames)
        text = event.alternatives[0].text
        dt = time.time() - start_time

        print(f"WER: {wer(text, transcript)} for {stt} in {dt:.2f}s")
        assert wer(text, transcript) <= WER_THRESHOLD
        assert event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT


STREAM_VAD = silero.VAD.load(min_silence_duration=0.75)
STREAM_STT: list[Callable[[], stt.STT]] = [
    pytest.param(lambda: aws.STT(), id="aws"),
    pytest.param(lambda: assemblyai.STT(), id="assemblyai"),
    pytest.param(lambda: deepgram.STT(), id="deepgram"),
    # pytest.param(lambda: google.STT(), id="google"),
    pytest.param(
        lambda: agents.stt.StreamAdapter(stt=openai.STT(), vad=STREAM_VAD),
        id="openai.stream",
    ),
    pytest.param(
        lambda: agents.stt.StreamAdapter(stt=openai.STT.with_groq(), vad=STREAM_VAD),
        id="openai.with_groq.stream",
    ),
    # pytest.param(
    #     lambda: google.STT(
    #         languages=["en-AU"],
    #         model="chirp_2",
    #         spoken_punctuation=False,
    #         location="us-central1",
    #     ),
    #     id="google.chirp_2",
    # ),
    pytest.param(lambda: azure.STT(), id="azure"),
    pytest.param(lambda: speechmatics.STT(), id="speechmatics"),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt_factory", STREAM_STT)
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_stream(stt_factory, sample_rate):
    stt = stt_factory()
    frames, transcript = await make_test_speech(
        chunk_duration_ms=10, sample_rate=sample_rate
    )

    stream = stt.stream()

    async def _stream_input():
        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(0.005)

        stream.end_input()

    async def _stream_output():
        text = ""
        # make sure the events are sent in the right order
        recv_start, recv_end = False, True
        start_time = time.time()

        async for event in stream:
            if event.type == agents.stt.SpeechEventType.START_OF_SPEECH:
                assert recv_end, "START_OF_SPEECH recv but no END_OF_SPEECH has been sent before"
                assert not recv_start
                recv_end = False
                recv_start = True
                continue

            if event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                if text != "":
                    text += " "
                text += event.alternatives[0].text
                # ensure STT is tagging languages correctly
                language = event.alternatives[0].language
                assert language is not None
                assert language.lower().startswith("en")

            if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                recv_start = False
                recv_end = True

        dt = time.time() - start_time
        print(f"WER: {wer(text, transcript)} for streamed {stt} in {dt:.2f}s")
        assert wer(text, transcript) <= WER_THRESHOLD

    await asyncio.wait_for(asyncio.gather(_stream_input(), _stream_output()), timeout=120)
    await stream.aclose()
