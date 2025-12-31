"""
Do speech recognition on a long audio file and compare the result with the expected transcript
"""

import asyncio
import time
from typing import Callable

import pytest
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import inference, stt, utils
from livekit.agents.log import logger
from livekit.agents.stt.stt import STT, SpeechData, SpeechEvent
from livekit.plugins import (
    assemblyai,
    aws,
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    fal,
    fireworksai,
    gladia,
    google,
    gradium,
    mistralai,
    nvidia,
    openai,
    rtzr,
    sarvam,
    soniox,
    speechmatics,
)

from .utils import make_test_speech, wer

SAMPLE_RATES = [24000, 44100]  # test multiple input sample rates
WER_THRESHOLD = 0.25


def parameter_factory(plugin):
    return pytest.param(lambda: plugin.STT(), id=plugin.__name__)


STTs: list[Callable[[], stt.STT]] = [
    parameter_factory(plugin)
    for plugin in [
        deepgram,
        assemblyai,
        speechmatics,
        elevenlabs,
        fireworksai,
        gladia,
        rtzr,
        fal,
        mistralai,
        nvidia,
        openai,
        cartesia,
        gradium,
        soniox,
        google,
        inference,
        azure,
        aws,
        sarvam,
        # TODO: only Business account allowed outside South Korea
        # clova,
        # TODO: https://github.com/spi-tch/spitch-python/issues/162
        # spitch,
    ]
] + [pytest.param(lambda: deepgram.STTv2(), id="livekit.plugins.deepgram.STTv2")]


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt_factory", STTs)
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_recognize(stt_factory: Callable[[], stt.STT], sample_rate: int, request):
    plugin_id = request.node.callspec.id.split("-")[0]  # e.g., "livekit.plugins.deepgram"
    try:
        stt_instance = stt_factory()
    except ValueError as e:
        pytest.skip(f"{plugin_id}: {e}")

    async with stt_instance as stt:
        label = f"{stt.model}@{stt.provider}"
        if not stt.capabilities.batch_recognition:
            pytest.skip(f"{label} does not support batch recognition")

        frames, transcript, duration = await make_test_speech(sample_rate=sample_rate)

        start_time = time.time()
        # Sarvam only supports <30s audio
        if stt.provider in {"Sarvam"} and duration > 30:
            frames, *_ = await make_test_speech(sample_rate=sample_rate, chunk_duration_ms=5 * 1000)
            event1 = await stt.recognize(buffer=frames[: len(frames) // 2])
            event2 = await stt.recognize(buffer=frames[len(frames) // 2 :])
            event = SpeechEvent(
                type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=event1.request_id,
                alternatives=[
                    SpeechData(
                        text=event1.alternatives[0].text + " " + event2.alternatives[0].text,
                        language=event1.alternatives[0].language,
                    )
                ],
            )
        else:
            event = await stt.recognize(buffer=frames)
        text = event.alternatives[0].text
        dt = time.time() - start_time

        print(f"WER: {wer(text, transcript)} for {stt} in {dt:.2f}s")
        # Relaxed WER threshold for some providers
        if stt.provider in {
            "Gladia",
        }:
            assert len(text) > 0 and wer(text, transcript) <= 1.0
        else:
            assert wer(text, transcript) <= WER_THRESHOLD
        assert event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt_factory", STTs)
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_stream(stt_factory: Callable[[], STT], sample_rate: int, request):
    plugin_id = request.node.callspec.id.split("-")[0]  # e.g., "livekit.plugins.deepgram"
    try:
        stt: STT = stt_factory()
    except ValueError as e:
        pytest.skip(f"{plugin_id}: {e}")

    label = f"{stt.model}@{stt.provider}"
    if not stt.capabilities.streaming:
        pytest.skip(f"{label} does not support streaming")

    frames, transcript, duration = await make_test_speech(
        chunk_duration_ms=10, sample_rate=sample_rate
    )

    stream = stt.stream()
    closing = False

    @utils.log_exceptions(logger=logger)
    async def _stream_input():
        nonlocal closing
        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(0.005)

        stream.end_input()
        closing = True

    @utils.log_exceptions(logger=logger)
    async def _stream_output():
        nonlocal closing
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
                if stt.provider not in {"FireworksAI", "RTZR", "livekit"}:
                    assert language is not None
                    assert language.lower().startswith("en")

            if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                recv_start = False
                recv_end = True
                # wait for the closing to be set
                await asyncio.sleep(1)
                if closing:
                    break

        dt = time.time() - start_time
        print(f"WER: {wer(text, transcript)} for streamed {stt} in {dt:.2f}s")
        # RTZR defaults to Korean
        if stt.provider in {
            "RTZR",
            "Deepgram",
            "Sarvam",
        }:
            assert len(text) > 0 and wer(text, transcript) <= 1.0
        else:
            assert wer(text, transcript) <= WER_THRESHOLD

    await asyncio.wait_for(asyncio.gather(_stream_input(), _stream_output()), timeout=120)
    await stream.aclose()
