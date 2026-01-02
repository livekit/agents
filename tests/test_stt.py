"""
Do speech recognition on a long audio file and compare the result with the expected transcript
"""

import asyncio
import time
from typing import Callable

import pytest
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import inference, stt
from livekit.agents.stt.stt import STT, RecognizeStream, SpeechData, SpeechEvent
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
    sarvam,
    soniox,
    speechmatics,
)

from .utils import make_test_speech, wer

SAMPLE_RATE = 24000
WER_THRESHOLD = 0.25
MAX_RETRIES = 2


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
        # rtzr,
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
async def test_recognize(stt_factory: Callable[[], stt.STT], request):
    plugin_id = request.node.callspec.id.split("-")[0]  # e.g., "livekit.plugins.deepgram"
    sample_rate = SAMPLE_RATE
    frames, transcript, duration = await make_test_speech(sample_rate=sample_rate)

    # TODO: differentiate missing key vs other errors
    try:
        stt_instance = stt_factory()
    except ValueError as e:
        pytest.skip(f"{plugin_id}: {e}")

    async with stt_instance as stt:
        label = f"{stt.model}@{stt.provider}"
        if not stt.capabilities.batch_recognition:
            pytest.skip(f"{label} does not support batch recognition")

        for attempt in range(2):
            try:
                start_time = time.time()

                # Sarvam only supports <30s audio
                if stt.provider in {"Sarvam"} and duration > 30:
                    frames, *_ = await make_test_speech(
                        sample_rate=sample_rate, chunk_duration_ms=5 * 1000
                    )
                    event1 = await stt.recognize(buffer=frames[: len(frames) // 2])
                    event2 = await stt.recognize(buffer=frames[len(frames) // 2 :])
                    event = SpeechEvent(
                        type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=event1.request_id,
                        alternatives=[
                            SpeechData(
                                text=event1.alternatives[0].text
                                + " "
                                + event2.alternatives[0].text,
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
                return
            except (AssertionError, Exception):
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed for {label}, retrying...")
                    continue
                else:
                    raise


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("stt_factory", STTs)
async def test_stream(stt_factory: Callable[[], STT], request):
    sample_rate = SAMPLE_RATE
    plugin_id = request.node.callspec.id.split("-")[0]
    frames, transcript, _ = await make_test_speech(chunk_duration_ms=10, sample_rate=sample_rate)

    # TODO: differentiate missing key vs other errors
    try:
        stt_instance: STT = stt_factory()
    except ValueError as e:
        pytest.skip(f"{plugin_id}: {e}")

    async with stt_instance as stt:
        label = f"{stt.model}@{stt.provider}"
        if not stt.capabilities.streaming:
            pytest.skip(f"{label} does not support streaming")

        for attempt in range(MAX_RETRIES):
            try:
                state = {"closing": False}

                async def _stream_input(
                    frames: list[rtc.AudioFrame], stream: RecognizeStream, state: dict = state
                ):
                    for frame in frames:
                        stream.push_frame(frame)
                        await asyncio.sleep(0.005)

                    stream.end_input()
                    state["closing"] = True

                async def _stream_output(stream: RecognizeStream, state: dict = state):
                    text = ""
                    # make sure the events are sent in the right order
                    recv_start, recv_end = False, True
                    start_time = time.time()
                    got_final_transcript = False

                    async for event in stream:
                        if event.type == agents.stt.SpeechEventType.START_OF_SPEECH:
                            assert recv_end, (
                                "START_OF_SPEECH recv but no END_OF_SPEECH has been sent before"
                            )
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
                            got_final_transcript = True
                            # Some providers don't send END_OF_SPEECH, break after final transcript
                            if state["closing"]:
                                break

                        if event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                            recv_start = False
                            recv_end = True
                            await asyncio.sleep(1)
                            if state["closing"]:
                                break

                    dt = time.time() - start_time
                    print(f"WER: {wer(text, transcript)} for streamed {stt} in {dt:.2f}s")
                    # Relaxed WER threshold for some providers
                    if stt.provider in {
                        "RTZR",  # RTZR defaults to Korean
                        "Deepgram",
                        "Sarvam",
                    }:
                        assert len(text) > 0 and wer(text, transcript) <= 1.0
                    else:
                        assert got_final_transcript, "No FINAL_TRANSCRIPT received"
                        assert wer(text, transcript) <= WER_THRESHOLD

                timed_out = False

                async def _run_test():
                    nonlocal timed_out
                    stream = None
                    try:
                        async with asyncio.timeout(60):
                            stream = stt.stream()
                            await asyncio.gather(
                                _stream_input(frames, stream), _stream_output(stream)
                            )
                    except TimeoutError:
                        timed_out = True
                    finally:
                        if stream is not None:
                            await stream.aclose()

                await _run_test()
                if timed_out:
                    pytest.fail(f"{label} streaming timed out after 60 seconds")
                return
            except (AssertionError, Exception):
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed for {label}, retrying...")
                    continue
                else:
                    raise
