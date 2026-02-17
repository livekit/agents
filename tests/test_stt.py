"""
Do speech recognition on a long audio file and compare the result with the expected transcript
"""

import asyncio
import math
import time
from collections.abc import Callable

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
] + [
    pytest.param(lambda: deepgram.STTv2(), id="livekit.plugins.deepgram.STTv2"),
    pytest.param(
        lambda: gradium.STT(model_endpoint="wss://us.api.gradium.ai/api/speech/asr"),
        id="livekit.plugins.gradium.STT",
    ),
]


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


async def batch_recognize(
    stt: stt.STT, frames: list[rtc.AudioFrame], n_batches: int = 1
) -> SpeechEvent:
    if n_batches == 1:
        return await stt.recognize(buffer=frames)
    if n_batches > len(frames):
        raise ValueError("n_batches must be less than or equal to the number of frames")

    batch_size: int = len(frames) // n_batches
    events: list[SpeechEvent] = []
    for i in range(n_batches):
        batch = frames[i * batch_size : (i + 1) * batch_size]
        events.append(await stt.recognize(buffer=batch))

    assert len(events) > 0
    return SpeechEvent(
        type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=events[0].request_id,
        alternatives=[
            SpeechData(
                text=" ".join(
                    [
                        event.alternatives[0].text
                        for event in events
                        if event.alternatives[0].text is not None
                    ]
                ),
                language=events[0].alternatives[0].language,
            )
        ],
    )


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
        if not stt.capabilities.offline_recognize:
            pytest.skip(f"{label} does not support batch recognition")

        for attempt in range(2):
            try:
                start_time = time.time()

                # WARN: Sarvam only supports <30s audio chunks
                if stt.provider == "Sarvam" and duration > 30:
                    frames, *_ = await make_test_speech(
                        sample_rate=sample_rate, chunk_duration_ms=5 * 1000
                    )
                    n_batches = math.ceil(duration / 30)
                else:
                    n_batches = 1
                event = await batch_recognize(stt, frames, n_batches)
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
                        "FireworksAI",
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
