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
    inworld,
    mistralai,
    nvidia,
    openai,
    reson8,
    rtzr,
    sarvam,
    silero,
    soniox,
    speechmatics,
    xai,
)

from .utils import make_test_speech, wer
from .utils.stt import assert_speech_end_time, collect_stream_events

pytestmark = pytest.mark.stt

SAMPLE_RATE = 24000
WER_THRESHOLD = 0.25
MAX_RETRIES = 2
STREAM_CHUNK_DURATION_MS = 10
TRAILING_SILENCE_DURATION_MS = 2000
# The last spoken word in tests/long.mp3 ends at approximately 44.4 seconds.
SPEECH_END_TIME = 44.4

UNTIMED_EOS_REASONS = {
    "livekit.plugins.aws": "AWS EOS timing is deferred until live validation",
    "livekit.plugins.cartesia": "Cartesia turns does not provide speech-end timestamps",
    "livekit.plugins.fireworksai": "Fireworks emits an untimed EOS during connection cleanup",
    "livekit.agents.inference": "The gateway-selected model has no verified endpoint timing",
    "livekit.agents.inference.cartesia": "Cartesia gateway duration measures billed audio",
    "livekit.plugins.mistralai.realtime_v3": (
        "Voxtral 3 server turn timing includes the endpointing delay"
    ),
    "livekit.plugins.sarvam": "Sarvam saaras:v4 sends no speech or word timestamps",
    "livekit.plugins.sarvam.realtime": "Sarvam can emit EOS before its timed final arrives",
    "livekit.plugins.nvidia": "NVIDIA EOS timing is deferred until live validation",
    "livekit.plugins.reson8": "Reson8 emits EOS without a provider speech-end timestamp",
}


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
        nvidia,
        openai,
        soniox,
        inference,
        azure,
        aws,
        reson8,
        # rtzr,
        # TODO: only Business account allowed outside South Korea
        # clova,
        # TODO: https://github.com/spi-tch/spitch-python/issues/162
        # spitch,
    ]
] + [
    pytest.param(
        lambda: mistralai.STT(model="voxtral-mini-transcribe-realtime-3"),
        id="livekit.plugins.mistralai.realtime_v3",
    ),
    pytest.param(lambda: cartesia.STT(model="ink-2"), id="livekit.plugins.cartesia"),
    pytest.param(lambda: sarvam.STT(model="saaras:v4"), id="livekit.plugins.sarvam"),
    pytest.param(
        lambda: google.STT(enable_voice_activity_events=True), id="livekit.plugins.google"
    ),
    pytest.param(lambda: cartesia.STT(model="ink-whisper"), id="livekit.plugins.cartesia._legacy"),
    pytest.param(lambda: deepgram.STTv2(), id="livekit.plugins.deepgram.STTv2"),
    pytest.param(
        lambda: gradium.STT(model_endpoint="wss://us.api.gradium.ai/api/speech/asr"),
        id="livekit.plugins.gradium.STT",
    ),
]

# Additional model configurations used only by streaming tests.
STREAM_ONLY_STTs: list[Callable[[], stt.STT]] = [
    parameter_factory(xai),
    parameter_factory(inworld),
] + [
    pytest.param(
        lambda: rtzr.STT(sample_rate=16000),
        id="livekit.plugins.rtzr",
    ),
    pytest.param(
        lambda: elevenlabs.STT(
            model="scribe_v2_realtime",
            include_timestamps=True,
            server_vad={"vad_silence_threshold_secs": 0.5},
        ),
        id="livekit.plugins.elevenlabs.realtime",
    ),
    pytest.param(
        lambda: mistralai.STT(model="voxtral-mini-transcribe-realtime-2602", vad=silero.VAD.load()),
        id="livekit.plugins.mistralai.realtime",
    ),
    pytest.param(
        lambda: mistralai.STT(model="voxtral-mini-transcribe-realtime-3", vad=silero.VAD.load()),
        id="livekit.plugins.mistralai.realtime_v3_vad",
    ),
    pytest.param(
        lambda: sarvam.STTRealtime(language="en-IN"), id="livekit.plugins.sarvam.realtime"
    ),
    pytest.param(lambda: openai.STT(use_realtime=True), id="livekit.plugins.openai.realtime"),
    pytest.param(
        lambda: inference.STT(model="deepgram/nova-3"), id="livekit.agents.inference.deepgram"
    ),
    pytest.param(
        lambda: inference.STT(model="cartesia/ink-whisper"),
        id="livekit.agents.inference.cartesia",
    ),
    pytest.param(
        lambda: inference.STT(
            model="assemblyai/universal-streaming", extra_kwargs={"format_turns": True}
        ),
        id="livekit.agents.inference.assemblyai",
    ),
    pytest.param(lambda: inference.STT(model="xai/stt-1"), id="livekit.agents.inference.xai"),
    pytest.param(
        lambda: assemblyai.STT(
            model="universal-3-5-pro",
            language_detection=True,
            min_turn_silence=75,
            max_turn_silence=1000,
            mode="balanced",
            interruption_delay=400,
            vad_threshold=0.3,
        ),
        id="livekit.plugins.assemblyai.u3_pro",
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
@pytest.mark.parametrize("stt_factory", STTs + STREAM_ONLY_STTs)
async def test_stream(stt_factory: Callable[[], STT], request):
    plugin_id = request.node.callspec.id
    frames, transcript, _ = await make_test_speech(
        chunk_duration_ms=STREAM_CHUNK_DURATION_MS, sample_rate=SAMPLE_RATE
    )
    silence = rtc.AudioFrame.create(
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLE_RATE * STREAM_CHUNK_DURATION_MS // 1000,
    )
    frames.extend([silence] * (TRAILING_SILENCE_DURATION_MS // STREAM_CHUNK_DURATION_MS))

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
                started_at = time.monotonic()
                async with stt.stream() as stream:
                    async with asyncio.timeout(120):
                        events, audio_started_at = await collect_stream_events(stream, frames)

                text = ""
                recv_start, recv_end = False, True
                got_final_transcript = False
                for event in events:
                    if event.type == agents.stt.SpeechEventType.START_OF_SPEECH:
                        assert recv_end, (
                            "START_OF_SPEECH recv but no END_OF_SPEECH has been sent before"
                        )
                        assert not recv_start
                        recv_end = False
                        recv_start = True
                    elif event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                        if text:
                            text += " "
                        text += event.alternatives[0].text
                        language = event.alternatives[0].language
                        # Voxtral realtime can omit its detected language.
                        if stt.provider not in {"FireworksAI", "RTZR", "livekit"} and (
                            language or stt.provider != "MistralAI"
                        ):
                            assert language is not None
                            assert language.lower().startswith("en")
                        got_final_transcript = True
                    elif event.type == agents.stt.SpeechEventType.END_OF_SPEECH:
                        recv_start = False
                        recv_end = True

                speech_end = assert_speech_end_time(
                    events,
                    audio_started_at=audio_started_at,
                    expected_end=SPEECH_END_TIME,
                    untimed_reason=UNTIMED_EOS_REASONS.get(plugin_id),
                )
                dt = time.monotonic() - started_at
                print(f"EOS: {speech_end}s; WER: {wer(text, transcript)} for {label} in {dt:.2f}s")
                # RTZR's native streaming model recognizes Korean.
                if stt.provider in {"Deepgram", "Sarvam", "FireworksAI", "RTZR"}:
                    assert text and wer(text, transcript) <= 1.0
                else:
                    assert got_final_transcript, "No FINAL_TRANSCRIPT received"
                    assert wer(text, transcript) <= WER_THRESHOLD
                return
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed for {label}, retrying...")
                else:
                    raise
