from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import ssl
import time

import aiohttp
import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APITimeoutError
from livekit.agents.tts import TTS
from livekit.agents.utils import AudioBuffer
from livekit.plugins import (
    aws,
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    groq,
    neuphonic,
    openai,
    playai,
    resemble,
    rime,
    speechify,
)

from .fake_tts import FakeTTS
from .toxic_proxy import Proxy, Toxiproxy
from .utils import EventCollector, wer

load_dotenv(override=True)


WER_THRESHOLD = 0.2
TEST_AUDIO_SYNTHESIZE = pathlib.Path(os.path.dirname(__file__), "long_synthesize.txt").read_text()

PROXY_LISTEN = "0.0.0.0:443"
OAI_LISTEN = "0.0.0.0:500"


def setup_oai_proxy(toxiproxy: Toxiproxy) -> Proxy:
    return toxiproxy.create("api.openai.com:443", "oai-stt-proxy", listen=OAI_LISTEN, enabled=True)


async def assert_valid_synthesized_audio(frames: AudioBuffer, sample_rate: int, num_channels: int):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)

    frame = rtc.combine_audio_frames(frames)
    data = frame.to_wav_bytes()
    form = aiohttp.FormData()
    form.add_field("file", data, filename="file.wav", content_type="audio/wav")
    form.add_field("model", "whisper-1")
    form.add_field("response_format", "verbose_json")

    ssl_ctx = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        async with session.post(
            "https://toxiproxy:500/v1/audio/transcriptions",
            data=form,
            headers={
                "Host": "api.openai.com",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
            ssl=ssl_ctx,
            server_hostname="api.openai.com",
        ) as resp:
            result = await resp.json()

    # semantic
    assert wer(result["text"], TEST_AUDIO_SYNTHESIZE) <= WER_THRESHOLD
    combined_frame = rtc.combine_audio_frames(frames)
    assert combined_frame.sample_rate == sample_rate, "sample rate should be the same"
    assert combined_frame.num_channels == num_channels, "num channels should be the same"

    # clipping
    # signal = np.array(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    # peak = np.iinfo(np.int16).max
    # num_clipped = np.sum((signal >= peak) | (signal <= -peak))
    # assert num_clipped >= 0, f"{num_clipped} samples are clipped"


SYNTHESIZE_TTS = [
    pytest.param(
        lambda: {
            "tts": cartesia.TTS(),
            "proxy-upstream": "api.cartesia.ai:443",
        },
        id="cartesia",
    ),
    pytest.param(
        lambda: {
            "tts": aws.TTS(region="us-west-2"),
            "proxy-upstream": "polly.us-west-2.amazonaws.com:443",
        },
        id="aws",
    ),
    pytest.param(
        lambda: {
            "tts": azure.TTS(),
            "proxy-upstream": "westus.tts.speech.microsoft.com:443",
        },
        id="azure",
    ),
    pytest.param(
        lambda: {
            "tts": deepgram.TTS(),
            "proxy-upstream": "api.deepgram.com:443",
        },
        id="deepgram",
    ),
    pytest.param(
        lambda: {
            "tts": elevenlabs.TTS(),
            "proxy-upstream": "api.elevenlabs.io:443",
        },
        id="elevenlabs",
    ),
    pytest.param(
        lambda: {
            "tts": google.TTS(),
            "proxy-upstream": "texttospeech.googleapis.com:443",
        },
        id="google",
    ),
    pytest.param(
        lambda: {
            "tts": groq.TTS(),
            "proxy-upstream": "api.groq.com:443",
        },
        id="groq",
    ),
    pytest.param(
        lambda: {
            "tts": neuphonic.TTS(),
            "proxy-upstream": "api.neuphonic.com:443",
        },
        id="neuphonic",
    ),
    pytest.param(
        lambda: {
            "tts": openai.TTS(),
            "proxy-upstream": "api.openai.com:443",
        },
        id="openai",
    ),
    pytest.param(
        lambda: {
            "tts": playai.TTS(),
            "proxy-upstream": "api.play.ht:443",
        },
        id="playai",
    ),
    pytest.param(
        lambda: {
            "tts": resemble.TTS(),
            "proxy-upstream": "f.cluster.resemble.ai:443",
        },
        id="resemble",
    ),
    pytest.param(
        lambda: {
            "tts": rime.TTS(
                model="mistv2",
            ),
            "proxy-upstream": "users.rime.ai:443",
        },
        id="rime",
    ),
    pytest.param(
        lambda: {
            "tts": speechify.TTS(),
            "proxy-upstream": "api.sws.speechify.com:443",
        },
        id="speechify",
    ),
]

PLUGIN = os.getenv("PLUGIN", "").strip()
if PLUGIN:
    SYNTHESIZE_TTS = [p for p in SYNTHESIZE_TTS if p.id.startswith(PLUGIN)]  # type: ignore


async def _do_synthesis(tts_v: TTS, *, conn_options: APIConnectOptions) -> list[rtc.AudioFrame]:
    from livekit.agents import tts

    audio_events: list[tts.SynthesizedAudio] = []
    async with tts_v.synthesize(text=TEST_AUDIO_SYNTHESIZE, conn_options=conn_options) as stream:
        async for audio in stream:
            audio_events.append(audio)

    assert all(not event.is_final for event in audio_events[:-1]), (
        "expected all audio events to be non-final"
    )
    assert all(0.05 < event.frame.duration < 0.25 for event in audio_events[:-1]), (
        "expected all frames to have a duration between 50ms and 250ms"
    )
    assert audio_events[-1].is_final, "expected last audio event to be final"
    assert 0 < audio_events[-1].frame.duration < 0.25, "expected last frame to not be empty"

    first_id = audio_events[0].request_id
    assert first_id, "expected to have a request_id"
    assert all(e.request_id == first_id for e in audio_events), (
        "expected all frames to have the same request_id, "
    )

    frames = [event.frame for event in audio_events]
    return frames


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize(tts_factory, toxiproxy: Toxiproxy, logger: logging.Logger):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts.label}-proxy"
    toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)

    metrics_collected_events = EventCollector(tts, "metrics_collected")

    try:
        frames = await asyncio.wait_for(
            _do_synthesis(tts, conn_options=APIConnectOptions(max_retry=3, timeout=5)), timeout=30
        )
    except asyncio.TimeoutError:
        pytest.fail("test timed out after 30 seconds")
    finally:
        await tts.aclose()

    assert metrics_collected_events.count == 1, (
        "expected 1 metrics collected event, got {metrics_collected_events.count}"
    )
    logger.info(f"metrics: {metrics_collected_events.events[0][0][0]}")

    await assert_valid_synthesized_audio(frames, tts.sample_rate, tts.num_channels)


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize_timeout(tts_factory, toxiproxy: Toxiproxy):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts.label}-proxy"
    p = toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)
    p.add_toxic(type="timeout", attributes={"timeout": 0})

    try:
        # test timeout
        start_time = time.time()
        try:
            with pytest.raises(APITimeoutError):
                await asyncio.wait_for(
                    _do_synthesis(tts, conn_options=APIConnectOptions(max_retry=0, timeout=2.5)),
                    timeout=10,
                )
        except asyncio.TimeoutError:
            pytest.fail("test timed out after 10 seconds")

        end_time = time.time()
        elapsed_time = end_time - start_time
        assert 1.5 <= elapsed_time <= 3.5, (
            f"expected timeout around 2 seconds, got {elapsed_time:.2f}s"
        )

        # test retries
        error_events = EventCollector(tts, "error")
        metrics_collected_events = EventCollector(tts, "metrics_collected")
        start_time = time.time()
        with pytest.raises(APITimeoutError):
            await _do_synthesis(
                tts, conn_options=APIConnectOptions(max_retry=3, timeout=0.5, retry_interval=0.0)
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        assert error_events.count == 4, "expected 4 errors, got {error_events.count}"
        assert 1 <= elapsed_time <= 3, (
            f"expected total timeout around 2 seconds, got {elapsed_time:.2f}s"
        )
        assert metrics_collected_events.count == 0, (
            "expected 0 metrics collected events, got {metrics_collected_events.count}"
        )
    finally:
        await tts.aclose()


async def test_error_prop():
    tts = FakeTTS(fake_audio_duration=0.0)
    try:
        with pytest.raises(APIError, match="no audio frames"):
            await _do_synthesis(tts, conn_options=APIConnectOptions(max_retry=0, timeout=0.5))

        tts.update_options(fake_exception=RuntimeError("test error"))
        with pytest.raises(RuntimeError, match="test error"):
            await _do_synthesis(tts, conn_options=APIConnectOptions(max_retry=0, timeout=0.5))
    finally:
        await tts.aclose()
