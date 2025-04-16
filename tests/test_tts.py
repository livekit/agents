from __future__ import annotations

import asyncio
import os
import pathlib
import time

import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import APIConnectOptions, APITimeoutError
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
)

from .toxic_proxy import Proxy, Toxiproxy
from .utils import wer

load_dotenv(override=True)


WER_THRESHOLD = 0.2
TEST_AUDIO_SYNTHESIZE = pathlib.Path(os.path.dirname(__file__), "long_synthesize.txt").read_text()

PROXY_LISTEN = "0.0.0.0:443"
OAI_LISTEN = "0.0.0.0:500"


def setup_oai_proxy(toxiproxy: Toxiproxy) -> Proxy:
    return toxiproxy.create("api.openai.com:443", "oai-stt-proxy", listen=OAI_LISTEN, enabled=True)


async def assert_valid_synthesized_audio(frames: AudioBuffer, sample_rate: int, num_channels: int):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)
    import httpx
    import openai as openai_client

    # don't verify ssl because of the proxy base_url being different
    client = openai_client.AsyncClient(http_client=httpx.AsyncClient(verify=False))

    whisper_stt = openai.STT(
        model="whisper-1", base_url="https://172.30.0.10:500/v1", client=client
    )
    res = await whisper_stt.recognize(buffer=frames)
    assert wer(res.alternatives[0].text, TEST_AUDIO_SYNTHESIZE) <= WER_THRESHOLD

    combined_frame = rtc.combine_audio_frames(frames)
    assert combined_frame.sample_rate == sample_rate, "sample rate should be the same"
    assert combined_frame.num_channels == num_channels, "num channels should be the same"


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
            "tts": rime.TTS(),
            "proxy-upstream": "users.rime.ai:443",
        },
        id="rime",
    ),
]

PLUGIN = os.getenv("PLUGIN", "").strip()
if PLUGIN:
    SYNTHESIZE_TTS = [p for p in SYNTHESIZE_TTS if p.id.startswith(PLUGIN)]  # type: ignore


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize(tts_factory, toxiproxy: Toxiproxy):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts.label}-proxy"
    toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)

    frames = []
    try:

        async def process_synthesis():
            async with tts.synthesize(
                text=TEST_AUDIO_SYNTHESIZE, conn_options=APIConnectOptions(max_retry=0, timeout=5)
            ) as stream:
                async for audio in stream:
                    frames.append(audio.frame)

        await asyncio.wait_for(process_synthesis(), timeout=25)
    except asyncio.TimeoutError:
        pytest.fail("test timed out after 25 seconds")
    finally:
        await tts.aclose()

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

    start_time = time.time()
    try:

        async def test_timeout_process():
            async with tts.synthesize(
                text=TEST_AUDIO_SYNTHESIZE,
                conn_options=APIConnectOptions(max_retry=0, timeout=5),
            ) as stream:
                async for _ in stream:
                    pass

        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(test_timeout_process(), timeout=10)
    except asyncio.TimeoutError:
        pytest.fail("test timed out after 10 seconds")
    finally:
        await tts.aclose()

    end_time = time.time()
    elapsed_time = end_time - start_time
    assert 4 <= elapsed_time <= 6, f"Expected timeout around 5 seconds, got {elapsed_time:.2f}s"
