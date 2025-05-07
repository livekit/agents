from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import ssl
import time
import io
import av
from av.error import InvalidDataError


import aiohttp
import pytest
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APITimeoutError
from livekit.agents import tts
from livekit.agents.utils import AudioBuffer, aio
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
    hume,
)

from .fake_tts import FakeTTS
from .toxic_proxy import Proxy, Toxiproxy
from .utils import EventCollector, wer, fake_llm_stream
from collections import defaultdict

load_dotenv(override=True)


WER_THRESHOLD = 0.2
TEST_AUDIO_SYNTHESIZE = pathlib.Path(os.path.dirname(__file__), "long_synthesize.txt").read_text()
TEST_AUDIO_SYNTHESIZE_MULTI_TOKENS = pathlib.Path(
    os.path.dirname(__file__), "long_synthesize_multi_tokens.txt"
).read_text()

PROXY_LISTEN = "0.0.0.0:443"
OAI_LISTEN = "0.0.0.0:500"


def setup_oai_proxy(toxiproxy: Toxiproxy) -> Proxy:
    return toxiproxy.create("api.openai.com:443", "oai-stt-proxy", listen=OAI_LISTEN, enabled=True)


async def assert_valid_synthesized_audio(
    *, frames: AudioBuffer, text: str, sample_rate: int, num_channels: int
):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)
    frame = rtc.combine_audio_frames(frames)

    # Make sure the data is PCM and can't be another container.
    # OpenAI STT seems to probe the input so the test could still pass even if the data isn't PCM.
    with pytest.raises(InvalidDataError):
        container = av.open(io.BytesIO(frame.data))
        container.close()

    assert len(frame.data) >= frame.samples_per_channel

    assert frame.sample_rate == sample_rate, "sample rate should be the same"
    assert frame.num_channels == num_channels, "num channels should be the same"

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
    assert wer(result["text"], text) <= WER_THRESHOLD

    # clipping
    # signal = np.array(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    # peak = np.iinfo(np.int16).max
    # num_clipped = np.sum((signal >= peak) | (signal <= -peak))
    # assert num_clipped <= 10, f"{num_clipped} samples are clipped"


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
    pytest.param(
        lambda: {
            "tts": speechify.TTS(),
            "proxy-upstream": "api.sws.speechify.com:443",
        },
        id="speechify",
    ),
    pytest.param(
        lambda: {
            "tts": hume.TTS(),
            "proxy-upstream": "api.hume.ai:443",
        },
        id="hume",
    ),
]

PLUGIN = os.getenv("PLUGIN", "").strip()
if PLUGIN:
    SYNTHESIZE_TTS = [p for p in SYNTHESIZE_TTS if p.id.startswith(PLUGIN)]  # type: ignore


async def _do_synthesis(tts_v: tts.TTS, segment: str, *, conn_options: APIConnectOptions):
    tts_stream = tts_v.synthesize(text=segment, conn_options=conn_options)
    audio_events = [event async for event in tts_stream]

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
    await assert_valid_synthesized_audio(
        frames=frames,
        text=TEST_AUDIO_SYNTHESIZE,
        sample_rate=tts_v.sample_rate,
        num_channels=tts_v.num_channels,
    )


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize(tts_factory, toxiproxy: Toxiproxy, logger: logging.Logger):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts_v: tts.TTS = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts_v.label}-proxy"
    toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)

    tts_v.prewarm()

    metrics_collected_events = EventCollector(tts_v, "metrics_collected")
    try:
        await asyncio.wait_for(
            _do_synthesis(
                tts_v, TEST_AUDIO_SYNTHESIZE, conn_options=APIConnectOptions(max_retry=3, timeout=5)
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        pytest.fail("test timed out after 30 seconds")
    finally:
        await tts_v.aclose()

    assert metrics_collected_events.count == 1, (
        f"expected 1 metrics collected event, got {metrics_collected_events.count}"
    )
    logger.info(f"metrics: {metrics_collected_events.events[0][0][0]}")


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize_timeout(tts_factory, toxiproxy: Toxiproxy):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts_v: tts.TTS = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts_v.label}-proxy"
    p = toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)
    p.add_toxic(type="timeout", attributes={"timeout": 0})

    try:
        # test timeout
        start_time = time.time()
        try:
            with pytest.raises(APITimeoutError):
                await asyncio.wait_for(
                    _do_synthesis(
                        tts_v,
                        TEST_AUDIO_SYNTHESIZE,
                        conn_options=APIConnectOptions(max_retry=0, timeout=2.5),
                    ),
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
        error_events = EventCollector(tts_v, "error")
        metrics_collected_events = EventCollector(tts_v, "metrics_collected")
        start_time = time.time()
        with pytest.raises(APITimeoutError):
            await _do_synthesis(
                tts_v,
                TEST_AUDIO_SYNTHESIZE,
                conn_options=APIConnectOptions(max_retry=3, timeout=0.5, retry_interval=0.0),
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
        await tts_v.aclose()


async def test_synthesize_error_propagation():
    tts = FakeTTS(fake_audio_duration=0.0)

    try:
        with pytest.raises(APIError, match="no audio frames"):
            await _do_synthesis(
                tts, "fake_text", conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
            )

        tts.update_options(fake_exception=RuntimeError("test error"))
        with pytest.raises(RuntimeError, match="test error"):
            await _do_synthesis(
                tts, "fake_text", conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
            )
    finally:
        await tts.aclose()


STREAM_TTS = [
    pytest.param(
        lambda: {
            "tts": cartesia.TTS(),
            "proxy-upstream": "api.cartesia.ai:443",
        },
        id="cartesia",
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
            "tts": deepgram.TTS(),
            "proxy-upstream": "api.deepgram.com:443",
        },
        id="deepgram",
    ),
    pytest.param(
        lambda: {
            "tts": resemble.TTS(),
            "proxy-upstream": "websocket.cluster.resemble.ai:443",
        },
        id="resemble",
    ),
]

PLUGIN = os.getenv("PLUGIN", "").strip()
if PLUGIN:
    STREAM_TTS = [p for p in STREAM_TTS if p.id.startswith(PLUGIN)]  # type: ignore


async def _do_stream(tts_v: tts.TTS, segments: list[str], *, conn_options: APIConnectOptions):
    async with tts_v.stream(conn_options=conn_options) as tts_stream:
        flush_times = []

        async def _push_text() -> None:
            for text in segments:
                async for token in fake_llm_stream(text, tokens_per_second=30.0):
                    tts_stream.push_text(token)

                tts_stream.flush()
                flush_times.append(time.time())

            tts_stream.end_input()

        push_text_task = asyncio.create_task(_push_text())

        audio_events: list[tts.SynthesizedAudio] = []
        audio_events_recv_times = []

        try:
            async for event in tts_stream:
                audio_events.append(event)
                audio_events_recv_times.append(time.time())
        except BaseException:
            await aio.cancel_and_wait(push_text_task)
            raise

        assert push_text_task.done(), "expected push_text_task to be done"

        request_id = audio_events[0].request_id
        assert request_id, "expected to have a request_id"
        assert all(e.request_id == request_id for e in audio_events), (
            "expected all frames to have the same request_id"
        )
        assert all(e.segment_id for e in audio_events), "expected all events to have a segment_id"

        by_segment: dict[str, list[tts.SynthesizedAudio]] = defaultdict(list)
        for e in audio_events:
            by_segment[e.segment_id].append(e)

        assert len(by_segment) == len(segments), (
            "expected one unique segment_id per pushed text segment"
        )

        assert len(by_segment) >= 1, "expected at least one segment"

        for seg_idx, (segment_text, segment_events) in enumerate(
            zip(segments, by_segment.values())
        ):
            *non_final, final = segment_events

            idx = audio_events.index(non_final[0])
            recv_time = audio_events_recv_times[idx]

            # if the first audio event is received after the flush, then there is no point
            # in using the streaming method for this TTS.
            # The above fake_llm_stream has a slow token/s rate of 30
            assert recv_time < flush_times[seg_idx], (
                "expected the first audio to be received before the first flush"
            )

            assert final.is_final, "last frame of a segment must be final"
            assert all(not e.is_final for e in non_final), (
                "only the last frame within a segment may be final"
            )

            assert 0 < final.frame.duration < 0.25, "expected final frame to be non-empty (<250 ms)"
            assert all(0.05 < e.frame.duration < 0.25 for e in non_final), (
                "expected non-final frames to be 50-250 ms"
            )

            frames = [e.frame for e in segment_events]
            await assert_valid_synthesized_audio(
                frames=frames,
                text=segment_text,
                sample_rate=tts_v.sample_rate,
                num_channels=tts_v.num_channels,
            )


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", STREAM_TTS)
async def test_stream(tts_factory, toxiproxy: Toxiproxy, logger: logging.Logger):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts_v: tts.TTS = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts_v.label}-proxy"
    toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)

    tts_v.prewarm()

    metrics_collected_events = EventCollector(tts_v, "metrics_collected")
    try:
        # test one segment
        await asyncio.wait_for(
            _do_stream(
                tts_v,
                [TEST_AUDIO_SYNTHESIZE],
                conn_options=APIConnectOptions(max_retry=3, timeout=5),
            ),
            timeout=30,
        )

        # the metrics could not be emitted if the _mark_started() method was never called in streaming mode
        assert metrics_collected_events.count == 1, (
            f"expected 1 metrics collected event, got {metrics_collected_events.count}"
        )
        logger.info(f"metrics: {metrics_collected_events.events[0][0][0]}")

        metrics_collected_events.clear()

        # test multiple segments in one stream
        # TODO: NOT SUPPORTED YET

        # await asyncio.wait_for(
        #     _do_stream(
        #         tts_v,
        #         [TEST_AUDIO_SYNTHESIZE, TEST_AUDIO_SYNTHESIZE_MULTI_TOKENS],
        #         conn_options=APIConnectOptions(max_retry=3, timeout=5),
        #     ),
        #     timeout=30,
        # )

        # assert metrics_collected_events.count == 2, (
        #     "expected 2 metrics collected event, got {metrics_collected_events.count}"
        # )
        # logger.info(f"1st segment metrics: {metrics_collected_events.events[0][0][0]}")
        # logger.info(f"2nd segment metrics: {metrics_collected_events.events[1][0][0]}")
    except asyncio.TimeoutError:
        pytest.fail("test timed out after 30 seconds")
    finally:
        await tts_v.aclose()


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", STREAM_TTS)
async def test_stream_timeout(tts_factory, toxiproxy: Toxiproxy):
    setup_oai_proxy(toxiproxy)
    tts_info: dict = tts_factory()
    tts_v: tts.TTS = tts_info["tts"]
    proxy_upstream = tts_info["proxy-upstream"]
    proxy_name = f"{tts_v.label}-proxy"
    p = toxiproxy.create(proxy_upstream, proxy_name, listen=PROXY_LISTEN, enabled=True)
    p.add_toxic(type="timeout", attributes={"timeout": 0})

    try:
        # test timeout
        start_time = time.time()
        try:
            with pytest.raises(APITimeoutError):
                await asyncio.wait_for(
                    _do_stream(
                        tts_v,
                        [TEST_AUDIO_SYNTHESIZE],
                        conn_options=APIConnectOptions(max_retry=0, timeout=2.5),
                    ),
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
        error_events = EventCollector(tts_v, "error")
        metrics_collected_events = EventCollector(tts_v, "metrics_collected")
        start_time = time.time()
        with pytest.raises(APITimeoutError):
            await _do_stream(
                tts_v,
                [TEST_AUDIO_SYNTHESIZE],
                conn_options=APIConnectOptions(max_retry=3, timeout=0.5, retry_interval=0.0),
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        assert error_events.count == 4, f"expected 4 errors, got {error_events.count}"
        assert 1 <= elapsed_time <= 3, (
            f"expected total timeout around 2 seconds, got {elapsed_time:.2f}s"
        )
        assert metrics_collected_events.count == 0, (
            f"expected 0 metrics collected events, got {metrics_collected_events.count}"
        )
    finally:
        await tts_v.aclose()
