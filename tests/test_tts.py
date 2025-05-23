from __future__ import annotations

import asyncio
import io
import logging
import os
import pathlib
import ssl
import time
import wave
from collections import defaultdict

import aiohttp
import av
import pytest
from av.error import InvalidDataError
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APITimeoutError, tokenize, tts
from livekit.agents.utils import AudioBuffer, aio
from livekit.plugins import (
    aws,
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    groq,
    hume,
    neuphonic,
    openai,
    playai,
    resemble,
    rime,
    speechify,
)

from .fake_tts import FakeTTS
from .toxic_proxy import Proxy, Toxiproxy
from .utils import EventCollector, fake_llm_stream, wer

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
async def test_tts_synthesize(tts_factory, toxiproxy: Toxiproxy, logger: logging.Logger):
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
async def test_tts_synthesize_timeout(tts_factory, toxiproxy: Toxiproxy):
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


async def test_tts_synthesize_error_propagation():
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
    pytest.param(
        lambda: {
            "tts": tts.StreamAdapter(
                tts=openai.TTS(), sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            ),
            "proxy-upstream": "api.openai.com:443",
        },
        id="openai-stream-adapter",
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
async def test_tts_stream(tts_factory, toxiproxy: Toxiproxy, logger: logging.Logger):
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

        if isinstance(tts_v, tts.StreamAdapter):
            assert metrics_collected_events.count >= 1, (
                f"expected >=1 metrics collected event, got {metrics_collected_events.count}"
            )
        else:
            assert metrics_collected_events.count == 1, (
                f"expected 1 metrics collected event, got {metrics_collected_events.count}"
            )

        for event in metrics_collected_events.events:
            logger.info(f"metrics: {event[0][0]}")

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
async def test_tts__stream_timeout(tts_factory, toxiproxy: Toxiproxy):
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


async def test_tts_audio_emitter(monkeypatch):
    monkeypatch.setattr(tts.tts, "lk_dump_tts", False)

    # build a known PCM chunk: 100 samples × 2 bytes each = 200 bytes
    # sample_rate=1000, frame_size_ms=100 => 100 samples per frame
    pcm_chunk = b"\xff\xff" * 100

    # --- Test streaming logic with explicit flush ---
    rx_stream = aio.Chan[tts.SynthesizedAudio]()
    emitter_stream = tts.AudioEmitter(label="test", dst_ch=rx_stream)
    emitter_stream.initialize(
        request_id="req-stream",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )

    # segment 1: flush before final
    emitter_stream.start_segment(segment_id="seg1")
    emitter_stream.push(pcm_chunk)
    emitter_stream.flush()  # non-final
    emitter_stream.push(pcm_chunk)
    emitter_stream.end_segment()  # final

    # segment 2: no flush, two pushes then end
    emitter_stream.start_segment(segment_id="seg2")
    emitter_stream.push(pcm_chunk)
    emitter_stream.push(pcm_chunk)
    emitter_stream.end_segment()

    # signal end of input so main loop can exit
    emitter_stream.end_input()
    await emitter_stream.join()
    rx_stream.close()

    msgs = [msg async for msg in rx_stream]
    assert len(msgs) == 4

    # seg1: one non-final, one final
    m0, m1, m2, m3 = msgs
    assert (m0.segment_id, m0.is_final, m0.frame.data.tobytes()) == ("seg1", False, pcm_chunk)
    assert (m1.segment_id, m1.is_final, m1.frame.data.tobytes()) == ("seg1", True, pcm_chunk)

    # seg2: direct end without flush still yields non-final then final
    assert (m2.segment_id, m2.is_final, m2.frame.data.tobytes()) == ("seg2", False, pcm_chunk)
    assert (m3.segment_id, m3.is_final, m3.frame.data.tobytes()) == ("seg2", True, pcm_chunk)

    # durations: two frames × 0.1s = 0.2s
    assert emitter_stream.pushed_duration(0) == 0.2
    assert emitter_stream.pushed_duration(1) == 0.2

    # --- Test multiple flush in streaming ---
    rx_multi = aio.Chan[tts.SynthesizedAudio]()
    emitter_multi = tts.AudioEmitter(label="multi", dst_ch=rx_multi)
    emitter_multi.initialize(
        request_id="req-multi",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )

    emitter_multi.start_segment(segment_id="S")
    emitter_multi.push(pcm_chunk)
    emitter_multi.flush()  # A (non-final)
    emitter_multi.push(pcm_chunk)
    emitter_multi.flush()  # B (non-final)
    emitter_multi.push(pcm_chunk)
    emitter_multi.end_segment()  # C (final)

    emitter_multi.end_input()
    await emitter_multi.join()
    rx_multi.close()

    msgs2 = [msg async for msg in rx_multi]
    assert len(msgs2) == 3
    assert (msgs2[0].frame.data.tobytes(), msgs2[0].is_final) == (pcm_chunk, False)
    assert (msgs2[1].frame.data.tobytes(), msgs2[1].is_final) == (pcm_chunk, False)
    assert (msgs2[2].frame.data.tobytes(), msgs2[2].is_final) == (pcm_chunk, True)

    # --- Test non-streaming logic (flush acts as final) ---
    rx_nostream = aio.Chan[tts.SynthesizedAudio]()
    emitter_nostream = tts.AudioEmitter(label="nos", dst_ch=rx_nostream)
    emitter_nostream.initialize(
        request_id="req-nos",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=False,
    )
    emitter_nostream.push(pcm_chunk)
    emitter_nostream.push(pcm_chunk)
    emitter_nostream.flush()  # acts as final

    # no end_input needed: flush() already closed in non-streaming
    await emitter_nostream.join()
    rx_nostream.close()

    msgs3 = [msg async for msg in rx_nostream]
    assert len(msgs3) == 2
    assert (msgs3[0].frame.data.tobytes(), msgs3[0].is_final) == (pcm_chunk, False)
    assert (msgs3[1].frame.data.tobytes(), msgs3[1].is_final) == (pcm_chunk, True)

    # --- Test direct end_segment without flush in streaming ---
    rx_noflush = aio.Chan[tts.SynthesizedAudio]()
    emitter_noflush = tts.AudioEmitter(label="noflush", dst_ch=rx_noflush)
    emitter_noflush.initialize(
        request_id="req-noflush",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )

    emitter_noflush.start_segment(segment_id="empty")
    emitter_noflush.push(pcm_chunk)
    emitter_noflush.end_segment()  # no prior flush
    emitter_noflush.end_input()
    await emitter_noflush.join()
    rx_noflush.close()

    msgs4 = [msg async for msg in rx_noflush]
    assert len(msgs4) == 1

    # no flush, direct end_segment will not having the "fake frame"
    assert msgs4[0].is_final is True and msgs4[0].frame.data.tobytes() == pcm_chunk

    # test fake audio
    rx_noflush = aio.Chan[tts.SynthesizedAudio]()
    emitter_noflush = tts.AudioEmitter(label="fakeaudio", dst_ch=rx_noflush)
    emitter_noflush.initialize(
        request_id="req-fake-audio",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )

    emitter_noflush.start_segment(segment_id="empty")
    emitter_noflush.push(pcm_chunk)
    emitter_noflush.flush()
    emitter_noflush.end_segment()  # no prior flush
    emitter_noflush.end_input()
    await emitter_noflush.join()
    rx_noflush.close()

    msgs5 = [msg async for msg in rx_noflush]
    assert len(msgs5) == 2
    assert msgs5[0].is_final is False and msgs5[0].frame.data.tobytes() == pcm_chunk
    assert msgs5[1].is_final is True and msgs5[1].frame.data.tobytes() == b"\x00\x00" * 10

    # --- No silence on empty flush or double flush ---
    rx_empty = aio.Chan[tts.SynthesizedAudio]()
    emitter_empty = tts.AudioEmitter(label="empty", dst_ch=rx_empty)
    emitter_empty.initialize(
        request_id="req-empty",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )
    emitter_empty.start_segment(segment_id="e1")
    emitter_empty.flush()  # no data
    emitter_empty.flush()  # still no data
    emitter_empty.end_segment()
    emitter_empty.end_input()
    await emitter_empty.join()
    rx_empty.close()

    msgs6 = [msg async for msg in rx_empty]
    # no data => no frames at all
    assert len(msgs6) == 0
    assert emitter_empty.pushed_duration(0) == 0.0


async def test_tts_audio_emitter_wav(monkeypatch):
    monkeypatch.setattr(tts.tts, "lk_dump_tts", False)

    # build a small WAV: 300 ms total (3 × 100 ms chunks)
    # sample_rate=1000, 1 channel, 16-bit
    pcm_chunk = b"\x7f\x7f" * 100  # 100 samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1000)
        wf.writeframes(pcm_chunk * 3)
    wav_bytes = buf.getvalue()

    # --- Streaming, two segments ---
    rx = aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="wav-multi", dst_ch=rx)
    emitter.initialize(
        request_id="req-wav-multi",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/wav",
        frame_size_ms=100,
        stream=True,
    )

    # Segment 1
    emitter.start_segment(segment_id="w1")
    emitter.push(wav_bytes)
    emitter.end_segment()

    # Segment 2
    emitter.start_segment(segment_id="w2")
    emitter.push(wav_bytes)
    emitter.end_segment()

    # finish and collect
    emitter.end_input()
    await emitter.join()
    rx.close()
    msgs = [msg async for msg in rx]

    # Expect 2 segments × 3 frames each = 6 frames
    assert len(msgs) == 6

    # Check IDs and is_final flags
    for i, msg in enumerate(msgs):
        expected_seg = "w1" if i < 3 else "w2"
        expected_final = i % 3 == 2
        assert msg.segment_id == expected_seg
        assert msg.is_final is expected_final

    # Use pushed_duration() to verify each segment duration = 0.3s
    assert pytest.approx(emitter.pushed_duration(0), rel=1e-3) == 0.3
    assert pytest.approx(emitter.pushed_duration(1), rel=1e-3) == 0.3

    # --- Non‐streaming with a single WAV blob ---
    rx2 = aio.Chan[tts.SynthesizedAudio]()
    emitter2 = tts.AudioEmitter(label="wav-nos", dst_ch=rx2)
    emitter2.initialize(
        request_id="req-wav-nos",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/wav",
        frame_size_ms=100,
        stream=False,
    )

    # push one WAV blob, then flush() to mark final
    emitter2.push(wav_bytes)
    emitter2.flush()

    await emitter2.join()
    rx2.close()
    msgs2 = [msg async for msg in rx2]

    # Should split into 3 frames, last one is_final=True
    assert len(msgs2) == 3
    for i, msg in enumerate(msgs2):
        assert msg.is_final is (i == 2)

    # Duration via pushed_duration() = 0.3s
    assert pytest.approx(emitter2.pushed_duration(0), rel=1e-3) == 0.3

    # --- Injected silence on flush + end_segment ---
    rx3 = aio.Chan[tts.SynthesizedAudio]()
    emitter3 = tts.AudioEmitter(label="silence-test", dst_ch=rx3)
    emitter3.initialize(
        request_id="req-silence",
        sample_rate=1000,
        num_channels=1,
        mime_type="audio/pcm",
        frame_size_ms=100,
        stream=True,
    )

    # one chunk, flush to emit it, then immediately end_segment
    emitter3.start_segment(segment_id="s1")
    emitter3.push(b"\xff\xff" * 100)
    emitter3.flush()  # emits real frame, last_frame cleared
    emitter3.end_segment()  # should inject a 10ms silent frame with all-zero data
    emitter3.end_input()
    await emitter3.join()
    rx3.close()

    msgs3 = [msg async for msg in rx3]
    # first: real frame, second: injected silence
    assert len(msgs3) == 2
    # verify first is real
    assert msgs3[0].is_final is False
    assert msgs3[0].frame.data.tobytes() == b"\xff\xff" * 100
    # verify second is silence, 10ms = 10 samples @1000Hz
    silence = msgs3[1].frame.data.tobytes()
    assert msgs3[1].is_final is True
    assert silence == b"\x00\x00" * 10
