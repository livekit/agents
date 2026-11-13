"""ElevenLabs TTS and text-to-dialogue against a local peer speaking the ElevenLabs JSON protocol."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, TypedDict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from typing_extensions import Unpack

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http, tts
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins.elevenlabs import TTS, TTSEncoding, VoiceSettings

PCM = b"\x0b\x00" * 4800
OPTIONS = APIConnectOptions(max_retry=0, read_timeout=0.5)
# a provider gone quiet times out at once
QUICK_TIMEOUT = APIConnectOptions(max_retry=0, read_timeout=0.05)
# chunks paced under the read timeout, together well over it
PACED, PACING = APIConnectOptions(max_retry=0, read_timeout=0.2), 0.08
FLASH, DIALOGUE = "eleven_flash_v2_5", "eleven_v3"
PCM_RATES: dict[int, TTSEncoding] = {16000: "pcm_16000", 24000: "pcm_24000", 44100: "pcm_44100"}


@asynccontextmanager
async def peer(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[tuple[str, http.HttpSession]]:
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    async with TestServer(app) as server, http.open() as session:
        yield str(server.make_url("")).rstrip("/"), session


async def collect(stream: AsyncIterable[tts.SynthesizedAudio]) -> list[tts.SynthesizedAudio]:
    async with asyncio.timeout(2):
        return [ev async for ev in stream]


def audio(frames: list[tts.SynthesizedAudio]) -> bytes:
    return b"".join(bytes(ev.frame.data) for ev in frames)


def assert_pcm(frames: list[tts.SynthesizedAudio], expected: bytes = PCM) -> None:
    """The frames carry `expected`, padded with at most 200 ms of silence."""
    pcm = audio(frames)
    assert pcm[: len(expected)] == expected
    assert not any(pcm[len(expected) :])
    assert len(pcm) - len(expected) < frames[0].frame.sample_rate * 2 // 10


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def final(model: str) -> str:
    return "is_final" if model == DIALOGUE else "isFinal"


class StreamOptions(TypedDict, total=False):
    chunk_length_schedule: list[int]
    auto_mode: bool


async def speak_one(
    text: str, model: str = FLASH, **options: Unpack[StreamOptions]
) -> tuple[list[tts.SynthesizedAudio], list[dict[str, object]]]:
    """Stream `text` to a peer that answers the closed context with PCM; frames and packets."""
    packets: list[dict[str, object]] = []

    async def speak(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                await ws.send_json({**ctx, "audio": b64(PCM)})
                await ws.send_json({**ctx, final(model): True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key",
            model=model,
            encoding="pcm_24000",
            base_url=url,
            http_session=session,
            **options,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text(text)
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()
    return frames, packets


@pytest.mark.parametrize("model", ["eleven_turbo_v2_5", "eleven_v3", "eleven_v3_conversational"])
async def test_synthesis_routes_models_and_preserves_fragmented_input(
    model: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Each model reaches its endpoint with whole words, phonemes intact.

    https://github.com/livekit/agents/pull/6838
    """
    paths, packets = [], []
    dialogue = model.startswith("eleven_v3")

    async def speak(request: web.Request) -> web.StreamResponse:
        paths.append(request.path)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                await ws.send_json({**ctx, "audio": "", "alignment": None})
                await ws.send_json({**ctx, "audio": b64(PCM)})
                await ws.send_json({**ctx, "is_final" if dialogue else "isFinal": True})
        return ws

    fragments = ["Hello ", "ref", "erence."]
    if not dialogue:
        fragments = ["Hello ", "ref", "erence ", '<phoneme alphabet="ipa" ph="tɛst">', "test"]
        fragments += ["</phoneme>", "."]
    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key",
            voice_id="custom-voice",
            model=model,
            encoding="pcm_24000",
            base_url=url,
            http_session=session,
            enable_ssml_parsing=not dialogue,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            for text in fragments:
                stream.push_text(text)
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    assert_pcm(frames)
    assert paths == [
        "/text-to-dialogue/multi-stream-input"
        if dialogue
        else "/text-to-speech/custom-voice/multi-stream-input"
    ]
    texts = [i["text"] for p in packets for i in p.get("inputs", [])] + [
        p["text"] for p in packets if "text" in p
    ]
    assert "reference" in "".join(texts)
    assert not any(t.strip() in ("ref", "erence") for t in texts)
    assert dialogue or any("<phoneme" in t and "</phoneme>" in t for t in texts)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_messages_for_an_unknown_context_are_ignored() -> None:
    """A flush_done for an already-closed context does not disturb the current one.

    https://github.com/livekit/agents/pull/5813
    """

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                await ws.send_json(
                    {"context_id": "already_closed", "flush_done": True, "status_code": 206}
                )
                await ws.send_json({**ctx, "audio": b64(PCM), "isFinal": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", encoding="pcm_24000", base_url=url, http_session=session)
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    assert_pcm(frames)


async def test_chunk_length_schedule_reaches_the_context_init_packet() -> None:
    """https://github.com/livekit/agents/issues/5007"""
    _, packets = await speak_one("Hello.", chunk_length_schedule=[80, 120], auto_mode=False)

    assert packets[0]["generation_config"] == {"chunk_length_schedule": [80, 120]}


async def test_dialogue_alignment_becomes_timed_words() -> None:
    """Dialogue alignment reaches the frames as a timed transcript. https://github.com/livekit/agents/pull/6838"""

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context"):
                alignment = {
                    "chars": list("hi "),
                    "char_start_times_ms": [0, 100, 200],
                    "char_durations_ms": [100, 100, 100],
                }
                ctx = {"context_id": packet["context_id"]}
                await ws.send_json({**ctx, "audio": b64(PCM), "alignment": alignment})
                await ws.send_json({**ctx, "is_final": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key",
            model=DIALOGUE,
            encoding="pcm_24000",
            preferred_alignment="original",
            base_url=url,
            http_session=session,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("hi ")
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    words = [w for ev in frames for w in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    assert [w.text.strip() for w in words] == ["hi"]


@pytest.mark.parametrize("rate", [16000, 24000, 44100])
@pytest.mark.parametrize("model", ["eleven_turbo_v2", "eleven_v3"])
async def test_pcm_request_rate_matches_emitted_duration(rate: int, model: str) -> None:
    """PCM output uses the requested sample rate without pitch changes. https://github.com/livekit/agents/issues/4138"""
    pcm = b"\x05\x00" * (rate // 5)
    requests = []

    async def speak(request: web.Request) -> web.StreamResponse:
        requests.append((request.path, request.query["output_format"], await request.json()))
        return web.Response(body=pcm, content_type="audio/pcm")

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key", model=model, encoding=PCM_RATES[rate], base_url=url, http_session=session
        )
        async with provider.synthesize("Hello.", conn_options=OPTIONS) as stream:
            frames = await collect(stream)
        await provider.aclose()

    [(path, output_format, body)] = requests
    assert path == (
        "/text-to-dialogue/stream"
        if model == "eleven_v3"
        else "/text-to-speech/hpp4J3VqNfWAUOO0d1Us/stream"
    )
    assert (output_format, body["model_id"]) == (f"pcm_{rate}", model)
    assert_pcm(frames, pcm)
    assert {ev.frame.sample_rate for ev in frames} == {rate}
    assert sum(ev.frame.duration for ev in frames) == pytest.approx(0.2, abs=0.05)


async def test_rejected_synthesis_raises_the_provider_status() -> None:
    """A 401 reaches the caller with the provider's message. https://github.com/livekit/agents/issues/5720"""

    async def reject(request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {"detail": {"code": "invalid_api_key", "message": "rejected key"}}, status=401
        )

    async with peer(reject) as (url, session):
        provider = TTS(api_key="key", encoding="pcm_24000", base_url=url, http_session=session)
        async with provider.synthesize("Hello.", conn_options=OPTIONS) as stream:
            with pytest.raises(APIStatusError) as error:
                await collect(stream)
        await provider.aclose()

    assert error.value.status_code == 401
    assert "rejected key" in str(error.value)


async def test_non_audio_response_is_an_error() -> None:
    """A JSON body where audio was expected fails the synthesis. https://github.com/livekit/agents/pull/567"""

    async def speak(request: web.Request) -> web.StreamResponse:
        return web.json_response({"detail": "not audio"})

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", encoding="pcm_24000", base_url=url, http_session=session)
        async with provider.synthesize("Hello.", conn_options=OPTIONS) as stream:
            with pytest.raises(APIError, match="non-audio"):
                await collect(stream)
        await provider.aclose()


async def test_socket_closed_mid_synthesis_raises_its_close_code() -> None:
    """https://github.com/livekit/agents/pull/4182"""
    closed = asyncio.Event()

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.close(code=1011)
        closed.set()
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", encoding="pcm_24000", base_url=url, http_session=session)
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            with pytest.raises(APIStatusError) as error:
                await collect(stream)
        await provider.aclose()

    assert error.value.status_code == 1011
    assert closed.is_set()


@pytest.mark.parametrize("model", [FLASH, DIALOGUE])
async def test_provider_error_reaches_the_caller(model: str) -> None:
    """A context error fails the synthesis with the provider's diagnostic. https://github.com/livekit/agents/pull/6838"""

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        packet = json.loads((await ws.receive()).data)
        await ws.send_json({"context_id": packet["context_id"], "error": "voice diagnostic"})
        async for _ in ws:
            pass
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key", model=model, encoding="pcm_24000", base_url=url, http_session=session
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            with pytest.raises(APIError) as error:
                await collect(stream)
        await provider.aclose()

    assert "voice diagnostic" in str(error.value.__cause__)


@pytest.mark.parametrize("audio_first", [False, True])
@pytest.mark.parametrize("model", [FLASH, DIALOGUE])
async def test_silence_after_the_last_input_times_out(model: str, audio_first: bool) -> None:
    """A provider that goes quiet before finishing the context times out the read.

    https://github.com/livekit/agents/pull/7627
    """

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context") and audio_first:
                await ws.send_json({"context_id": packet["context_id"], "audio": b64(PCM)})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key", model=model, encoding="pcm_24000", base_url=url, http_session=session
        )
        async with provider.stream(conn_options=QUICK_TIMEOUT) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            with pytest.raises(APIError) as error:
                await collect(stream)
        await provider.aclose()

    assert "timed out" in str(error.value.__cause__)


@pytest.mark.parametrize("model", [FLASH, DIALOGUE])
async def test_audio_arriving_slower_than_the_whole_read_timeout_completes(model: str) -> None:
    """Each audio chunk restarts the read timeout. https://github.com/livekit/agents/pull/7627"""
    chunk = b"\x0b\x00" * 2400

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                for _ in range(4):
                    await asyncio.sleep(PACING)  # the provider's pacing, under the read timeout
                    await ws.send_json({**ctx, "audio": b64(chunk)})
                await ws.send_json({**ctx, final(model): True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key", model=model, encoding="pcm_24000", base_url=url, http_session=session
        )
        async with provider.stream(conn_options=PACED) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    assert_pcm(frames, chunk * 4)


@pytest.mark.parametrize("updated", [False, True])
async def test_empty_input_flush_finishes_after_option_update(updated: bool) -> None:
    """A textless stream finishes after flush, also after an option update. https://github.com/livekit/agents/pull/4499"""
    flushed = []

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("flush"):
                flushed.append(packet)
            if packet.get("close_context"):
                await ws.send_json({"context_id": packet["context_id"], "isFinal": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", base_url=url, http_session=session, encoding="pcm_24000")
        if updated:
            provider.update_options(voice_id="updated-voice")
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.flush()
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    assert frames == []
    assert flushed


async def test_interrupted_stream_still_releases_its_context_and_drops_its_late_audio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An interrupted synthesis closes its context; audio flushed after it goes nowhere.

    https://github.com/livekit/agents/issues/6929
    """
    first_text = asyncio.Event()
    closed_contexts: list[str] = []
    late = b"\x7f\x00" * 2400

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = {"context_id": packet["context_id"]}
            if packet.get("text", " ") != " ":
                first_text.set()
            if packet.get("close_context"):
                closed_contexts.append(packet["context_id"])
                if len(closed_contexts) == 1:  # the interrupted one
                    await ws.send_json({**ctx, "audio": b64(late), "isFinal": True})
                else:
                    await ws.send_json({**ctx, "audio": b64(PCM), "isFinal": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", encoding="pcm_24000", base_url=url, http_session=session)
        interrupted = provider.stream(conn_options=OPTIONS)
        interrupted.push_text("Hello there. ")
        interrupted.flush()
        await asyncio.wait_for(first_text.wait(), 2)
        await interrupted.aclose()

        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Next.")
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    assert len(closed_contexts) == 2
    assert_pcm(frames)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_prewarm_opens_the_socket_before_the_first_synthesis() -> None:
    """https://github.com/livekit/agents/issues/7582"""
    connected = asyncio.Event()

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connected.set()
        async for _ in ws:
            pass
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", base_url=url, http_session=session)
        provider.prewarm()
        await asyncio.wait_for(connected.wait(), 2)
        await provider.aclose()


async def test_dialogue_prewarm_reconnects_when_the_idle_socket_closes() -> None:
    """The server closes idle dialogue sockets; prewarm opens a new one. https://github.com/livekit/agents/issues/7582"""
    connections = 0
    reconnected = asyncio.Event()

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if connections == 1:
            await ws.close()
            return ws
        reconnected.set()
        async for _ in ws:
            pass
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", model=DIALOGUE, base_url=url, http_session=session)
        provider.prewarm()
        await asyncio.wait_for(reconnected.wait(), 2)
        await provider.aclose()

    assert connections == 2


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/8464 reproduces: empty final response leaves failed socket reusable",
)
async def test_failed_empty_synthesis_retires_socket() -> None:
    """An empty failed synthesis retires its socket before the next request. https://github.com/livekit/agents/issues/8464"""
    connections = 0

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                if connections > 1:
                    await ws.send_json({**ctx, "audio": b64(PCM)})
                await ws.send_json({**ctx, "isFinal": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", base_url=url, http_session=session, encoding="pcm_24000")
        try:
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_text("Hello.")
                stream.end_input()
                with pytest.raises(APIError):
                    await collect(stream)
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_text("Hello.")
                stream.end_input()
                assert_pcm(await collect(stream))
        finally:
            await provider.aclose()

    assert connections == 2


@pytest.mark.parametrize("alignment", ["normalized", "original"])
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7773 reproduces: dialogue alignment restarts on each audio chunk",
)
async def test_dialogue_alignment_offsets_across_chunks(
    alignment: Literal["normalized", "original"],
) -> None:
    """Dialogue word timings increase across chunk-relative provider alignments. https://github.com/livekit/agents/issues/7773"""

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet.get("close_context"):
                ctx = {"context_id": packet["context_id"]}
                for word in ["Hello ", "world "]:
                    chars = {
                        "chars": list(word),
                        "char_start_times_ms": [i * 20 for i in range(len(word))],
                        "char_durations_ms": [20] * len(word),
                    }
                    await ws.send_json({**ctx, "audio": b64(PCM), "alignment": chars})
                await ws.send_json({**ctx, "is_final": True})
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key",
            model=DIALOGUE,
            base_url=url,
            http_session=session,
            encoding="pcm_24000",
            preferred_alignment=alignment,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Hello world ")
            stream.end_input()
            frames = await collect(stream)
        await provider.aclose()

    words = [w for ev in frames for w in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    assert [w.text.strip() for w in words] == ["Hello", "world"]
    assert words[1].start_time >= 0.2
    assert words[1].start_time >= words[0].end_time


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/4221 reproduces: early isFinal drops text supplied after a pause",
)
async def test_input_after_early_final_is_not_lost() -> None:
    """Provider final before input ends cannot discard later text. https://github.com/livekit/agents/issues/4221"""
    first_sent = asyncio.Event()
    texts = []

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            text = packet.get("text", "").strip()
            if text:
                texts.append(text)
                await ws.send_json(
                    {"context_id": packet["context_id"], "audio": b64(PCM), "isFinal": True}
                )
                first_sent.set()
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", base_url=url, http_session=session, encoding="pcm_24000")
        try:
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_text("Before ")
                stream.flush()
                await asyncio.wait_for(first_sent.wait(), 1)
                first = await asyncio.wait_for(anext(stream), 1)
                # receiving a frame yields to the provider's final processing before more input
                await asyncio.sleep(0)
                stream.push_text("After ")
                stream.end_input()
                frames = [first, *await collect(stream)]
        finally:
            await provider.aclose()

    assert "After" in texts
    assert_pcm(frames, PCM + PCM)


async def test_abandoned_stream_and_provider_close_release_the_socket() -> None:
    """Abandoned streams and provider shutdown close the socket. https://github.com/livekit/agents/pull/1468"""
    received = asyncio.Event()
    closed = asyncio.Event()

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for _ in ws:
                received.set()
        finally:
            closed.set()
        return ws

    async with peer(speak) as (url, session):
        provider = TTS(api_key="key", base_url=url, http_session=session)
        stream = provider.stream(conn_options=OPTIONS)
        stream.push_text("Hello ")
        stream.flush()
        await asyncio.wait_for(received.wait(), 1)
        await asyncio.wait_for(stream.aclose(), 1)
        await provider.aclose()
        await asyncio.wait_for(closed.wait(), 1)

        assert not session.closed


async def test_updated_voice_settings_replace_the_previous_ones() -> None:
    """Each synthesis uses the latest voice settings, with no null fields.

    https://github.com/livekit/agents/issues/6027
    """
    packets, queries = [], []

    async def speak(request: web.Request) -> web.StreamResponse:
        queries.append(dict(request.query))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            if packet.get("close_context"):
                await ws.send_json(
                    {"context_id": packet["context_id"], "audio": b64(PCM), "isFinal": True}
                )
        return ws

    sent_settings = []
    async with peer(speak) as (url, session):
        provider = TTS(
            api_key="key",
            base_url=url,
            http_session=session,
            encoding="pcm_24000",
            apply_text_normalization="off",
        )
        for stability in [0.3, 0.7]:
            provider.update_options(
                voice_settings=VoiceSettings(stability=stability, similarity_boost=0.6)
            )
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_text("Hello ")
                stream.end_input()
                assert_pcm(await collect(stream))
            sent_settings.append(next(p for p in reversed(packets) if "voice_settings" in p))
        await provider.aclose()

    assert [s["voice_settings"]["stability"] for s in sent_settings] == [0.3, 0.7]
    assert all(v is not None for s in sent_settings for v in s["voice_settings"].values())
    assert queries[-1]["apply_text_normalization"] == "off"
