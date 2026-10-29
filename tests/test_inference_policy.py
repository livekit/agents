"""Operation policy survives public inference and fallback adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Never

import pytest

from livekit.agents import APIConnectOptions, APITimeoutError, inference, stt, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.audio import AudioBuffer

pytestmark = pytest.mark.unit


class _PolicyTTS(tts.TTS[Never]):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.calls: list[APIConnectOptions] = []

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        self.calls.append(conn_options)
        return _Batch(tts=self, input_text=text, conn_options=conn_options)


class _Batch(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="policy", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        output_emitter.push(b"\x00\x00" * 2400)
        output_emitter.flush()


class _PolicySTT(stt.STT[Never]):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self.calls: list[APIConnectOptions] = []

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language, conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        self.calls.append(conn_options)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT)

    def stream(self, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        self.calls.append(conn_options)
        return _Speech(stt=self, conn_options=conn_options)


class _Speech(stt.RecognizeStream):
    async def _run(self) -> None:
        async for _ in self.input:
            pass
        self.send(stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT))


@pytest.mark.parametrize("streaming", [False, True])
async def test_fallback_synthesis_delivers_with_each_calls_explicit_policy(streaming: bool) -> None:
    provider = _PolicyTTS()
    adapter = tts.FallbackAdapter([provider], max_retry_per_tts=0)
    first = APIConnectOptions(
        max_retry=2,
        pool_timeout=None,
        connect_timeout=0.75,
        write_timeout=0.5,
        read_timeout=0.125,
        total_timeout=8.0,
        retry_interval=0,
    )
    try:
        for conn in [first, replace(first, read_timeout=0.25)]:
            if streaming:
                async with adapter.stream(conn_options=conn) as source:
                    source.push_text("hello.")
                    source.end_input()
                    assert [audio async for audio in source]
            else:
                async with adapter.synthesize("hello", conn_options=conn) as source:
                    assert [audio async for audio in source]
        assert len(provider.calls) == 2
        for actual, read in zip(provider.calls, [0.125, 0.25], strict=True):
            assert actual.read_timeout == read
            assert actual.connect_timeout == 0.75
            assert actual.write_timeout == 0.5
            assert actual.pool_timeout is None
            assert actual.max_retry == 0
            assert actual.total_timeout is not None and 0 < actual.total_timeout <= 8.0
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_fallback_recognition_uses_the_call_policy(streaming: bool) -> None:
    provider = _PolicySTT()
    adapter = stt.FallbackAdapter([provider], attempt_timeout=10, max_retry_per_stt=0)
    conn = APIConnectOptions(max_retry=0, read_timeout=0.125, total_timeout=8)
    try:
        if streaming:
            async with adapter.stream(conn_options=conn) as source:
                source.end_input()
                assert [event async for event in source]
        else:
            assert await adapter.recognize([], conn_options=conn)
        assert provider.calls
        assert provider.calls[0].read_timeout == 0.125
        assert provider.calls[0].total_timeout is not None
        assert 0 < provider.calls[0].total_timeout <= 8
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("modality", ["stt", "tts"])
async def test_native_zero_call_deadline_expires_before_network_io(modality: str) -> None:
    conn = APIConnectOptions(max_retry=0, total_timeout=0)
    if modality == "tts":
        provider = inference.TTS(model="cartesia/sonic-3", api_key="k", api_secret="s")
        source = provider.synthesize("hello", conn_options=conn)
    else:
        provider = inference.STT(model="deepgram/nova-3", api_key="k", api_secret="s")
        source = provider.stream(conn_options=conn)
        source.end_input()
    try:
        with pytest.raises(APITimeoutError) as exc:
            async with source:
                async for _ in source:
                    pytest.fail("an expired call cannot produce output")
        assert exc.value.phase == "total"
    finally:
        await source.aclose()
        await provider.aclose()


@pytest.mark.parametrize("modality", ["stt", "tts"])
async def test_native_provider_default_applies_when_a_call_omits_options(modality: str) -> None:
    # No HTTP request is allowed: zero total is a policy outcome before launch.
    default = APIConnectOptions(max_retry=0, total_timeout=0)
    if modality == "tts":
        provider = inference.TTS(
            model="cartesia/sonic-3", api_key="k", api_secret="s", conn_options=default
        )
        source = provider.synthesize("hello")
    else:
        provider = inference.STT(
            model="deepgram/nova-3", api_key="k", api_secret="s", conn_options=default
        )
        source = provider.stream()
        source.end_input()
    try:
        with pytest.raises(APITimeoutError) as exc:
            async with source:
                async for _ in source:
                    pytest.fail("an expired default policy cannot produce output")
        assert exc.value.phase == "total"
    finally:
        await source.aclose()
        await provider.aclose()


@pytest.mark.parametrize("modality", ["stt", "tts"])
async def test_explicit_call_options_replace_a_native_providers_expired_default(
    modality: str,
) -> None:
    import asyncio
    import base64

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    connections = 0

    async def gateway(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            record = message.json()
            if record["type"] == "session.create":
                await socket.send_json({"type": "session.created", "session_id": "policy"})
            elif record["type"] == "session.flush":
                await socket.send_json(
                    {"type": "output_audio", "audio": base64.b64encode(b"\0\0" * 2400).decode()}
                )
                await socket.send_json({"type": "done"})
                await socket.close()
                return socket
            elif record["type"] == "session.finalize":
                await socket.send_json({"type": "final_transcript", "transcript": "hello"})
                await socket.send_json({"type": "session.finalized"})
                await socket.close()
                return socket
        return socket

    app = web.Application()
    app.router.add_get("/" + modality, gateway)
    async with TestServer(app) as server:
        kwargs = {
            "base_url": str(server.make_url("")).rstrip("/"),
            "api_key": "k",
            "api_secret": "s",
            "conn_options": APIConnectOptions(max_retry=0, total_timeout=0),
        }
        conn = APIConnectOptions(max_retry=0, total_timeout=1)
        if modality == "tts":
            provider = inference.TTS(model="cartesia/sonic-3", **kwargs)
            source = provider.synthesize("hello", conn_options=conn)
        else:
            provider = inference.STT(model="deepgram/nova-3", **kwargs)
            source = provider.stream(conn_options=conn)
            source.end_input()
        try:

            async def collect() -> list:
                async with source:
                    return [item async for item in source]

            assert await asyncio.wait_for(collect(), timeout=2)
            assert connections == 1
        finally:
            await source.aclose()
            await provider.aclose()


async def test_vad_adapter_preserves_the_utterances_recognition_retry_budget() -> None:
    from livekit.agents import (
        VAD,
        APIConnectionError,
        VADCapabilities,
        VADEvent,
        VADEventType,
        VADStream,
    )
    from livekit.agents.utils.audio import silence_frame

    class OneUtterance(VAD):
        def __init__(self):
            super().__init__(capabilities=VADCapabilities(update_interval=0.01))

        def stream(self):
            return Utterance(self)

    class Utterance(VADStream):
        async def _main_task(self):
            from livekit import rtc

            frames = [item async for item in self.input if isinstance(item, rtc.AudioFrame)]
            self.send(
                VADEvent(
                    type=VADEventType.END_OF_SPEECH,
                    samples_index=160,
                    timestamp=0.01,
                    speech_duration=0.01,
                    silence_duration=0,
                    frames=frames,
                )
            )

    class FlappingSTT(_PolicySTT):
        async def _recognize_impl(self, buffer, *, language, conn_options):
            event = await super()._recognize_impl(
                buffer, language=language, conn_options=conn_options
            )
            if len(self.calls) == 1:
                raise APIConnectionError("retry this utterance once")
            event.alternatives.append(stt.SpeechData(language="en", text="hello"))
            return event

    provider = FlappingSTT()
    adapter = stt.StreamAdapter(
        stt=provider,
        vad=OneUtterance(),
    )
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0, read_timeout=0.125)
        ) as source:
            source.push_frame(silence_frame(duration=0.01, sample_rate=16000))
            source.end_input()
            events = [event async for event in source]
        assert any(event.alternatives and event.alternatives[0].text == "hello" for event in events)
        assert len(provider.calls) == 2
        assert all(conn.read_timeout == 0.125 for conn in provider.calls)
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("explicit", [False, True])
async def test_eot_resolves_connection_policy_before_opening_the_gateway(explicit: bool) -> None:
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    connections = 0

    async def gateway(request: web.Request) -> web.Response:
        nonlocal connections
        connections += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get("/eot", gateway)
    async with TestServer(app) as server:
        provider = inference.TurnDetector(
            version="v1",
            base_url=str(server.make_url("")).rstrip("/"),
            api_key="k",
            api_secret="s",
            local_fallback=False,
            conn_options=APIConnectOptions(connect_timeout=0, max_retry=0),
        )
        source = provider.stream(
            **(
                {"conn_options": APIConnectOptions(connect_timeout=1, max_retry=0)}
                if explicit
                else {}
            )
        )
        try:
            prediction = await asyncio.wait_for(source.predict(), timeout=2)
            assert prediction.end_of_turn_probability == 1.0
            assert connections == int(explicit)
        finally:
            await source.aclose()


async def test_fallback_completion_forwards_the_call_policy() -> None:
    from livekit.agents import llm

    class Provider(llm.LLM[Never]):
        def __init__(self):
            super().__init__()
            self.calls: list[APIConnectOptions] = []

        def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
            self.calls.append(conn_options)
            return Tokens(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)

    class Tokens(llm.LLMStream):
        async def _run(self):
            self.send(llm.ChatChunk(id="policy", delta=llm.ChoiceDelta(content="hello")))

    provider = Provider()
    fallback = llm.FallbackAdapter([provider], attempt_timeout=5)
    conn = APIConnectOptions(max_retry=2, read_timeout=0.125, total_timeout=8)
    try:
        async with fallback.chat(chat_ctx=llm.ChatContext(), conn_options=conn) as source:
            assert (
                "".join([chunk.delta.content or "" async for chunk in source if chunk.delta])
                == "hello"
            )
        assert len(provider.calls) == 1
        assert provider.calls[0].read_timeout == 0.125
        assert provider.calls[0].total_timeout is not None
        assert 0 < provider.calls[0].total_timeout <= 8
        assert provider.calls[0].max_retry == 0
    finally:
        await fallback.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_native_tts_public_calls_have_one_retry_owner(streaming: bool) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from livekit.agents import APIError

    requests = 0

    async def gateway(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(status=503, text="controlled unavailable gateway")

    app = web.Application()
    app.router.add_get("/tts", gateway)
    async with TestServer(app) as server:
        provider = inference.TTS(
            model="cartesia/sonic-3",
            api_key="k",
            api_secret="s",
            base_url=str(server.make_url("")).rstrip("/"),
        )
        conn = APIConnectOptions(max_retry=1, retry_interval=0, total_timeout=2)
        source = (
            provider.stream(conn_options=conn)
            if streaming
            else provider.synthesize(
                "hello",
                conn_options=conn,
            )
        )
        try:
            with pytest.raises(APIError):
                async with source:
                    if streaming:
                        source.push_text("hello")
                        source.end_input()
                    async for _ in source:
                        pytest.fail("an unavailable gateway cannot produce audio")
            assert requests == 2
        finally:
            await source.aclose()
            await provider.aclose()


@pytest.mark.parametrize("failure_after_audio", [False, True])
async def test_native_tts_replays_input_only_before_audio_is_delivered(
    failure_after_audio: bool,
) -> None:
    import asyncio
    import base64
    from contextlib import nullcontext

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from livekit.agents import APIError

    commands: list[list[dict]] = []
    delivered = asyncio.Event()
    pcm = b"\x00\x00" * 2400

    async def gateway(request: web.Request) -> web.WebSocketResponse:
        observed: list[dict] = []
        commands.append(observed)
        attempt = len(commands)
        socket = web.WebSocketResponse(timeout=0.1)
        await socket.prepare(request)
        async for message in socket:
            record = message.json()
            observed.append(record)
            if record["type"] != "session.flush":
                continue
            if attempt > 1 or failure_after_audio:
                await socket.send_json(
                    {"type": "output_audio", "audio": base64.b64encode(pcm).decode()}
                )
            if attempt > 1:
                await socket.send_json({"type": "done"})
            elif failure_after_audio:
                await delivered.wait()
            await socket.close()
            return socket
        return socket

    app = web.Application()
    app.router.add_get("/tts", gateway)
    async with TestServer(app) as server:
        provider = inference.TTS(
            model="cartesia/sonic-3",
            api_key="k",
            api_secret="s",
            base_url=str(server.make_url("")).rstrip("/"),
        )
        frames = []
        try:
            expected_error = pytest.raises(APIError) if failure_after_audio else nullcontext()
            with expected_error:
                async with provider.stream(
                    conn_options=APIConnectOptions(max_retry=1, retry_interval=0, total_timeout=2),
                ) as source:
                    source.push_text("hello")
                    source.end_input()
                    async for audio in source:
                        frames.append(audio.frame)
                        delivered.set()
            received = b"".join(bytes(frame.data) for frame in frames)
            if failure_after_audio:
                assert 0 < len(received) <= len(pcm)
                assert received == pcm[: len(received)]
            else:
                assert received == pcm
            assert len(commands) == (1 if failure_after_audio else 2)
            for observed in commands:
                assert [record["type"] for record in observed] == [
                    "session.create",
                    "input_transcript",
                    "session.flush",
                ]
                assert observed[1]["transcript"] == "hello "
        finally:
            await provider.aclose()
