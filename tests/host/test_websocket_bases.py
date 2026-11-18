"""The socket bases: a plugin's typed protocol against a loopback provider, one fake per kind
(``tests/type_checks/websocket_protocols.py``, which strict mypy also checks)."""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar, NotRequired, TypedDict

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import Agent, APIConnectOptions, APIError, VoiceSession, _core, stt, tts
from livekit.agents._websocket import _Decoder
from tests.fakes import until
from tests.type_checks.websocket_protocols import (
    CodecSttStream,
    LoopbackMux,
    LoopbackRealtime,
    LoopbackRealtimeSession,
    LoopbackStt,
    LoopbackSttStream,
    LoopbackTts,
    LoopbackTtsStream,
    event_frame,
)

NO_RETRY = APIConnectOptions(max_retry=0)

Handler = Callable[[web.WebSocketResponse, web.Request], Awaitable[None]]


@pytest.fixture
async def peer() -> AsyncIterator[Callable[[Handler], Awaitable[str]]]:
    """Serves one handler per test on a loopback port; returns its ``ws://`` URL."""
    runners: list[web.AppRunner] = []

    async def serve(handler: Handler, headers: dict[str, str] | None = None) -> str:
        async def accept(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            ws.headers.update(headers or {})
            await ws.prepare(request)
            await handler(ws, request)
            return ws

        app = web.Application()
        app.router.add_get("/ws", accept)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"ws://127.0.0.1:{port}/ws"

    yield serve
    for runner in runners:
        await runner.cleanup()


def audio(ms: int) -> rtc.AudioFrame:
    samples = 16 * ms
    return rtc.AudioFrame(b"\x00\x01" * samples, 16000, 1, samples)


async def events(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    return [event async for event in stream]


# Catches: hand-written send/receive loops (the goodbye never sent, the provider's last answer
# dropped at the input's end), audio sent in the caller's frame size, JSON written by hand.
async def test_stt_stream_sends_typed_commands_and_reads_until_the_provider_finishes(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    seen: list[str] = []
    query: dict[str, str] = {}

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        query.update(request.query)
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.BINARY:
                seen.append(f"<{len(msg.data)} bytes>")
                continue
            command = json.loads(msg.data)
            seen.append(command["type"])
            if command["type"] == "finalize":
                await ws.send_json({"type": "transcript", "text": "hello", "final": True})
            elif command["type"] == "close":
                await ws.send_json({"type": "finished"})

    url = await peer(provider)
    stream = LoopbackStt(url).stream(conn_options=NO_RETRY)
    for _ in range(10):
        stream.push_frame(audio(10))
    stream.end_input()
    got = await events(stream)
    finals = [e for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hello"]
    assert seen == ["<1600 bytes>", "<1600 bytes>", "finalize", "close"]
    assert query == {"encoding": "linear16"}


# Catches: a drain that runs out while a speech is still open (its final never came) ending the
# stream cleanly without an error, or losing the words its interim heard.
async def test_an_open_speech_at_the_drains_read_limit_fails_with_a_read_timeout(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    seen: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                seen.append(json.loads(msg.data)["type"])
                if seen[-1] == "finalize":
                    await ws.send_json({"type": "transcript", "text": "hel", "final": False})

    read_timeout = 5.0
    stream = LoopbackStt(await peer(provider)).stream(
        conn_options=APIConnectOptions(max_retry=0, read_timeout=read_timeout)
    )
    got: list[stt.SpeechEvent] = []

    async def read() -> None:
        async for event in stream:
            got.append(event)

    async with stream:
        reader = asyncio.create_task(read())
        stream.push_frame(audio(100))
        stream.end_input()
        await until(lambda: "close" in seen, "the goodbye is sent")
        await until(lambda: stream.speaking, "the interim opens a speech")
        _core.skip_host_time(read_timeout)
        await until(reader.done, "the stream ends")
        with pytest.raises(APIError) as failed:
            reader.result()
    assert failed.value.kind == "timeout"
    finals = [e for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hel"]


# Catches: a stream failed by what the plugin never reads: a new event type the provider added,
# a tag that is a list or an object (an uncaught TypeError), a field it sends beyond the declared
# ones, or a value it added to a field the plugin reads.
async def test_unknown_message_types_are_skipped_and_extra_fields_ignored(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for kind in _commands(ws, []):
            if kind == "close":
                await ws.send_json({"type": "speech_started_v2", "at": 1.5})
                await ws.send_json({"type": "speech_started_v2", "at": 2.5})
                await ws.send_json({"type": ["transcript"]})
                await ws.send_json({"type": {"name": "transcript"}})
                transcript = {"type": "transcript", "text": "hello", "final": True}
                await ws.send_json({**transcript, "words": [], "meta": {"x": 1}})
                await ws.send_json({"type": "finished", "reason": "drained", "request_id": "r-1"})

    stream = LoopbackStt(await peer(provider)).stream(conn_options=NO_RETRY)
    stream.push_frame(audio(50))
    stream.end_input()
    got = await events(stream)
    finals = [e for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["hello"]


class _Untagged(TypedDict):
    text: NotRequired[str]


class _UntaggedRequired(TypedDict):
    text: str


class _TypedFields(TypedDict):
    final: NotRequired[bool]
    confidence: NotRequired[float]


@pytest.mark.parametrize("frame", ['{"final": "false"}', '{"confidence": "0.9"}'])
def test_read_fields_are_validated_without_coercing_the_provider_payload(frame: str) -> None:
    with pytest.raises(APIError) as failed:
        _Decoder(_TypedFields).decode(frame, "loopback")
    assert failed.value.kind == "protocol"


# Catches: an untagged message type with a required field, which fails a stream with protocol on
# the first message of a shape the provider added; an all-optional one decodes any object and
# skips a JSON value that is no object.
def test_an_untagged_inbound_declares_every_field_optional() -> None:
    with pytest.raises(TypeError):
        _Decoder(_UntaggedRequired)
    decoder = _Decoder(_Untagged)
    assert decoder.decode('{"event": "new"}', "loopback") == {}
    assert decoder.decode("[1, 2]", "loopback") is None
    with pytest.raises(APIError) as failed:
        decoder.decode('{"text": 5}', "loopback")
    assert failed.value.kind == "protocol"


# Catches: a declared message whose read field is malformed logged and dropped, or retried as if
# the next connection would answer differently.
async def test_a_declared_message_with_a_wrong_typed_field_fails_with_protocol(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        await ws.send_json({"type": "transcript", "text": 5, "final": True})
        await ws.receive()

    stream = LoopbackStt(await peer(provider)).stream(conn_options=NO_RETRY)
    with pytest.raises(APIError) as failed:
        await events(stream)
    assert (failed.value.kind, failed.value.retryable) == ("protocol", False)
    assert '"text": 5' in (failed.value.detail or "")


# Catches: a binary-framed protocol (AWS event-stream) forced through JSON validation, or its
# signed URL computed once instead of per connection.
async def test_a_binary_codec_decodes_its_own_frames(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    received: list[str] = []
    query: dict[str, str] = {}

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        query.update(request.query)
        msg = await ws.receive()
        (length,) = struct.unpack(">I", msg.data[:4])
        received.append(json.loads(msg.data[4 : 4 + length])[":event-type"])
        await ws.send_bytes(event_frame({":message-type": "event"}, b"hi there"))
        await ws.send_bytes(event_frame({":message-type": "exception"}, b"throttled"))
        await ws.receive()

    stt_ = LoopbackStt(await peer(provider))
    stream = CodecSttStream(stt=stt_, conn_options=NO_RETRY)
    stream.push_frame(audio(10))
    got: list[stt.SpeechEvent] = []
    with pytest.raises(APIError) as failed:
        async for event in stream:
            got.append(event)
    assert [e.alternatives[0].text for e in got if e.alternatives] == ["hi there"]
    assert failed.value.detail == "throttled"
    assert received == ["AudioEvent"]
    assert query == {"X-Amz-Signature": "fresh"}


# Catches: each multiplexed stream opening its own socket, or audio reaching the wrong stream.
async def test_tts_streams_share_the_multiplexed_socket(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    connections = 0

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        nonlocal connections
        connections += 1
        async for msg in ws:
            command = json.loads(msg.data)
            context = command["context_id"]
            if command.get("final"):
                await ws.send_json({"type": "done", "context_id": context})
            elif text := command.get("text"):
                tone = b"\x01\x00" if text.startswith("one") else b"\x02\x00"
                await ws.send_bytes(context.encode() + b"|" + tone * 160)

    tts_ = LoopbackTts(await peer(provider))

    async def say(text: str) -> bytes:
        stream = tts_.stream(conn_options=NO_RETRY)
        stream.push_text(text)
        stream.end_input()
        return b"".join([bytes(audio.frame.data) async for audio in stream])

    one, two = await asyncio.gather(say("one fish."), say("two fish."))
    assert one and set(one[::2]) == {1}
    assert two and set(two[::2]) == {2}
    assert connections == 1
    await tts_.mux.aclose()


# Catches: a later segment's audio played before the segment ahead of it finished, or segments
# merged into one (the flush lost), when the provider answers them out of order.
async def test_tts_segments_play_in_order_whatever_order_the_provider_answers(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    ended: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            command = json.loads(msg.data)
            if command.get("final"):
                ended.append(command["context_id"])
            if len(ended) == 2:  # answer the second segment first
                for context, tone in zip(reversed(ended), (b"\x02\x00", b"\x01\x00"), strict=True):
                    await ws.send_bytes(context.encode() + b"|" + tone * 160)
                    await ws.send_json({"type": "done", "context_id": context})

    tts_ = LoopbackTts(await peer(provider))
    stream = tts_.stream(conn_options=NO_RETRY)
    stream.push_text("one fish.")
    stream.flush()
    stream.push_text("two fish.")
    stream.end_input()
    heard = [(audio.segment_id, bytes(audio.frame.data)) async for audio in stream]
    segments = list(dict.fromkeys(segment for segment, _ in heard))
    assert segments == ended
    tones = [data[0] for _, data in heard if any(data)]
    assert tones == sorted(tones) and set(tones) == {1, 2}
    await tts_.mux.aclose()


# Catches: a realtime plugin's own reconnect loop and ping answering, and a change applied to
# the live session instead of a new one.
async def test_realtime_answers_protocol_frames_and_reconnects_on_change(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    instructions: list[str] = []
    seen: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        instructions.append(request.query["instructions"])
        await ws.send_json({"type": "ping"})
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.BINARY:
                seen.append("audio")
                continue
            command = json.loads(msg.data)
            seen.append(command["type"])
            if command["type"] == "say":
                await ws.send_json({"type": "said", "text": command["text"]})

    model = LoopbackRealtime(await peer(provider), NO_RETRY)
    session = model.session()
    try:
        await until(lambda: "pong" in seen, "the ping answered")
        session.push_audio(audio(10))
        session.generate_reply(instructions="hello")
        await until(lambda: session.heard == ["hello"], "the reply heard")
        await session.update_instructions("be brief")
        await until(lambda: instructions == ["", "be brief"], "a new session")
        # a control message goes now, ahead of input not yet sent
        assert seen[0] == "pong" and sorted(seen[1:3]) == ["audio", "say"]
    finally:
        await session.aclose()


class LimitedRealtime(LoopbackRealtime):
    """Its sessions are replaced after a minute."""

    def __init__(self, url: str, conn_options: APIConnectOptions) -> None:
        super().__init__(url, conn_options)
        self.sessions: list[LoopbackRealtimeSession] = []

    def session(self, *, turn_detection_disabled: bool = False) -> LoopbackRealtimeSession:
        session = LoopbackRealtimeSession(self)
        session.max_session_duration = 60.0
        # a skip past the limit steps through each ping's answer deadline before a real socket
        # could carry the pong
        session.ping_interval = None
        self.sessions.append(session)
        return session


async def test_a_reply_requested_after_reconnect_goes_on_the_new_connection(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    opened: list[str] = []
    seen: list[tuple[str, str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        instructions = request.query["instructions"]
        opened.append(instructions)
        async for message in ws:
            if message.type is aiohttp.WSMsgType.TEXT:
                packet = json.loads(message.data)
                seen.append((instructions, packet["type"]))
                if packet["type"] == "say":
                    await ws.send_json({"type": "said", "text": "hello"})
                    await ws.send_json({"type": "done"})

    session = LoopbackRealtime(await peer(provider), NO_RETRY).session()
    try:
        await until(lambda: opened == [""], "the first connection opens")
        await session.update_instructions("new instructions")
        request = session.generate_reply(instructions="hello")
        await until(
            lambda: session.heard == ["hello"], "the reply requested after reconnect arrives"
        )
        await request
        assert seen == [("new instructions", "say")]
    finally:
        await session.aclose()


# Catches: a realtime session past its limit replaced while a reply is still generating, which
# cuts the reply the person is hearing.
async def test_a_realtime_replacement_waits_for_the_reply_in_progress(
    peer: Callable[[Handler], Awaitable[str]],
) -> None:
    sockets: list[web.WebSocketResponse] = []
    asked: asyncio.Queue[web.WebSocketResponse] = asyncio.Queue()
    pongs: asyncio.Queue[None] = asyncio.Queue()

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        sockets.append(ws)
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                kind = json.loads(msg.data)["type"]
                if kind == "say":
                    asked.put_nowait(ws)
                elif kind == "pong":
                    pongs.put_nowait(None)

    async def ping(ws: web.WebSocketResponse) -> None:
        await ws.send_json({"type": "ping"})
        async with asyncio.timeout(2):
            await pongs.get()

    model = LimitedRealtime(await peer(provider), NO_RETRY)
    async with VoiceSession(llm=model, vad=None) as s:
        await s.start(Agent(instructions="Help."))
        reply = s.generate_reply(instructions="greet")
        async with asyncio.timeout(2):
            ws = await asked.get()
        await ws.send_json({"type": "said", "text": "Hello, "})
        await until(lambda: model.sessions[-1].heard == ["Hello, "], "the reply began")
        connections = len(sockets)
        _core.skip_host_time(60)
        # each message received asks whether the connection can be replaced; twice, so the
        # first answer was given
        await ping(ws)
        await ping(ws)
        assert len(sockets) == connections and not ws.closed
        await ws.send_json({"type": "said", "text": "how can I help?"})
        await ws.send_json({"type": "done"})
        await reply
        # the generation ends once its streams are read out, just after its playout: each
        # message asks again
        async with asyncio.timeout(2):
            while len(sockets) == connections:
                if not ws.closed:
                    await ws.send_json({"type": "ping"})
                await asyncio.sleep(0.01)

    assert ("assistant", "Hello, how can I help?") in [
        (i.role, i.text_content) for i in s.history.items if i.type == "message"
    ]


# The kinds' own options, each against a loopback provider.


async def _commands(ws: web.WebSocketResponse, seen: list[str]) -> AsyncIterator[str]:
    """Each message the plugin sends, recorded: binary audio as ``audio``, a command by type."""
    async for msg in ws:
        kind = "audio" if msg.type is aiohttp.WSMsgType.BINARY else json.loads(msg.data)["type"]
        seen.append(kind)
        yield kind


def _final(text: str) -> dict[str, object]:
    return {"type": "transcript", "text": text, "final": True}


class SessionPerFlushStt(LoopbackSttStream):
    new_session_per_flush = True


# Catches: a provider that ends its session at each flush (rtzr, gladia) read as a lost
# connection, the audio after the flush sent on the finished session, or its last transcript cut.
async def test_stt_opens_a_new_session_after_each_flush(
    peer: Callable[..., Awaitable[str]],
) -> None:
    sessions: list[list[str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        seen: list[str] = []
        sessions.append(seen)
        async for kind in _commands(ws, seen):
            if kind == "finalize":
                await ws.send_json(_final(f"session {len(sessions)}"))
                await ws.close()

    stream = SessionPerFlushStt(stt=LoopbackStt(await peer(provider)), conn_options=NO_RETRY)
    stream.push_frame(audio(50))
    stream.flush()
    stream.push_frame(audio(50))
    stream.end_input()
    got = await events(stream)
    finals = [e.alternatives[0].text for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals == ["session 1", "session 2"]
    # the input's end flushes the last session, which then owes nothing: no third one opens
    assert sessions == [["audio", "finalize"], ["audio", "finalize"]]


class FinalizedWhenQuietStt(LoopbackSttStream):
    audio_idle = 2.0

    def on_audio_idle(self) -> Any:
        return {"type": "finalize"}


# Catches: a provider's silence rule (finalize after a quiet spell) written as the plugin's own
# timer, or its frame sent again and again while the audio stays silent.
async def test_stt_a_silence_sends_the_input_idle_frame_once(
    peer: Callable[..., Awaitable[str]],
) -> None:
    seen: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for kind in _commands(ws, seen):
            if kind == "close":
                await ws.send_json({"type": "finished"})

    stream = FinalizedWhenQuietStt(stt=LoopbackStt(await peer(provider)), conn_options=NO_RETRY)
    async with stream:
        stream.push_frame(audio(50))
        await until(lambda: seen == ["audio"], "the audio")
        _core.skip_host_time(2.0)
        await until(lambda: seen == ["audio", "finalize"], "the idle frame")
        _core.skip_host_time(2.0)
        stream.end_input()
        await events(stream)
    # the idle frame once, then the end's flush and goodbye
    assert seen == ["audio", "finalize", "finalize", "close"]


# Catches: two streams sharing a provider's rate limit (one frame per interval per key) sending at
# once, or a paced frame never released.
async def test_streams_sharing_a_pacing_space_their_frames(
    peer: Callable[..., Awaitable[str]],
) -> None:
    # whether the clock had moved past the interval when each audio frame came
    released = asyncio.Event()
    arrived: list[bool] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for kind in _commands(ws, []):
            if kind == "audio":
                arrived.append(released.is_set())

    paced = type("PacedStt", (LoopbackSttStream,), {"pacing": stt.WebSocketPacing(min_interval=5)})
    model = LoopbackStt(await peer(provider))
    streams = [paced(stt=model, conn_options=NO_RETRY) for _ in range(2)]
    try:
        for stream in streams:
            stream.push_frame(audio(50))
        await until(lambda: len(arrived) == 1, "the first frame")
        released.set()
        _core.skip_host_time(5)
        await until(lambda: len(arrived) == 2, "the second frame, once the interval passed")
        assert arrived == [False, True]
    finally:
        for stream in streams:
            await stream.aclose()


class DirectTtsStream(LoopbackTtsStream):
    """The loopback TTS on a connection of its own."""

    def __init__(self, *, tts: LoopbackTts, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = None
        self._url = tts.mux.url

    async def connect(self, conn_options: APIConnectOptions) -> Any:
        return _tts.WebSocketTarget(self._url, request_id_header="x-request-id")


_tts = tts


def _model(url: str, stream: type[LoopbackTtsStream], *, as_pushed: bool = False) -> LoopbackTts:
    """The loopback TTS, its streams of ``stream``'s class; ``as_pushed``: each pushed text goes
    as it is, not batched into sentences."""

    class Model(LoopbackTts):
        def __init__(self) -> None:
            unit = "text" if as_pushed else "sentences"
            capabilities = tts.TTSCapabilities(streaming=True, text_unit=unit)
            tts.TTS.__init__(self, capabilities=capabilities, sample_rate=16000, num_channels=1)
            self.mux = LoopbackMux(url)

        def _stream(self, *, conn_options: APIConnectOptions) -> LoopbackTtsStream:
            return stream(tts=self, conn_options=conn_options)

    return Model()


class SegmentPerConnectionTts(DirectTtsStream):
    connection_per_segment = True


def _tone(context: str, value: int) -> bytes:
    return context.encode() + b"|" + bytes([value, 0]) * 160


async def _say(stream: tts.SynthesizeStream, *sentences: str) -> list[tts.SynthesizedAudio]:
    for i, sentence in enumerate(sentences):
        if i:
            stream.flush()
        stream.push_text(sentence)
    stream.end_input()
    return [audio async for audio in stream]


# Catches: a provider that takes one segment per connection (telnyx closes after its audio,
# minimax says its task finished) failing the stream as a lost connection, its next segment sent
# on the finished session, or a connection opened only to end the stream.
async def test_tts_opens_a_connection_per_segment_completed_by_its_done_or_the_close(
    peer: Callable[..., Awaitable[str]],
) -> None:
    connections: list[list[str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        texts: list[str] = []
        connections.append(texts)
        async for msg in ws:
            command = json.loads(msg.data)
            texts.append(command["text"])
            if command["final"]:
                await ws.send_bytes(_tone(command["context_id"], len(connections)))
                if len(connections) == 1:
                    await ws.close()  # ends the segment by closing
                else:
                    await ws.send_json({"type": "done", "context_id": command["context_id"]})

    model = _model(await peer(provider), SegmentPerConnectionTts)
    heard = await _say(model.stream(conn_options=NO_RETRY), "one fish.", "two fish.")
    tones = sorted({bytes(a.frame.data)[0] for a in heard if any(bytes(a.frame.data))})
    assert tones == [1, 2]
    assert len(connections) == 2
    assert [t for t in connections[0] if t] == ["one fish."]
    await model.mux.aclose()


class AnsweredPerTextTts(LoopbackTtsStream):
    """A provider that answers each text on its own, with no end-of-segment message."""

    def __init__(self, *, tts: LoopbackTts, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._owed: dict[str, int] = {}
        self._ended: set[str] = set()

    def encode(self, item: Any) -> Any:
        if isinstance(item, tts.SegmentText):
            self._owed[item.segment_id] = self._owed.get(item.segment_id, 0) + 1
        if isinstance(item, tts.SegmentEnd):
            self._ended.add(item.segment_id)
            if not self._owed.get(item.segment_id):
                self.completed.append(item.segment_id)
                self.report(tts.SegmentDone(item.segment_id))
            return None
        return super().encode(item)

    answered: ClassVar[list[str]] = []
    completed: ClassVar[list[str]] = []

    def on_message(self, message: Any) -> None:
        context, audio = message.split(b"|", 1)
        segment = context.decode()
        self.report(tts.SegmentAudio(segment, audio))
        self.answered.append(segment)
        self._owed[segment] -= 1
        if not self._owed[segment] and segment in self._ended:
            self.report(tts.SegmentDone(segment))


# Catches: a segment whose texts were all answered before its end never completed (the provider
# sends no end-of-segment message), so the stream waits until the read limit fails it.
async def test_tts_encode_completes_a_segment_whose_answers_all_came(
    peer: Callable[..., Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            command = json.loads(msg.data)
            await ws.send_bytes(_tone(command["context_id"], 1))

    model = _model(await peer(provider), AnsweredPerTextTts, as_pushed=True)
    stream = model.stream(conn_options=NO_RETRY)
    heard: list[tts.SynthesizedAudio] = []

    async def read() -> None:
        async for audio in stream:
            heard.append(audio)

    async with stream:
        reader = asyncio.create_task(read())
        AnsweredPerTextTts.answered.clear()
        AnsweredPerTextTts.completed.clear()
        stream.push_text("One fish.")
        await until(lambda: AnsweredPerTextTts.answered, "the first text's answer")
        # the segment's end comes after its only answer: encode completes it
        stream.end_input()
        await until(reader.done, "the stream ends")
        reader.result()
    assert AnsweredPerTextTts.completed == AnsweredPerTextTts.answered[:1]
    assert heard
    await model.mux.aclose()


class PoolMux(LoopbackMux):
    routed = False


# Catches: a connect per utterance on a provider whose sockets can be reused, or a socket reused
# after a stream left it mid-answer, its late audio then reaching the next stream.
async def test_a_pooled_socket_is_reused_only_after_a_stream_that_ended_cleanly(
    peer: Callable[..., Awaitable[str]],
) -> None:
    connections = 0
    finals: list[str] = []
    texts: dict[str, str] = {}

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        nonlocal connections
        connections += 1
        async for msg in ws:
            command = json.loads(msg.data)
            texts[command["context_id"]] = texts.get(command["context_id"], "") + command["text"]
            if command["final"]:
                finals.append(texts[command["context_id"]].strip())
            if command["final"] and finals[-1] != "never":
                await ws.send_bytes(_tone(command["context_id"], 1))
                await ws.send_json({"type": "done", "context_id": command["context_id"]})

    url = await peer(provider)
    model = LoopbackTts(url)
    model.mux = PoolMux(url)
    for _ in range(2):
        assert await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    assert connections == 1, "the second stream took the first one's socket"

    cut = model.stream(conn_options=NO_RETRY)
    cut.push_text("never")
    cut.flush()
    await until(lambda: "never" in finals, "the provider holds its answer")
    await cut.aclose()
    assert await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    assert connections == 2, "a socket left mid-answer is not reused"
    await model.mux.aclose()


# Catches: a context the provider expired while the LLM was slow between sentences failing the
# stream (its end read as a protocol violation), or the context kept alive by frames sent through
# the gap: the next sentence opens a new context on the same socket.
async def test_a_context_the_provider_expired_in_a_gap_reopens_on_the_next_sentence(
    peer: Callable[..., Awaitable[str]], caplog: pytest.LogCaptureFixture
) -> None:
    sockets: list[web.WebSocketResponse] = []
    received: list[tuple[str, str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        sockets.append(ws)
        async for msg in ws:
            command = json.loads(msg.data)
            await ws.send_bytes(_tone(command["context_id"], 1))
            received.append((command["context_id"], command["text"].strip()))
            if command["final"]:
                await ws.send_json({"type": "done", "context_id": command["context_id"]})

    model = _model(await peer(provider), LoopbackTtsStream, as_pushed=True)
    stream = model.stream(conn_options=NO_RETRY)
    heard: list[tts.SynthesizedAudio] = []

    async def read() -> None:
        async for audio in stream:
            heard.append(audio)

    async with stream:
        reader = asyncio.create_task(read())
        stream.push_text("one fish. ")
        await until(lambda: received, "the first sentence answered")
        first = received[0][0]
        # the LLM is slow: the provider expires the open context
        _core.skip_host_time(30)
        await sockets[0].send_json({"type": "done", "context_id": first})
        await until(
            lambda: "context expired" in caplog.text or reader.done(), "the context's end read"
        )
        stream.push_text("two fish.")
        stream.end_input()
        await until(reader.done, "the stream ends")
        reader.result()
    second = received[1][0]
    assert second != first
    assert received == [(first, "one fish."), (second, "two fish."), (second, "")]
    assert heard
    assert len(sockets) == 1
    assert caplog.text.count("context expired") == 1, "one warning"
    await model.mux.aclose()


class NamedTts(DirectTtsStream):
    def on_message(self, message: Any) -> None:
        if isinstance(message, bytes) and message.startswith(b"rid|"):
            self.report(tts.ProviderRequestId(message[4:].decode()))
        elif message == b"fail":
            raise APIError("the provider failed", kind="server")
        else:
            super().on_message(message)


# Catches: the provider's request ids (the handshake's, and one a message names) lost from the
# audio and the errors, so a call cannot be traced on the provider's side.
async def test_the_providers_request_ids_reach_the_audio_and_the_errors(
    peer: Callable[..., Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        said = ""
        async for msg in ws:
            command = json.loads(msg.data)
            said += command["text"]
            if command["final"] and said.startswith("fail"):
                await ws.send_bytes(b"rid|req-3")
                await ws.send_bytes(b"fail")
            elif command["final"]:
                await ws.send_bytes(b"rid|req-2")
                await ws.send_bytes(_tone(command["context_id"], 1))
                await ws.send_json({"type": "done", "context_id": command["context_id"]})

    model = _model(await peer(provider, {"x-request-id": "req-1"}), NamedTts)
    heard = await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    ids = {request_id for audio in heard for request_id in audio.provider_request_ids}
    assert {"req-1", "req-2"} <= ids

    with pytest.raises(APIError) as failed:
        await _say(model.stream(conn_options=NO_RETRY), "fail now.")
    assert failed.value.request_id == "req-3"
    await model.mux.aclose()


class BilledTts(DirectTtsStream):
    def on_message(self, message: Any) -> None:
        if isinstance(message, bytes) and message.startswith(b"usage:"):
            _, segment, characters = message.decode().split(":")
            self.report(tts.SegmentUsage(segment, characters_count=int(characters)))
        else:
            super().on_message(message)


# Catches: what the provider billed for a segment dropped, or counted on another segment's audio
# and metrics, when a WebSocket stream reports it with the segment's answer.
async def test_a_segments_billed_usage_rides_on_its_final_audio(
    peer: Callable[..., Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            command = json.loads(msg.data)
            if command["final"]:
                context = command["context_id"]
                await ws.send_bytes(_tone(context, 1))
                await ws.send_bytes(f"usage:{context}:7".encode())
                await ws.send_json({"type": "done", "context_id": context})

    model = _model(await peer(provider), BilledTts)
    heard = await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    billed = [(a.is_final, a.characters_count) for a in heard if a.characters_count is not None]
    assert billed == [(True, 7)]
    await model.mux.aclose()


# The kinds' own terms over the shared layer.

RETRY_ONCE = APIConnectOptions(max_retry=1, retry_interval=0)


def _kind(error: Exception) -> str | None:
    return error.kind if isinstance(error, APIError) else None


# Catches: a WebSocket stream's retry reported by nobody (the socket layer's own retry hook
# shadowing the stream's), so the provider's recoverable error event never fires.
async def test_a_websocket_stt_and_tts_retry_emits_the_recoverable_error_event(
    peer: Callable[..., Awaitable[str]],
) -> None:
    attempts: list[int] = []

    async def flaky(ws: web.WebSocketResponse, request: web.Request) -> None:
        attempts.append(len(attempts))
        if len(attempts) % 2:
            await ws.close(code=1011)  # a server error: the attempt is retried
            return
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                command = json.loads(msg.data)
                if command.get("type") == "close":
                    await ws.send_json({"type": "finished"})
                elif command.get("final"):
                    await ws.send_bytes(_tone(command["context_id"], 1))
                    await ws.send_json({"type": "done", "context_id": command["context_id"]})

    url = await peer(flaky)
    stt_ = LoopbackStt(url)
    stt_errors: list[stt.STTError] = []
    stt_.on("error", stt_errors.append)
    stream = stt_.stream(conn_options=RETRY_ONCE)
    stream.push_frame(audio(50))
    stream.end_input()
    await events(stream)
    assert [(e.recoverable, _kind(e.error)) for e in stt_errors] == [(True, "server")]

    model = _model(url, DirectTtsStream)
    tts_errors: list[tts.TTSError] = []
    model.on("error", tts_errors.append)
    assert await _say(model.stream(conn_options=RETRY_ONCE), "one fish.")
    assert [(e.recoverable, _kind(e.error)) for e in tts_errors] == [(True, "server")]
    await model.mux.aclose()


class ReconnectingStt(LoopbackSttStream):
    max_session_duration = 60.0
    # a skip past the limit steps through each ping's answer deadline before a real socket could
    # carry the pong
    ping_interval = None


# Catches: a connection that reached its maximum session replaced only at a flush, so a stream
# that never flushes (provider VAD) reaches the provider's hard limit as an unexpected close.
async def test_a_never_flushed_stt_stream_is_replaced_at_its_session_limit(
    peer: Callable[..., Awaitable[str]],
) -> None:
    sessions: list[list[str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        seen: list[str] = []
        sessions.append(seen)
        async for _ in _commands(ws, seen):
            pass

    stream = ReconnectingStt(stt=LoopbackStt(await peer(provider)), conn_options=NO_RETRY)
    async with stream:
        stream.push_frame(audio(50))
        await until(lambda: sessions and sessions[0] == ["audio"], "the first audio")
        _core.skip_host_time(60)
        await until(lambda: len(sessions) == 2, "a new session past the limit")
        stream.push_frame(audio(50))
        await until(lambda: sessions[1] == ["audio"], "the audio after the limit")
        assert sessions[0] == ["audio"]


# Catches: a provider that finishes its session while audio still comes (its own end of
# session) read as the end of the whole stream, the audio after it dropped.
async def test_an_stt_session_finished_mid_stream_continues_on_a_new_connection(
    peer: Callable[..., Awaitable[str]],
) -> None:
    sessions: list[list[str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        seen: list[str] = []
        sessions.append(seen)
        async for kind in _commands(ws, seen):
            if kind == "audio" and len(sessions) == 1:
                await ws.send_json(_final("first"))
                await ws.send_json({"type": "finished"})
            elif kind == "close":
                await ws.send_json(_final("second"))
                await ws.send_json({"type": "finished"})
        seen.append("<closed>")

    stream = LoopbackStt(await peer(provider)).stream(conn_options=NO_RETRY)
    async with stream:
        stream.push_frame(audio(50))
        await until(lambda: sessions and "<closed>" in sessions[0], "the first session finished")
        stream.push_frame(audio(50))
        stream.end_input()
        got = await events(stream)
    finals = [e.alternatives[0].text for e in got if e.type is stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals == ["first", "second"]
    assert len(sessions) == 2 and sessions[1][0] == "audio"


class CompletedPerSegmentTts(DirectTtsStream):
    """A provider that takes a segment per session and answers each text on its own: ``encode``
    completes a segment whose answer came."""

    connection_per_segment = True
    answered: ClassVar[list[str]] = []

    def encode(self, item: Any) -> Any:
        if isinstance(item, tts.SegmentEnd):
            if item.segment_id in self.answered:
                self.report(tts.SegmentDone(item.segment_id))
            return None
        return super().encode(item)

    def on_message(self, message: Any) -> None:
        context, audio = message.split(b"|", 1)
        self.report(tts.SegmentAudio(context.decode(), audio))
        self.answered.append(context.decode())


# Catches: a segment completed outside ``on_message`` (from ``encode``) never finishing its
# session, so the next segment waits out the read limit or goes on the finished session.
async def test_a_segment_completed_by_encode_finishes_its_session(
    peer: Callable[..., Awaitable[str]],
) -> None:
    connections: list[list[str]] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        texts: list[str] = []
        connections.append(texts)
        async for msg in ws:
            command = json.loads(msg.data)
            texts.append(command["text"])
            await ws.send_bytes(_tone(command["context_id"], len(connections)))

    model = _model(await peer(provider), CompletedPerSegmentTts, as_pushed=True)
    stream = model.stream(conn_options=NO_RETRY)
    CompletedPerSegmentTts.answered.clear()
    heard: list[tts.SynthesizedAudio] = []

    async def read() -> None:
        async for audio in stream:
            heard.append(audio)

    async with stream:
        reader = asyncio.create_task(read())
        stream.push_text("One fish.")
        await until(lambda: len(CompletedPerSegmentTts.answered) == 1, "the first answer")
        stream.flush()
        stream.push_text("Two fish.")
        await until(lambda: len(CompletedPerSegmentTts.answered) == 2, "the second answer")
        stream.end_input()
        await until(reader.done, "the stream ends")
        reader.result()
    assert connections == [["One fish."], ["Two fish."]]
    assert sorted({bytes(a.frame.data)[0] for a in heard if any(bytes(a.frame.data))}) == [1, 2]
    await model.mux.aclose()


class CountingMux(LoopbackMux):
    decoded: ClassVar[int] = 0

    def decode(self, frame: bytes | str) -> Any:
        CountingMux.decoded += 1
        return super().decode(frame)


class RoutedTtsStream(LoopbackTtsStream):
    def decode(self, frame: bytes | str) -> Any:
        raise AssertionError("a routed frame is decoded once, by the multiplexer")


# Catches: a multiplexer handing a stream its decoded messages by frame bytes, so two identical
# frames share one entry and the stream decodes the second again (or gets the wrong message).
async def test_a_routed_message_reaches_its_stream_decoded_once(
    peer: Callable[..., Awaitable[str]],
) -> None:
    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            command = json.loads(msg.data)
            if command.get("final"):
                tone = _tone(command["context_id"], 1)
                for _ in range(3):
                    await ws.send_bytes(tone)
                await ws.send_json({"type": "done", "context_id": command["context_id"]})

    url = await peer(provider)
    model = _model(url, RoutedTtsStream)
    model.mux = CountingMux(url)
    CountingMux.decoded = 0
    heard = await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    assert heard
    assert CountingMux.decoded == 4
    await model.mux.aclose()


class FlushThenCloseTts(DirectTtsStream):
    """A provider whose segment end is two messages: a flush, then the context's close."""

    def encode(self, item: Any) -> Any:
        if isinstance(item, tts.SegmentEnd):
            self.send_now({"context_id": item.segment_id, "text": "flush", "final": False})
        return super().encode(item)


# Catches: a control message sent from ``encode`` written after the frame ``encode`` returned,
# so the provider reads a context's flush after its close.
async def test_a_control_message_sent_while_encoding_goes_ahead_of_the_item(
    peer: Callable[..., Awaitable[str]],
) -> None:
    texts: list[str] = []

    async def provider(ws: web.WebSocketResponse, request: web.Request) -> None:
        async for msg in ws:
            command = json.loads(msg.data)
            texts.append(command["text"] or "<end>")
            if command["final"]:
                await ws.send_bytes(_tone(command["context_id"], 1))
                await ws.send_json({"type": "done", "context_id": command["context_id"]})

    model = _model(await peer(provider), FlushThenCloseTts)
    assert await _say(model.stream(conn_options=NO_RETRY), "one fish.")
    assert texts[-2:] == ["flush", "<end>"]
    await model.mux.aclose()


@pytest.mark.parametrize("failure", ["close", "timeout"])
async def test_a_promised_completion_is_required_even_after_a_final_transcript(
    peer: Callable[[Handler], Awaitable[str]],
    failure: str,
) -> None:
    class Promised(LoopbackSttStream):
        completion: ClassVar[stt.Completion] = "session_finished"

    seen: list[str] = []

    async def provider(ws: web.WebSocketResponse, _: web.Request) -> None:
        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                command = json.loads(msg.data)["type"]
                seen.append(command)
                if command == "close":
                    await ws.send_json({"type": "transcript", "text": "one segment", "final": True})
                    if failure == "close":
                        await ws.close(code=1000)

    url = await peer(provider)
    stream = Promised(
        stt=LoopbackStt(url), conn_options=APIConnectOptions(max_retry=0, read_timeout=5)
    )
    got: list[stt.SpeechEvent] = []

    async def read() -> None:
        async for event in stream:
            got.append(event)

    async with stream:
        reader = asyncio.create_task(read())
        try:
            stream.push_frame(audio(100))
            stream.end_input()
            await until(
                lambda: any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in got),
                "the final segment arrives before completion",
            )
            if failure == "timeout":
                _core.skip_host_time(5)
            await until(reader.done, "the missing completion fails")
            with pytest.raises(APIError):
                reader.result()
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
