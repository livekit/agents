"""xAI plugin: streaming STT reconnects, batch transcript decoding and TTS socket reuse."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import (
    Agent,
    APIConnectOptions,
    APIError,
    UserInputTranscribedEvent,
    VoiceSession,
    http,
    stt,
    tts,
)
from livekit.plugins import xai
from livekit.plugins.xai import stt as xai_stt, tts as xai_tts
from tests.fakes import FakeAudioOutput, audio_frame, until

OPTIONS = APIConnectOptions(max_retry=3, retry_interval=0.01, connect_timeout=1, read_timeout=1)


async def test_a_dropped_stt_socket_reconnects_and_keeps_transcribing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that dies under the stream is replaced, not spun on.

    https://github.com/livekit/agents/issues/7363
    """
    connections = 0

    async def listen(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        if connections == 1:
            await socket.close()
            return socket
        await socket.send_json({"type": "transcript.created"})
        await socket.send_json(
            {"type": "transcript.done", "text": "hello", "language": "en", "words": []}
        )
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/v1/stt", listen)
    async with TestServer(app) as server, http.open():
        monkeypatch.setattr(xai_stt, "XAI_WEBSOCKET_URL", str(server.make_url("/v1/stt")))
        model = xai.STT(api_key="test-key")
        stream = model.stream(conn_options=OPTIONS)
        try:
            async with asyncio.timeout(5):
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        break
        finally:
            await stream.aclose()
            await model.aclose()

    assert event.alternatives[0].text == "hello"
    assert connections == 2


async def _recognize(monkeypatch: pytest.MonkeyPatch, body: dict[str, object]) -> stt.SpeechEvent:
    async def transcribe(request: web.Request) -> web.Response:
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/v1/stt", transcribe)
    async with TestServer(app) as server, http.open():
        monkeypatch.setattr(xai_stt, "XAI_REST_URL", str(server.make_url("/v1/stt")))
        model = xai.STT(api_key="test-key", enable_diarization=True)
        try:
            return await model.recognize(audio_frame(0.1), conn_options=OPTIONS)
        finally:
            await model.aclose()


async def test_batch_transcript_keeps_word_timings_and_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch transcript carries its words' timings and the diarized speaker."""
    words = [{"text": "hello", "start": 0.1, "end": 0.7, "speaker": 0}]

    event = await _recognize(monkeypatch, {"text": "hello", "language": "en", "words": words})

    [speech] = event.alternatives
    assert (speech.text, speech.speaker_id) == ("hello", "S0")
    assert speech.words is not None
    assert [(w.start_time, w.end_time) for w in speech.words] == [(0.1, 0.7)]


async def test_a_malformed_batch_transcript_breaks_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcript the service mangled fails as an API error, not a validation crash."""
    with pytest.raises(APIError, match="invalid xAI transcript") as failure:
        await _recognize(monkeypatch, {"text": "hello", "words": [{"text": ["bad"]}]})

    assert failure.value.kind == "protocol", "a broken contract is never retried"


class _TTSPeer:
    """Answers each segment's ``text.done`` with 10 ms of audio and ``audio.done``."""

    def __init__(self) -> None:
        self.connections = 0

    async def speak(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        delta = base64.b64encode(b"\x00\x00" * 240).decode()
        async for message in socket:
            if json.loads(message.data)["type"] == "text.done":
                await socket.send_json({"type": "audio.delta", "delta": delta})
                await socket.send_json({"type": "audio.done"})
        return socket


async def test_streaming_segments_reuse_one_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each flushed segment is spoken over the same connection.

    https://github.com/livekit/agents/issues/6078
    """
    peer = _TTSPeer()
    app = web.Application()
    app.router.add_get("/v1/tts", peer.speak)
    async with TestServer(app) as server, http.open():
        monkeypatch.setattr(xai_tts, "XAI_WEBSOCKET_URL", str(server.make_url("/v1/tts")))
        model = xai.TTS(api_key="test-key")
        try:
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_text("first")
                stream.flush()
                stream.push_text("second")
                stream.end_input()
                heard = b"".join([event.frame.data.tobytes() async for event in stream])
        finally:
            await model.aclose()

    assert heard == b"\x00\x00" * 480
    assert peer.connections == 1


class _EagerDoneSession:
    """An http session whose socket delivers audio.done while the text.done send is in flight.

    A local server cannot order the reply before the send returns, so this race needs a socket.
    """

    def __init__(self) -> None:
        self.replies: asyncio.Queue[http.WebSocketMessage] = asyncio.Queue()
        self.answered = False
        self.starved = asyncio.Event()  # every reply was read and the client waits for another

    async def ws_connect(self, url: str, **_kwargs: object) -> _EagerDoneSession:
        return self

    async def send_str(self, data: str) -> None:
        if json.loads(data)["type"] == "text.done":
            delta = base64.b64encode(b"\x00\x00" * 240).decode()
            for reply in ({"type": "audio.delta", "delta": delta}, {"type": "audio.done"}):
                self.replies.put_nowait(
                    http.WebSocketMessage(http.WebSocketMessageType.TEXT, json.dumps(reply))
                )
            self.answered = True
            for _ in range(10):
                await asyncio.sleep(0)

    async def receive(self) -> http.WebSocketMessage:
        if self.answered and self.replies.empty():
            self.starved.set()
        return await self.replies.get()

    async def close(self) -> None:
        pass


async def _heard(stream: tts.SynthesizeStream) -> bytes:
    return b"".join([event.frame.data.tobytes() async for event in stream])


@pytest.mark.xfail(
    strict=True,
    raises=TimeoutError,
    reason="https://github.com/livekit/agents/issues/6079 reproduces: early audio.done is dropped",
)
async def test_audio_done_before_text_done_returns_still_completes() -> None:
    """A segment whose audio.done lands while text.done is still being sent ends with its audio.

    https://github.com/livekit/agents/issues/6079
    """
    # a stand-in session: only it can answer before the send returns
    session = _EagerDoneSession()
    model = xai.TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    try:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("hello")
            stream.end_input()
            hearing = asyncio.create_task(_heard(stream))
            await asyncio.wait(
                [hearing, asyncio.create_task(session.starved.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for _ in range(20):  # a stream still able to end does so within a few turns
                if hearing.done():
                    break
                await asyncio.sleep(0)
            if not hearing.done():
                hearing.cancel()
                raise TimeoutError("the stream waits for a reply that never comes")
            heard = hearing.result()
    finally:
        await model.aclose()

    assert heard == b"\x00\x00" * 240


class _GrokVoicePeer:
    """xAI's realtime socket: keeps the connection and acks every item the client creates."""

    def __init__(self) -> None:
        self.socket: web.WebSocketResponse | None = None

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        socket = self.socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            event = json.loads(message.data)
            if event["type"] == "conversation.item.create":
                added = {"type": "conversation.item.added", "item": event["item"]}
                await socket.send_json({**added, "previous_item_id": None})
        return socket

    async def send(self, **event: object) -> None:
        assert self.socket is not None
        await self.socket.send_json(event)


async def test_repeated_item_transcription_emits_one_final() -> None:
    """xAI re-sending one item's transcript at each commit yields one final user turn.

    https://github.com/livekit/agents/issues/6710
    """
    peer = _GrokVoicePeer()
    app = web.Application()
    app.router.add_get("/v1/realtime", peer.handle)
    transcripts: list[UserInputTranscribedEvent] = []
    async with TestServer(app) as server, http.open():
        model = xai.realtime.RealtimeModel(
            api_key="test-key", base_url=str(server.make_url("/v1/realtime").with_scheme("ws"))
        )
        session = VoiceSession(llm=model, vad=None)
        session.output.audio = FakeAudioOutput()
        session.on("user_input_transcribed", transcripts.append)
        await session.start(Agent(instructions="Answer."), room=None)
        await until(lambda: peer.socket, "the session connects")

        user = {"id": "item_1", "type": "message", "role": "user", "content": []}
        await peer.send(type="conversation.item.added", previous_item_id=None, item=user)
        for transcript in ("what is", "what is the time"):
            await peer.send(
                type="conversation.item.input_audio_transcription.completed",
                item_id="item_1",
                transcript=transcript,
            )
        await peer.send(type="response.created", response={"id": "resp_1"})
        await peer.send(type="response.output_audio.delta", item_id="msg_1", delta="AAA=")
        await until(lambda: any(t.is_final for t in transcripts), "the turn is final")
        await session.aclose()

    assert [t.transcript for t in transcripts if t.is_final] == ["what is the time"]
    user_turns = [m for m in session.history.messages() if m.role == "user"]
    assert [m.text_content for m in user_turns] == ["what is the time"]
