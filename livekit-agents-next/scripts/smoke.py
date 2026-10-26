"""Smoke run of the copy: a session over ``_core.FakeRoom`` on virtual time, with a scripted LLM,
a scripted STT and a silent TTS written as plugins (so all three cross the core's provider views).
It starts, says something, replies, runs a tool, handles speech from the room,
explicitly switches to a second agent whose ``on_enter`` speaks, and closes.
Then a second session on a scripted realtime model replies through the core's realtime leg, a
third on the OpenAI realtime model (the core's own client) against a local socket, and
one session per overridden node shows the core calling it: an ``llm_node`` that rewrites the
text, a ``tts_node`` that doubles the audio and an ``stt_node`` that redacts the transcript.

    python livekit-agents-next/scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from aiohttp import web

from livekit import rtc
from livekit.agents import (
    AMD,
    Agent,
    AMDCategory,
    APIConnectOptions,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    FlushSentinel,
    LiveSession,
    ModelSettings,
    RunContext,
    _core,
    function_tool,
    llm,
    room_io,
    stt,
    tts,
)
from livekit.agents.llm._realtime.openai import RealtimeModel as OpenAIRealtimeModel
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid
from livekit.agents.voice.avatar import DataStreamAudioOutput

SAMPLE_RATE = 24000

# (words in the user's message, the tool it calls, its arguments from the message)
TOOL_CALLS = [
    ("weather", "get_weather", lambda text: {"city": "Paris"}),
    ("billing", "to_billing", lambda text: {}),
    ("order", "look_up_order", lambda text: {}),
]


class ScriptedLLM(llm.LLM):
    """Calls the tool a message asks for when the agent has it, reads a tool's output back,
    else echoes."""

    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        return _Stream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Stream(llm.LLMStream):
    async def _run(self) -> None:
        last = self._chat_ctx.items[-1]
        names = {t.info.name for t in self._tools if isinstance(t, llm.FunctionTool)}
        text = (last.text_content or "") if last.type == "message" else ""
        delta = llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
        if last.type == "function_call_output":
            delta = llm.ChoiceDelta(role="assistant", content=f"The tool said: {last.output}")
        for words, name, arguments in TOOL_CALLS:
            if words in text and name in names:
                call = llm.FunctionToolCall(
                    name=name, arguments=json.dumps(arguments(text)), call_id=shortuuid("call_")
                )
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
                break
        self._event_ch.send_nowait(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


class ScriptedSTT(stt.STT):
    """A streaming STT that hears "my name is Theo" in half a second of anything."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return _Recognize(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class _Recognize(stt.RecognizeStream):
    async def _run(self) -> None:
        heard = 0.0
        async for frame in self._input_ch:
            if not isinstance(frame, rtc.AudioFrame) or heard < 0:
                continue
            heard += frame.duration
            if heard >= 0.5:
                heard = -1.0
                said = stt.SpeechData(language="en", text="my name is Theo", confidence=1.0)
                for kind, alternatives in (
                    (stt.SpeechEventType.START_OF_SPEECH, []),
                    (stt.SpeechEventType.FINAL_TRANSCRIPT, [said]),
                    (stt.SpeechEventType.END_OF_SPEECH, []),
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=kind, alternatives=alternatives)
                    )


class SilentTTS(tts.TTS):
    """20 ms of silence per character, as one chunked synthesis."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _Chunked(tts=self, input_text=text, conn_options=conn_options)


class _Chunked(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(self.input_text))))
        output_emitter.flush()


class Billing(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You handle billing.")

    async def on_enter(self) -> None:
        self.session.say("Billing here, how can I help?")


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are the front desk.")

    async def on_enter(self) -> None:
        self.session.say("Hello, front desk.")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        print(f"on_user_turn_completed {new_message.text_content!r}")

    @function_tool
    async def get_weather(self, ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @function_tool
    async def to_billing(self, ctx: RunContext) -> None:
        """Transfer the caller to billing."""
        ctx.session.update_agent(Billing())


def print_item(ev: object) -> None:
    item = ev.item  # type: ignore[attr-defined]
    if item.type == "message":
        print(f"conversation_item {item.role}: {item.text_content!r}")
    else:
        print(f"conversation_item {item.type}: {item.old_agent_id} -> {item.new_agent_id}")


async def settle(
    room: _core.FakeRoom, seconds: float, *, hold: Callable[[], bool] = lambda: False
) -> None:
    """Run the virtual clock forward in 20 ms steps, letting the loop work between them; no step
    is taken while ``hold()`` says real-time work the clock must wait for is out."""
    for _ in range(int(seconds / 0.02)):
        async with asyncio.timeout(30):
            while hold():
                await asyncio.sleep(0.001)
        room.advance(0.02)
        for _ in range(5):
            await asyncio.sleep(0)


class ScriptedRealtime(llm.RealtimeModel):
    """A realtime model that answers each reply request with one spoken message."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=True,
            )
        )

    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        return _RealtimeSession(self)


class _RealtimeSession(llm.RealtimeSession):
    def __init__(self, model: ScriptedRealtime) -> None:
        super().__init__(model)
        self._chat_ctx = llm.ChatContext.empty()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def update_options(self, **_: object) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(self, **_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
        async def _one(value: object) -> object:
            yield value

        modalities: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        silence = rtc.AudioFrame(b"\x00\x00" * 4800, SAMPLE_RATE, 1, 4800)
        message = llm.MessageGeneration(
            message_id=shortuuid("msg_"),
            text_stream=_one("Hello from the realtime model."),
            audio_stream=_one(silence),
            modalities=modalities,
        )
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        fut.set_result(
            llm.GenerationCreatedEvent(
                message_stream=_one(message),
                function_stream=_empty(),
                user_initiated=True,
                response_id=shortuuid("resp_"),
            )
        )
        return fut

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        pass


async def _empty() -> object:
    return
    yield


async def realtime() -> None:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(llm=ScriptedRealtime(), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="realtime"), room=room)
    handle = session.generate_reply(instructions="greet the user")
    await settle(room, 2)
    assert handle.done(), "the realtime reply did not finish"
    await session.aclose()
    print(f"played_bytes={room.played_bytes()}")


class _DyingSession(_RealtimeSession):
    """Fails for good as soon as it opens."""

    def __init__(self, model: ScriptedRealtime) -> None:
        super().__init__(model)
        asyncio.get_running_loop().call_soon(
            self.emit,
            "error",
            llm.RealtimeModelError(
                timestamp=0.0, label="dying", error=RuntimeError("gone"), recoverable=False
            ),
        )


class DyingRealtime(ScriptedRealtime):
    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        return _DyingSession(self)


async def realtime_fallback() -> None:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    primary, backup = DyingRealtime(), ScriptedRealtime()
    adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
    changes: list[tuple[bool, bool]] = []
    adapter.on(
        "realtime_availability_changed",
        lambda ev: changes.append((ev.realtime_model is primary, ev.available)),
    )
    session = LiveSession(llm=adapter, vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="fallback"), room=room)
    await settle(room, 0.5)
    handle = session.generate_reply(instructions="greet the user")
    await settle(room, 2)
    assert handle.done(), "the backup model did not answer"
    assert changes == [(True, False)], changes
    await session.aclose()
    print(f"availability {changes}, played_bytes={room.played_bytes()}")


def _square(level: float, ms: int = 100) -> rtc.AudioFrame:
    """A frame whose RMS is ``level``: a duplex model's output, its own silence included."""
    n = SAMPLE_RATE * ms // 1000
    amp = int(level * 32767)
    data = b"".join(
        (amp if i % 2 == 0 else -amp).to_bytes(2, "little", signed=True) for i in range(n)
    )
    return rtc.AudioFrame(data, SAMPLE_RATE, 1, n)


class ScriptedDuplex(llm.DuplexModel):
    """A duplex model that answers each ask with one utterance and its transcript."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=False, auto_tool_reply_generation=False
            )
        )
        self.sessions: list[_DuplexScript] = []

    def session(self) -> llm.DuplexSession:
        self.sessions.append(_DuplexScript(self))
        return self.sessions[-1]

    async def aclose(self) -> None:
        pass


class _DuplexScript(llm.DuplexSession):
    def __init__(self, model: ScriptedDuplex) -> None:
        super().__init__(model)
        self.audio: asyncio.Queue[llm.DuplexAudioFrame] = asyncio.Queue()
        self.configured_with: list[object] = []

    @property
    def audio_stream(self) -> AsyncIterator[llm.DuplexAudioFrame]:
        async def _frames() -> AsyncIterator[llm.DuplexAudioFrame]:
            while True:
                yield await self.audio.get()

        return _frames()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    async def _update_instructions(self, instructions: str) -> None:
        self.configured_with.append(instructions)

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        pass

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def _update_options(self, **_: object) -> None:
        pass

    def _generate_reply(self, **_: object) -> None:
        for level, count in ((0.001, 20), (0.3, 3)):
            for _ in range(count):
                self.audio.put_nowait(llm.DuplexAudioFrame(_square(level)))
        self.emit(
            "transcript_delta",
            llm.DuplexOutputTranscriptDelta(
                text="Hello from the duplex model.", start_ms=2000, end_ms=2300
            ),
        )
        for _ in range(8):
            self.audio.put_nowait(llm.DuplexAudioFrame(_square(0.001)))


async def duplex() -> None:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    model = ScriptedDuplex()
    session = LiveSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are duplex.", name="duplex"), room=room)
    await settle(room, 0.2)
    handle = session.generate_reply()
    await settle(room, 4)
    assert handle.done(), "the duplex reply did not finish"
    script = model.sessions[0]
    assert script._configured.is_set(), "the configuration never reached the model as one unit"
    assert script.configured_with == ["You are duplex."], script.configured_with
    await session.aclose()
    print(f"played_bytes={room.played_bytes()}")


async def _openai_server(ws: web.WebSocketResponse, sessions: list[dict[str, Any]]) -> None:
    """The OpenAI Realtime protocol's happy path: items are acked, a reply is one spoken message."""
    tail: str | None = None
    async for msg in ws:
        ev = json.loads(msg.data)
        kind = ev["type"]
        if kind == "session.update":
            sessions.append(ev["session"])
        elif kind == "conversation.item.create":
            item = {**ev["item"], "id": ev["item"].get("id") or shortuuid("item_")}
            await ws.send_json(
                {"type": "conversation.item.added", "previous_item_id": tail, "item": item}
            )
            tail = item["id"]
        elif kind == "conversation.item.delete":
            await ws.send_json({"type": "conversation.item.deleted", "item_id": ev["item_id"]})
        elif kind == "response.create":
            resp, item = shortuuid("resp_"), shortuuid("item_")
            meta = ev["response"]["metadata"]
            pcm = base64.b64encode(b"\x00\x00" * 4800).decode()
            for out in (
                {"type": "response.created", "response": {"id": resp, "metadata": meta}},
                {"type": "response.output_item.added", "item": {"id": item, "type": "message"}},
                {
                    "type": "conversation.item.added",
                    "previous_item_id": tail,
                    "item": {"id": item, "type": "message", "role": "assistant", "content": []},
                },
                {"type": "response.content_part.added", "item_id": item, "part": {"type": "audio"}},
                {
                    "type": "response.output_audio_transcript.delta",
                    "item_id": item,
                    "delta": "Hello from the core's realtime client.",
                },
                {"type": "response.output_audio.delta", "item_id": item, "delta": pcm},
                {"type": "response.output_item.done", "item": {"id": item, "type": "message"}},
                {"type": "response.done", "response": {"id": resp, "status": "completed"}},
            ):
                await ws.send_json(out)
            tail = item


async def native_realtime() -> None:
    sessions: list[dict[str, Any]] = []

    async def socket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await _openai_server(ws, sessions)
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime", socket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    model = OpenAIRealtimeModel(
        api_key="sk-local", base_url=f"http://127.0.0.1:{port}/v1", voice="cedar"
    )
    session = LiveSession(llm=model, vad=None)
    session.on("conversation_item_added", print_item)
    metrics: list[Any] = []
    session.on("metrics_collected", lambda ev: metrics.append(ev.metrics))
    await session.start(Agent(instructions="You are realtime.", name="native"), room=room)
    handle = session.generate_reply(instructions="greet the user")
    for _ in range(300):
        if handle.done():
            break
        room.advance(0.02)
        await asyncio.sleep(0.01)
    assert handle.done(), "the native realtime reply did not finish"
    assert sessions[0]["audio"]["output"]["voice"] == "cedar", sessions[0]
    await session.aclose()
    await runner.cleanup()
    realtime_metrics = [m for m in metrics if m.type == "realtime_model_metrics"]
    # the connection's, then the reply's
    assert [m.request_id != "" for m in realtime_metrics] == [False, True], metrics
    print(f"played_bytes={room.played_bytes()} realtime_metrics={len(realtime_metrics)}")


class Shouting(Agent):
    """Rewrites the model's text, and ends the spoken segment after it."""

    def __init__(self) -> None:
        super().__init__(instructions="You shout.", name="shouting")

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[str | FlushSentinel]:
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                yield chunk.delta.content.upper()
        yield FlushSentinel()


class Repeated(Agent):
    """Plays every synthesized frame ``copies`` times."""

    def __init__(self, copies: int) -> None:
        super().__init__(instructions="You echo.", name=f"repeated_{copies}")
        self._copies = copies

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async for frame in Agent.default.tts_node(self, text, model_settings):
            for _ in range(self._copies):
                yield frame


class Redacting(Agent):
    """Hides the caller's name in what the STT heard."""

    def __init__(self) -> None:
        super().__init__(instructions="You redact.", name="redacting")

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterator[stt.SpeechEvent]:
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            for alt in ev.alternatives:
                alt.text = alt.text.replace("Theo", "[name]")
            yield ev


async def node_session(agent: Agent, act: Callable[[LiveSession, _core.FakeRoom], Any]) -> int:
    """Run ``act`` on a session started on ``agent``; the bytes the room played."""
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    session.on("conversation_item_added", print_item)
    session.on("user_input_transcribed", lambda ev: print(f"user_input {ev.transcript!r}"))
    await session.start(agent, room=room)
    await settle(room, 1)
    await act(session, room)
    await session.aclose()
    played: int = room.played_bytes()
    return played


async def nodes() -> None:
    print("== llm_node")
    said: list[str] = []

    def record(session: LiveSession) -> None:
        session.on("conversation_item_added", lambda ev: said.append(ev.item.text_content or ""))

    async def reply(session: LiveSession, room: _core.FakeRoom) -> None:
        record(session)
        session.generate_reply(user_input="hello there")
        await settle(room, 3)

    await node_session(Shouting(), reply)
    assert "YOU SAID: HELLO THERE" in said, said

    async def say(session: LiveSession, room: _core.FakeRoom) -> None:
        handle = session.say("Twenty characters...")
        await settle(room, 3)
        assert handle.done(), "say() did not finish"

    print("== tts_node")
    once = await node_session(Repeated(1), say)
    twice = await node_session(Repeated(2), say)
    print(f"played_bytes once={once} twice={twice}")
    assert twice == 2 * once, (once, twice)

    print("== stt_node")
    heard: list[str] = []

    async def speak(session: LiveSession, room: _core.FakeRoom) -> None:
        session.on("user_input_transcribed", lambda ev: heard.append(ev.transcript))
        room.speak(frames=40)
        await settle(room, 4)

    await node_session(Redacting(), speak)
    assert heard == ["my name is [name]"], heard


async def avatar() -> None:
    print("== avatar")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You speak through an avatar."), room=room)
    session.output.replace_audio_tail(DataStreamAudioOutput(room, destination_identity="avatar"))
    await settle(room, 0.5)
    handle = session.say("Twenty characters...")
    await settle(room, 1)
    assert not room.byte_streams(), "no audio before the avatar joins"
    room.join("avatar")
    await settle(room, 2)
    streams = room.byte_streams()
    topic, destinations, attributes, size, closed = streams[0]
    assert (topic, destinations, closed) == ("lk.audio_stream", ["avatar"], True), streams
    assert attributes == {"sample_rate": "24000", "num_channels": "1"}, attributes
    assert not handle.done(), "the speech waits for the avatar's playback"
    payload = json.dumps({"playback_position": 0.5, "interrupted": False})
    assert room.rpc("lk.playback_finished", "avatar", payload) == "ok"
    await settle(room, 1)
    assert handle.done(), "the avatar's lk.playback_finished ends the speech"
    await session.aclose()
    print(f"avatar streams={len(streams)} bytes={size} played_in_room={room.played_bytes()}")


class RecordingLLM(ScriptedLLM):
    """``ScriptedLLM`` keeping the ids of every context it is asked with."""

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[list[str]] = []

    def chat(self, *, chat_ctx: llm.ChatContext, **kwargs: Any) -> llm.LLMStream:
        self.contexts.append([item.id for item in chat_ctx.items])
        return super().chat(chat_ctx=chat_ctx, **kwargs)


class MarkupTTS(SilentTTS):
    """``SilentTTS`` speaking Cartesia's markup, keeping what the core told it about expressive."""

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            return "cartesia"

    def __init__(self) -> None:
        super().__init__()
        self.expressive: list[bool] = []

    def _set_expressive(self, enabled: bool) -> None:
        super()._set_expressive(enabled)
        self.expressive.append(enabled)


async def expressive() -> None:
    print("== expressive")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    model, voice = RecordingLLM(), MarkupTTS()
    session = LiveSession(stt=ScriptedSTT(), llm=model, tts=voice, vad=None, expressive=True)
    await session.start(Agent(instructions="You are expressive."), room=room)
    await settle(room, 1)
    session.generate_reply(user_input="hello there")
    await settle(room, 3)
    session.update_options(expressive=False)
    session.generate_reply(user_input="and again")
    await settle(room, 3)
    await session.aclose()
    guide = "lk.expressive.instructions"
    print(f"guide per request {[guide in ids for ids in model.contexts]}, tts {voice.expressive}")
    assert [guide in ids for ids in model.contexts] == [True, False], model.contexts
    assert voice.expressive == [True, False], voice.expressive


async def background_audio() -> None:
    print("== background audio")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You hum while you think."), room=room)

    async def tone() -> AsyncIterator[rtc.AudioFrame]:
        for _ in range(20):
            yield rtc.AudioFrame(b"\x10\x00" * 480, 48000, 1, 480)

    player = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.5),
        thinking_sound=[AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, probability=1.0)],
    )
    await player.start(room=room, agent_session=session)

    async def until(done: Callable[[], bool]) -> None:
        # files decode and host streams are pulled in real time, the room plays in virtual time
        for _ in range(300):
            if done():
                return
            await settle(room, 0.1)
            await asyncio.sleep(0.01)
        raise AssertionError("background audio stalled")

    before = room.played_bytes()
    once = player.play(tone())
    await until(once.done)
    session.generate_reply(user_input="hello there")
    await settle(room, 2)
    looped = player.play(BuiltinAudioClip.HOLD_MUSIC, loop=True)
    playing = room.played_bytes()
    await until(lambda: room.played_bytes() > playing)
    assert not looped.done()
    looped.stop()
    assert looped.done()

    # a close cancelled while a sound still stops leaves the player open, thinking sounds on
    release, pulled = asyncio.Event(), asyncio.Event()

    async def stubborn() -> AsyncIterator[rtc.AudioFrame]:
        try:
            yield rtc.AudioFrame(b"\x10\x00" * 480, 48000, 1, 480)
            pulled.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    player.play(stubborn())
    await until(pulled.is_set)
    close = asyncio.create_task(player.aclose())
    await asyncio.sleep(0.1)
    close.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await close
    assert not player._player.closed
    assert player._agent_state_changed in session._events["agent_state_changed"]
    assert not player.play(BuiltinAudioClip.HOLD_MUSIC).done()
    release.set()
    await player.aclose()
    await session.aclose()
    played = room.played_bytes() - before
    print(f"background audio played={played}")
    assert played > 0, played


class ClassifyingLLM(ScriptedLLM):
    """Answers AMD's classification with ``label``; anything else as ``ScriptedLLM``."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.verdict = label
        # classifications asked for and not answered yet
        self.classifying = 0

    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        names = {t.info.name for t in tools or [] if isinstance(t, llm.FunctionTool)}
        if "save_prediction" not in names:
            return super().chat(chat_ctx=chat_ctx, tools=tools)
        self.classifying += 1
        return _Classify(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Classify(llm.LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, ClassifyingLLM)
        try:
            call = llm.FunctionToolCall(
                name="save_prediction",
                arguments=json.dumps({"label": self._llm.verdict}),
                call_id=shortuuid("call_"),
            )
            delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
            self._event_ch.send_nowait(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))
        finally:
            self._llm.classifying -= 1


async def amd() -> None:
    """AMD on the session's transcripts: an IVR greeting, handed to the navigator."""
    print("== amd")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    classifier = ClassifyingLLM("machine-ivr")
    session = LiveSession(stt=ScriptedSTT(), llm=classifier, tts=SilentTTS())
    heard = asyncio.Event()
    session.on("user_input_transcribed", lambda ev: ev.is_final and heard.set())
    await session.start(Agent(instructions="You call people."), room=room)
    async with AMD(session, llm=None, stt=None, suppress_compatibility_warning=True) as detector:
        assert session.amd is detector and detector.pending
        room.speak(frames=40)
        # AMD's timers run on the room's clock, the greeting's recognition on real time: the
        # clock stays put until the greeting is transcribed, and while a classification is out,
        # so a loaded machine cannot reach the no-speech or detection timeout first
        await asyncio.wait_for(heard.wait(), 30)
        result = asyncio.ensure_future(detector.execute())
        await settle(room, 30, hold=lambda: classifier.classifying > 0)
        assert result.done(), "AMD gave no verdict"
        verdict = result.result()
        print(f"amd {verdict.category.value} reason={verdict.reason} {verdict.transcript!r}")
        assert verdict.category == AMDCategory.MACHINE_IVR, verdict
        assert "send_dtmf_events" in [getattr(t, "id", None) for t in session.tools], "no IVR"
    assert session.amd is None
    await session.aclose()


async def filler() -> None:
    """A tool's ``with_filler`` callable, asked by the core while the tool blocks."""
    print("== filler")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    release = asyncio.Event()
    steps: list[int] = []

    def source(step: int) -> str:
        steps.append(step)
        return "One moment please."

    @function_tool
    async def look_up_order(ctx: RunContext) -> str:
        """Look up the caller's order."""
        async with ctx.with_filler(source, delay=0.5, interval=1.0, max_steps=2):
            await release.wait()
        return "shipped"

    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You track orders.", tools=[look_up_order]), room=room)
    session.generate_reply(user_input="where is my order?")
    await settle(room, 4)
    release.set()
    await settle(room, 4)
    spoken = [text for topic, text in room.text_streams() if topic == "lk.transcription"]
    print(f"filler steps={steps} spoken={spoken}")
    assert steps == [0, 1], steps
    assert spoken.count("One moment please.") == 2, spoken
    assert "The tool said: shipped" in spoken, spoken
    await session.aclose()


class KeytermSTT(ScriptedSTT):
    """``ScriptedSTT`` taking keyterms, recording each list it is handed."""

    def __init__(self) -> None:
        stt.STT.__init__(
            self,
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False, keyterms=True),
        )
        self.pushed: list[list[str]] = []

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self.pushed.append(list(keyterms))


class KeytermLLM(ScriptedLLM):
    """Answers each detection pass: "Theo" pending, then confirmed."""

    def __init__(self) -> None:
        super().__init__()
        self.passes = 0

    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        names = {t.info.name for t in tools or [] if isinstance(t, llm.FunctionTool)}
        assert names == {"record_keyterms"}, names
        self.passes += 1
        return _Keyterms(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Keyterms(llm.LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, KeytermLLM)
        update = {"pending": ["Theo"], "confirm": [], "remove": []}
        if self._llm.passes > 1:
            update = {"pending": [], "confirm": ["Theo"], "remove": []}
        call = llm.FunctionToolCall(
            name="record_keyterms", arguments=json.dumps(update), call_id=shortuuid("call_")
        )
        delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
        self._event_ch.send_nowait(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


async def keyterms() -> None:
    """Keyterm detection: an LLM pass per user turn, a confirmed term pushed to the STT."""
    print("== keyterm detection")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    recogniser, detector = KeytermSTT(), KeytermLLM()
    session = LiveSession(
        stt=recogniser,
        llm=ScriptedLLM(),
        tts=SilentTTS(),
        vad=None,
        stt_context_options={
            "keyterms": ["LiveKit"],
            "keyterm_detection": {"enabled": True, "llm": detector},
        },
    )
    await session.start(Agent(instructions="You listen for names."), room=room)
    room.speak(frames=40)
    await settle(room, 6)
    session.generate_reply(user_input="yes, Theo, T H E O")
    await settle(room, 6)
    print(f"keyterm passes={detector.passes} keyterms={session.keyterms}")
    assert detector.passes >= 2, detector.passes
    assert session.keyterms == ["LiveKit", "Theo"], session.keyterms
    assert recogniser.pushed[-1] == ["LiveKit", "Theo"], recogniser.pushed
    await session.aclose()


async def room_camera() -> None:
    """``RoomOptions(video_input=True)``: the core reads the person's camera, as
    ``session.input.video``."""
    print("== room camera")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(
        Agent(instructions="You can see."),
        room=room,
        room_options=room_io.RoomOptions(video_input=True),
    )
    assert isinstance(session.input.video, _core.RoomVideoInput), session.input.video
    # the room's microphone too, as `RoomIO` sets `session.input.audio`
    assert isinstance(session.input.audio, _core.RoomAudioInput), session.input.audio
    print(f"video input {type(session.input.video).__name__}")
    await session.aclose()


async def job() -> None:
    """A console job, as a job process runs it: the entrypoint connects (twice: once is made) and
    waits for the participant in the room, for whom a participant entrypoint runs, an API call is
    skipped, the
    shutdown it asks for runs the shutdown callback and tells the parent."""
    from livekit.agents.ipc import proto
    from livekit.agents.ipc.job_proc_lazy_main import _JobProc
    from livekit.agents.job import (
        JobAcceptArguments,
        JobContext,
        JobExecutorType,
        JobProcess,
        RunningJobInfo,
    )
    from livekit.protocol import agent as agent_proto, models

    print("== job")
    seen: list[str] = []

    async def on_participant(ctx: JobContext, p: rtc.RemoteParticipant) -> None:
        seen.append(f"participant {p.identity}")

    async def on_shutdown(reason: str) -> None:
        seen.append(f"shutdown callback {reason!r}")

    async def entrypoint(ctx: JobContext) -> None:
        ctx.add_participant_entrypoint(on_participant)
        with contextlib.suppress(ValueError):
            ctx.add_participant_entrypoint(on_participant)
            seen.append("added twice")
        ctx.add_shutdown_callback(on_shutdown)
        await asyncio.gather(ctx.connect(), ctx.connect())
        seen.append(f"waited for {(await ctx.wait_for_participant()).identity}")
        seen.append(f"delete_room {type(await ctx.delete_room()).__name__}")
        ctx.shutdown("done")

    class Parent:
        """The IPC client the job process tells its parent through."""

        sent: list[str] = []

        async def send(self, msg: object) -> None:
            self.sent.append(type(msg).__name__)

        def stop(self) -> None:
            pass

    runner = _JobProc(
        lambda _: None,
        entrypoint,
        None,
        session_end_timeout=5.0,
        executor_type=JobExecutorType.THREAD,
    )
    runner._client = Parent()  # type: ignore[assignment]
    runner._job_proc = JobProcess(
        executor_type=JobExecutorType.THREAD, user_arguments=None, http_proxy=None
    )
    request = proto.StartJobRequest()
    request.running_job = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=agent_proto.Job(id="AJ_smoke", room=models.Room(name="console", sid="RM_smoke")),
        url="",
        token="",
        worker_id="smoke",
        fake_job=True,
    )
    runner._start_job(request)
    assert runner._job_task is not None
    await asyncio.wait_for(runner._job_task, 30)
    print(f"job {seen} sent={Parent.sent}")
    # the participant's entrypoint runs beside the entrypoint, in no set order with it
    assert sorted(seen) == [
        "delete_room DeleteRoomResponse",
        "participant mock_user",
        "shutdown callback 'done'",
        "waited for mock_user",
    ], seen
    assert Parent.sent == ["ShuttingDown", "Exiting"], Parent.sent


def agent_name(session: LiveSession) -> str:
    return type(session.current_agent).__name__


def log_transport() -> None:
    """A job process's log record whose ``extra`` holds an object pickle refuses (an ``Agent``)
    reaches the supervisor with that field as its ``repr``, and no logging error."""
    print("== log transport, a non-picklable extra")
    import pickle
    import socket

    from livekit.agents.ipc.log_queue import LogQueueHandler
    from livekit.agents.utils.aio import duplex_unix

    child, parent = socket.socketpair()
    parent.settimeout(5)  # a record never sent fails rather than hangs
    handler = LogQueueHandler(duplex_unix._Duplex.open(child))
    errors: list[logging.LogRecord] = []
    handler.handleError = errors.append  # type: ignore[method-assign]

    @function_tool
    async def local_tool() -> str:
        """A tool pickle cannot reach (a local function)."""
        return ""

    agent = Agent(instructions="x", name="pii", tools=[local_tool])
    record = logging.LogRecord("livekit.agents", logging.WARNING, __file__, 1, "rejected", (), None)
    record.__dict__["lk.pii.output"] = agent
    handler.emit(record)
    received = pickle.loads(duplex_unix._Duplex.open(parent).recv_bytes())
    handler.close()
    handler.thread.join()
    assert not errors, errors
    assert received.__dict__["lk.pii.output"] == repr(agent), received.__dict__
    print("record received, extra as its repr")


async def main() -> int:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)

    def on_tools(ev: object) -> None:
        calls = [(c.name, o.output) for c, o in ev.zipped()]  # type: ignore[attr-defined]
        print(f"function_tools_executed {calls}")

    def on_state(ev: object) -> None:
        # the session's state is the event's by the time a listener runs
        assert session.agent_state == ev.new_state  # type: ignore[attr-defined]
        print(f"agent_state {ev.new_state}")  # type: ignore[attr-defined]

    session.on("agent_state_changed", on_state)
    session.on("user_input_transcribed", lambda ev: print(f"user_input {ev.transcript!r}"))
    session.on("conversation_item_added", print_item)
    session.on("function_tools_executed", on_tools)
    session.on("speech_created", lambda ev: print(f"speech_created {ev.source}"))
    session.on("close", lambda ev: print(f"close {ev.reason.value}"))

    print("== start")
    await session.start(FrontDesk(), room=room)
    await settle(room, 2)

    print("== say")
    handle = session.say("Let me say something.")
    await settle(room, 2)
    assert handle.done(), "say() did not finish"

    print("== say, streamed words and pre-synthesised audio")

    async def words() -> AsyncIterator[str]:
        for word in ("Streamed ", "words."):
            yield word

    async def frames() -> AsyncIterator[rtc.AudioFrame]:
        for _ in range(10):
            yield rtc.AudioFrame(b"\0\0" * 480, 48000, 1, 480)

    handle = session.say(words())
    await settle(room, 2)
    assert handle.done(), "say(AsyncIterable) did not finish"
    before = room.played_bytes()
    handle = session.say("Pre-recorded.", audio=frames())
    await settle(room, 2)
    assert handle.done(), "say(audio=) did not finish"
    assert room.played_bytes() > before, "say(audio=) played nothing"

    print("== generate_reply")
    session.generate_reply(user_input="hello there")
    await settle(room, 3)

    print("== tool")
    session.generate_reply(user_input="what's the weather like?")
    await settle(room, 4)

    print("== speech from the room")
    room.speak(frames=40)
    await settle(room, 6)

    print("== session.run(), asserted")
    result = session.run(user_input="what's the weather like?")
    await settle(room, 4)
    assert result.done(), "the run did not settle"
    result.expect.next_event().is_function_call(name="get_weather")
    result.expect.next_event().is_function_call_output(output="sunny in Paris")
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()
    print(f"run events: {[type(e).__name__ for e in result.events]}")

    print("== handoff")
    session.generate_reply(user_input="put me through to billing")
    await settle(room, 4)
    assert agent_name(session) == "Billing", agent_name(session)

    print("== close")
    await session.aclose()

    print(f"played_bytes={room.played_bytes()} text_streams={len(room.text_streams())}")
    print("history:")
    for item in session.history.items:
        detail = getattr(item, "text_content", None) or getattr(item, "name", "")
        print(f"  {item.type} {getattr(item, 'role', '')} {detail or ''}".rstrip())

    print("== agent.run(), the text path")
    text = await Agent(instructions="You answer by text.", name="text", llm=ScriptedLLM()).run(
        "hello by text"
    )
    print(f"output {text.output!r}")
    assert text.output == "You said: hello by text", text.output

    print("== realtime")
    await realtime()

    print("== duplex")
    await duplex()

    print("== realtime fallback")
    await realtime_fallback()

    print("== native realtime")
    await native_realtime()

    await nodes()
    await avatar()
    await expressive()
    await background_audio()
    await amd()
    await filler()
    await keyterms()
    await room_camera()
    await job()
    log_transport()
    return 0


class _Tracebacks(logging.StreamHandler):  # type: ignore[type-arg]
    """Prints the warnings and errors, and counts those carrying an exception (a callback the
    emitter or the loop swallowed): the smoke fails on any."""

    count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            _Tracebacks.count += 1
        super().emit(record)


if __name__ == "__main__":
    logging.getLogger().addHandler(_Tracebacks())
    code = asyncio.run(main())
    if _Tracebacks.count:
        print(f"FAILED: {_Tracebacks.count} logged traceback(s)")
        code = 1
    sys.exit(code)
