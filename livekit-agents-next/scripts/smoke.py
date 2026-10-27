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
import subprocess
import sys
from collections.abc import AsyncIterable, AsyncIterator, Callable
from pathlib import Path
from typing import Any

from aiohttp import web

from livekit import api, rtc
from livekit.agents import (
    AMD,
    VAD,
    Agent,
    AMDCategory,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    FlushSentinel,
    ModelSettings,
    RunContext,
    VADCapabilities,
    VADEvent,
    VADEventType,
    VADStream,
    VoiceSession,
    _core,
    function_tool,
    llm,
    room_io,
    stt,
    tts,
)
from livekit.agents.llm._realtime import RealtimeModel as OpenAIRealtimeModel
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid
from livekit.agents.voice.avatar import DataStreamAudioOutput

SAMPLE_RATE = 24000

# (words in the user's message, the tool it calls, its arguments from the message)
TOOL_CALLS = [
    ("weather", "get_weather", lambda text: {"city": "Paris"}),
    ("billing", "to_billing", lambda text: {}),
    ("order", "look_up_order", lambda text: {}),
    ("keypad", "send_dtmf_events", lambda text: {"events": ["1", "#"]}),
    ("bye", "end_call", lambda text: {}),
    ("book", "book", lambda text: {"day": "Monday"}),
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
        self.send(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


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
        async for frame in self.input:
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
                    self.send(stt.SpeechEvent(type=kind, alternatives=alternatives))


class BatchSTT(stt.STT):
    """A recognize-only STT: it hears "batch words" in any utterance of a tenth of a second."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.utterances: list[float] = []

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        self.utterances.append(sum(f.duration for f in frames))  # type: ignore[attr-defined]
        said = stt.SpeechData(language="en", text="batch words", confidence=1.0)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[said])


class ScriptedVAD(VAD):
    """Speech starts with the first frame and ends after half a second, its frames attached."""

    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.032))

    def stream(self) -> VADStream:
        return _Detect(self)


class _Detect(VADStream):
    async def _main_task(self) -> None:
        speech: list[rtc.AudioFrame] = []
        ended = False
        async for frame in self.input:
            if not isinstance(frame, rtc.AudioFrame) or ended:
                continue
            kind = VADEventType.START_OF_SPEECH if not speech else None
            speech.append(frame)
            if sum(f.duration for f in speech) >= 0.5:
                kind = VADEventType.END_OF_SPEECH
            if kind is not None:
                self.send(
                    VADEvent(
                        type=kind,
                        samples_index=0,
                        timestamp=0.0,
                        speech_duration=0.0,
                        silence_duration=0.0,
                        frames=list(speech) if kind == VADEventType.END_OF_SPEECH else [],
                    )
                )
            ended = kind == VADEventType.END_OF_SPEECH


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
    session = VoiceSession(llm=ScriptedRealtime(), vad=None)
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
    session = VoiceSession(llm=adapter, vad=None)
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
    session = VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
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
    session = VoiceSession(llm=model, vad=None)
    session.on("conversation_item_added", print_item)
    usage: list[Any] = []
    session.on("session_usage_updated", lambda ev: usage.append(ev.usage.model_usage))
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
    # the connection's record, then the reply's, each folded into the realtime model's usage
    assert [[u.model for u in models] for models in usage] == [["gpt-realtime"]] * 2, usage
    print(f"played_bytes={room.played_bytes()} usage_updates={len(usage)}")


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


async def node_session(agent: Agent, act: Callable[[VoiceSession, _core.FakeRoom], Any]) -> int:
    """Run ``act`` on a session started on ``agent``; the bytes the room played."""
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
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

    def record(session: VoiceSession) -> None:
        session.on("conversation_item_added", lambda ev: said.append(ev.item.text_content or ""))

    async def reply(session: VoiceSession, room: _core.FakeRoom) -> None:
        record(session)
        session.generate_reply(user_input="hello there")
        await settle(room, 3)

    await node_session(Shouting(), reply)
    assert "YOU SAID: HELLO THERE" in said, said

    async def say(session: VoiceSession, room: _core.FakeRoom) -> None:
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

    async def speak(session: VoiceSession, room: _core.FakeRoom) -> None:
        session.on("user_input_transcribed", lambda ev: heard.append(ev.transcript))
        room.speak(frames=40)
        await settle(room, 4)

    await node_session(Redacting(), speak)
    assert heard == ["my name is [name]"], heard

    print("== recognize-only stt, over the vad")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    batch = BatchSTT()
    session = VoiceSession(stt=batch, llm=ScriptedLLM(), tts=SilentTTS(), vad=ScriptedVAD())
    heard.clear()
    session.on("user_input_transcribed", lambda ev: heard.append(ev.transcript))
    await session.start(Agent(instructions="x", name="batch"), room=room)
    await settle(room, 1)
    room.speak(frames=40)
    await settle(room, 4)
    await session.aclose()
    print(f"heard {heard} utterances {batch.utterances}")
    assert heard == ["batch words"], heard
    assert batch.utterances and batch.utterances[0] >= 0.5, batch.utterances


async def avatar() -> None:
    print("== avatar")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
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


async def avatar_runner() -> None:
    print("== avatar runner")
    from livekit.agents.voice.avatar import (
        AudioReceiver,
        AudioSegmentEnd,
        AvatarOptions,
        AvatarRunner,
        VideoGenerator,
    )

    class Receiver(AudioReceiver):
        def __init__(self) -> None:
            super().__init__()
            self.items: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
            self.reports: list[tuple[str, float, bool]] = []

        def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
            self.reports.append(("finished", round(playback_position, 3), interrupted))

        async def notify_playback_started(self) -> None:
            self.reports.append(("started", 0.0, False))

        def __aiter__(self) -> Receiver:
            return self

        async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
            return await self.items.get()

    class Generator(VideoGenerator):
        def __init__(self) -> None:
            self.out: asyncio.Queue[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd] = (
                asyncio.Queue()
            )
            self.cleared = 0

        async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
            if isinstance(frame, rtc.AudioFrame):
                await self.out.put(rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGBA, bytes(16)))
            await self.out.put(frame)

        def clear_buffer(self) -> None:
            self.cleared += 1
            while not self.out.empty():
                self.out.get_nowait()

        async def __aiter__(self):  # type: ignore[override]
            while True:
                yield await self.out.get()

    room = _core.FakeRoom(sample_rate=48000)
    receiver, generator = Receiver(), Generator()
    options = AvatarOptions(
        video_width=2, video_height=2, video_fps=25, audio_sample_rate=24000, audio_channels=1
    )
    runner = AvatarRunner(room, audio_recv=receiver, video_gen=generator, options=options)
    published: list[bool] = []
    pushed: list[str] = []

    async def publish() -> None:
        published.append(True)

    async def push(frame: rtc.VideoFrame | rtc.AudioFrame) -> None:
        pushed.append(type(frame).__name__)

    runner._media._core_publish = publish  # type: ignore[method-assign]
    runner._media._core_push = push  # type: ignore[method-assign]
    await runner.start()
    assert not published, "a lazy runner publishes on its first frame"
    frame = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    for _ in range(2):
        receiver.items.put_nowait(frame)
    receiver.items.put_nowait(AudioSegmentEnd())
    await settle(room, 0.2)
    assert published == [True], published
    assert pushed == ["VideoFrame", "AudioFrame"] * 2, pushed
    assert receiver.reports == [("started", 0.0, False), ("finished", 0.04, False)], (
        receiver.reports
    )

    receiver.items.put_nowait(frame)
    await settle(room, 0.1)
    receiver.emit("clear_buffer")
    await settle(room, 0.1)
    assert generator.cleared == 1
    assert receiver.reports[-1] == ("finished", 0.02, True), receiver.reports
    await runner.aclose()
    try:
        await runner.wait_for_complete()
        raise AssertionError("the loops were cut by aclose")
    except asyncio.CancelledError:
        pass
    print(f"avatar runner pushed={len(pushed)} reports={receiver.reports}")


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
    session = VoiceSession(stt=ScriptedSTT(), llm=model, tts=voice, vad=None, expressive=True)
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
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
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
            self.send(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))
        finally:
            self._llm.classifying -= 1


async def amd() -> None:
    """AMD on the session's transcripts: an IVR greeting, handed to the navigator."""
    print("== amd")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    classifier = ClassifyingLLM("machine-ivr")
    session = VoiceSession(stt=ScriptedSTT(), llm=classifier, tts=SilentTTS())
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

    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
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


async def tool_outcomes() -> None:
    """A toolset that lists its tools in ``setup()`` (as an MCP server's does) runs them, and a
    tool that requires approval asks its read-back, then runs on the repeat after the answer."""
    print("== tool outcomes")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")

    @function_tool
    async def look_up_order(ctx: RunContext) -> str:
        """Look up the caller's order."""
        return "shipped"

    @function_tool(needs_approval=True)
    async def book(ctx: RunContext, day: str) -> str:
        """Book an appointment."""
        return f"booked {day}"

    class Late(llm.Toolset):
        async def setup(self) -> Late:
            self._tools = [look_up_order]
            return self

    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    agent = Agent(instructions="You book.", tools=[Late(id="late"), book])
    await session.start(agent, room=room)
    session.generate_reply(user_input="where is my order?")
    await settle(room, 4)
    session.generate_reply(user_input="book me on monday")
    await settle(room, 4)
    session.generate_reply(user_input="yes, book it")
    await settle(room, 4)
    spoken = [text for topic, text in room.text_streams() if topic == "lk.transcription"]
    print(f"tool outcomes spoken={spoken}")
    assert "The tool said: shipped" in spoken, spoken
    readback = "Please confirm: Book an appointment (day='Monday'). Yes or no?"
    assert readback in spoken, spoken
    assert "The tool said: booked Monday" in spoken, spoken
    outputs = [i.output for i in session.history.items if i.type == "function_call_output"]
    assert any(o.startswith("not run yet: the person answered 'yes, book it'") for o in outputs), (
        outputs
    )
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
        self.send(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


async def keyterms() -> None:
    """Keyterm detection: an LLM pass per user turn, a confirmed term pushed to the STT."""
    print("== keyterm detection")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    recogniser, detector = KeytermSTT(), KeytermLLM()
    session = VoiceSession(
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
    session = VoiceSession(llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
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


async def console() -> None:
    """``lk agent console``: the CLI's socket is the session's audio and its remote session's
    transport. The CLI asks for the session's state and a typed turn, sees the session's events,
    acks the played audio, and the socket ends when the session closes."""
    import struct

    from livekit.agents.voice.remote_session import ConsoleCall
    from livekit.protocol.agent_pb import agent_session as pb

    print("== console")
    accepted: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.get_running_loop().create_future()
    )
    server = await asyncio.start_server(lambda r, w: accepted.set_result((r, w)), "127.0.0.1", 0)
    ConsoleCall.addr = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(FrontDesk())
    reader, writer = await asyncio.wait_for(accepted, 10)

    def send(msg: pb.AgentSessionMessage) -> None:
        data = msg.SerializeToString()
        writer.write(struct.pack(">I", len(data)) + data)

    async def recv() -> pb.AgentSessionMessage | None:
        try:
            (size,) = struct.unpack(">I", await reader.readexactly(4))
            return pb.AgentSessionMessage.FromString(await reader.readexactly(size))
        except asyncio.IncompleteReadError:
            return None

    events: list[str] = []
    responses: dict[str, pb.SessionResponse] = {}

    async def until(request_id: str) -> pb.SessionResponse:
        while request_id not in responses:
            msg = await asyncio.wait_for(recv(), 10)
            assert msg is not None, "the socket ended early"
            kind = msg.WhichOneof("message")
            if kind == "event":
                events.append(msg.event.WhichOneof("event"))
            elif kind == "response":
                responses[msg.response.request_id] = msg.response
            elif kind == "audio_playback_flush":
                send(
                    pb.AgentSessionMessage(
                        audio_playback_finished=pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFinished()
                    )
                )
        return responses[request_id]

    send(
        pb.AgentSessionMessage(
            request=pb.SessionRequest(
                request_id="req_state", get_session_state=pb.SessionRequest.GetSessionState()
            )
        )
    )
    state = await until("req_state")
    assert not state.error and state.get_session_state.agent_id, state
    send(
        pb.AgentSessionMessage(
            request=pb.SessionRequest(
                request_id="req_run", run_input=pb.SessionRequest.RunInput(text="hello there")
            )
        )
    )
    run = await until("req_run")
    assert not run.error, run.error
    items = [i.WhichOneof("item") for i in run.run_input.items]
    print(f"console run_input items {items}")
    assert "message" in items, items
    await session.aclose()
    while (msg := await asyncio.wait_for(recv(), 10)) is not None:
        if msg.WhichOneof("message") == "event":
            events.append(msg.event.WhichOneof("event"))
    print(f"console events {sorted(set(events))}")
    assert {"agent_state_changed", "conversation_item_added"} <= set(events), events
    writer.close()
    server.close()


async def job() -> None:
    """A console job, as a job process runs it: the entrypoint connects (twice: once is made) and
    waits for the participant in the room, for whom a participant entrypoint runs, an API call is
    skipped, the
    shutdown it asks for runs the shutdown callback and tells the parent."""
    from livekit.agents.ipc.job_proc_lazy_main import _JobProc
    from livekit.agents.job import (
        JobAcceptArguments,
        JobContext,
        JobExecutorType,
        RunningJobInfo,
    )
    from livekit.protocol import agent as agent_proto, models

    print("== job")
    seen: list[str] = []

    async def on_participant(ctx: JobContext, p: rtc.RemoteParticipant) -> None:
        seen.append(f"participant {p.identity}")

    async def on_shutdown(reason: str) -> None:
        seen.append(f"shutdown callback {reason!r}")

    async def on_shutdown_no_reason() -> None:
        seen.append("shutdown callback, no reason")

    async def entrypoint(ctx: JobContext) -> None:
        ctx.add_participant_entrypoint(on_participant)
        with contextlib.suppress(ValueError):
            ctx.add_participant_entrypoint(on_participant)
            seen.append("added twice")
        ctx.add_shutdown_callback(on_shutdown)
        ctx.add_shutdown_callback(on_shutdown_no_reason)
        await asyncio.gather(ctx.connect(), ctx.connect())
        seen.append(f"waited for {(await ctx.wait_for_participant()).identity}")
        seen.append(f"delete_room {type(await ctx.delete_room()).__name__}")
        ctx.shutdown("done")

    import socket
    import struct

    runner = _JobProc(
        entrypoint,
        None,
        session_end_timeout=5.0,
        executor_type=JobExecutorType.THREAD,
    )
    # the parent's side of the socket, written and read by hand
    parent, child = socket.socketpair()

    def frame(msg_id: int, fields: dict[str, Any]) -> bytes:
        data = _core.ipc_encode(msg_id, fields)
        return struct.pack(">I", len(data)) + data

    init = {"asyncio_debug": False, "http_proxy": ""}
    init.update(ping_interval=0.0, ping_timeout=0.0, high_ping_threshold=0.0)
    parent.sendall(frame(0, init))
    client = await asyncio.to_thread(_core.IpcClient, child.detach(), runner.initialize, job=True)
    runner._client = client
    reader, writer = await asyncio.open_connection(sock=parent)

    async def received() -> str:
        (size,) = struct.unpack(">I", await reader.readexactly(4))
        msg_id, _ = _core.ipc_decode(await reader.readexactly(size))
        return {1: "InitializeResponse", 6: "Exiting", 11: "ShuttingDown"}[msg_id]

    assert await received() == "InitializeResponse"
    serving = asyncio.ensure_future(client.serve(runner))
    request = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=agent_proto.Job(id="AJ_smoke", room=models.Room(name="console", sid="RM_smoke")),
        url="",
        token="",
        worker_id="smoke",
        fake_job=True,
    )
    writer.write(frame(4, request._to_wire()))
    sent = [await asyncio.wait_for(received(), 30) for _ in range(2)]
    await asyncio.wait_for(serving, 30)
    assert runner._job_task is not None
    await asyncio.wait_for(runner._job_task, 30)
    await client.aclose()
    writer.close()
    print(f"job {seen} sent={sent}")
    # the participant's entrypoint runs beside the entrypoint, in no set order with it
    assert sorted(seen) == [
        "delete_room DeleteRoomResponse",
        "participant mock_user",
        "shutdown callback 'done'",
        "shutdown callback, no reason",
        "waited for mock_user",
    ], seen
    assert sent == ["ShuttingDown", "Exiting"], sent
    events, attributes = _core.job_timeline(1.0, 2.0, 0.0, 4.0, 5.0, 3_000_000_000_000_000_000)
    assert [e[0] for e in events][-1] == "job_started_on_server" and len(events) == 5, events
    assert dict(attributes) == {
        "lk.job.accept_latency": 1.0,
        "lk.job.entrypoint_latency": 1.0,
        "lk.job.dispatch_latency": 4.0,
    }, attributes


async def workflows() -> None:
    """The workflow checks (the core's) and keypad reception over a room: three keys and the
    stop key, then a sequence the inter-digit timeout cuts short."""
    from livekit.agents.beta import workflows as wf
    from livekit.agents.llm import ToolError

    print("== workflows")
    assert wf.validate_card_number("4242424242424242")
    assert not wf.validate_card_number("4242424242424241")
    assert not wf.is_expired(12, 99)
    assert wf.normalize_email_address(" theo@livekit.io ") == "theo@livekit.io"
    assert wf.normalize_phone_number("+1 (510) 555-0123") == "+15105550123"
    assert str(wf.normalize_date_of_birth(90, 5, 2)) == "1990-05-02"
    for check, value in ((wf.normalize_email_address, "theo"), (wf.normalize_phone_number, "12")):
        with contextlib.suppress(ToolError):
            check(value)
            raise AssertionError(f"{value!r} was accepted")
    assert wf.format_dtmf([wf.DtmfEvent.ONE, wf.DtmfEvent.POUND]) == "1 #"
    assert wf.dtmf_event_to_code(wf.DtmfEvent.D) == 15

    room = _core.FakeRoom()
    pending = asyncio.ensure_future(wf.receive_dtmf(room, num_digits=3))
    await asyncio.sleep(0.05)
    for key in "123#":
        room.press_dtmf(key)
    result = await pending
    print(f"received {result!r}")
    assert result.user_input == "1 2 3", result
    pending = asyncio.ensure_future(wf.receive_dtmf(room, num_digits=4, timeout=0.1))
    await asyncio.sleep(0.05)
    room.press_dtmf("9")
    with contextlib.suppress(ToolError):
        await pending
        raise AssertionError("a short sequence completed")


class _FakeTransferHost:
    """The SDK side of a warm transfer, over fake rooms: records what the core asked for."""

    def __init__(self, human: _core.FakeRoom, http: list[tuple[int, str]] | None = None) -> None:
        self.human = human
        self.http = http or []
        self.calls: list[str] = []
        self.io = {"audio_input": True, "audio_output": False, "transcription_output": True}

    async def _core_start_hold(self) -> object:
        self.calls.append("hold")
        return "hold-handle"

    def _core_stop_hold(self, hold: object) -> None:
        self.calls.append(f"stop {hold}")

    def _core_io_state(self) -> dict[str, bool]:
        return dict(self.io)

    def _core_set_io(self, state: dict[str, bool]) -> None:
        self.io = state
        self.calls.append(f"io {sorted(k for k, v in state.items() if v)}")

    async def _core_connect_human(self, room_name: str, identity: str) -> _core.FakeRoom:
        self.calls.append(f"connect {room_name} as {identity}")
        return self.human

    async def _core_start_human(self) -> None:
        self.calls.append("start human")

    def _core_close_human(self, drain: bool) -> None:
        self.calls.append(f"close human drain={drain}")

    async def _core_disconnect_human(self) -> None:
        self.calls.append("disconnect human")

    async def _core_create_sip_participant(self, request: dict[str, Any]) -> None:
        self.calls.append(f"sip {request['sip_call_to']} trunk={request['sip_trunk_id']}")

    async def _core_connect_twilio_call(self, room_name: str, identity: str) -> str:
        return "wss://connector/stream?a=1&b=2"

    async def _core_http_post(
        self, url: str, headers: dict[str, str], form: list[tuple[str, str]]
    ) -> tuple[int, str]:
        fields = dict(form)
        self.calls.append(f"post {url.rsplit('/', 1)[-1]} from={fields.get('From')}")
        assert headers["Authorization"].startswith("Basic "), headers
        assert fields.get("Twiml", "").count("&amp;") in (0, 1), fields
        return self.http.pop(0)

    async def _core_move_participant(self, room: str, identity: str, destination: str) -> None:
        self.calls.append(f"move {identity} {room} -> {destination}")

    async def _core_delete_caller_room(self) -> None:
        self.calls.append("delete caller room")


async def tools() -> None:
    """The prebuilt tools in a session: keypad tones published by the core, then ``end_call``,
    after whose goodbye the core closes the session."""
    from livekit.agents.beta import EndCallTool, send_dtmf_events

    print("== tools")
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    closed: list[str] = []
    session.on("close", lambda ev: closed.append(ev.reason.value))
    agent = Agent(instructions="You answer the phone.", tools=[EndCallTool(), send_dtmf_events])
    await session.start(agent, room=room)
    session.generate_reply(user_input="press the keypad for me")
    await settle(room, 1)
    await asyncio.sleep(0.8)  # the tones are 0.3 s apart on the wall clock, as the reference's
    await settle(room, 2)
    print(f"dtmf published {room.dtmf_published()}")
    assert room.dtmf_published() == [(1, "1"), (11, "#")], room.dtmf_published()
    session.generate_reply(user_input="that's all, bye")
    await settle(room, 6)
    spoken = [text for topic, text in room.text_streams() if topic == "lk.transcription"]
    print(f"closed {closed} after {spoken[-1]!r}")
    assert closed == ["user_initiated"], closed
    assert spoken[-1] == "The tool said: say goodbye to the user", spoken


async def toolsets() -> None:
    """Tool search: a search loads the matching tools (ranked by the core); the proxy returns
    their schemas and calls one by name, its arguments checked by the core."""
    from livekit.agents.beta import KeywordSearchStrategy, ToolProxyToolset, ToolSearchToolset
    from livekit.agents.llm import ToolError

    print("== toolsets")

    @function_tool
    async def get_weather(city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @function_tool
    async def get_stock(symbol: str) -> str:
        """A stock's price."""
        return f"{symbol}: 100"

    search = ToolSearchToolset(id="search", tools=[get_weather, get_stock])
    await search.setup()
    print(await search._handle_search({"query": "weather"}))
    assert [t.id for t in search.tools] == ["tool_search", "get_weather"], search.tools
    with contextlib.suppress(ToolError):
        await search._handle_search({"query": "nothing matches"})
        raise AssertionError("an empty search loaded tools")
    keyword = ToolSearchToolset(
        id="kw", tools=[get_weather, get_stock], search_strategy=KeywordSearchStrategy()
    )
    await keyword.setup()
    await keyword._handle_search({"query": "stock"})
    assert [t.id for t in keyword.tools] == ["tool_search", "get_stock"], keyword.tools

    proxy = ToolProxyToolset(id="proxy", tools=[get_weather, get_stock])
    await proxy.setup()
    schemas = await proxy._handle_search({"query": "stock"})
    assert json.loads(schemas)["name"] == "get_stock", schemas
    called = await proxy._handle_call(None, {"name": "get_weather", "parameters": {"city": "Oslo"}})
    print(f"proxy called: {called}")
    assert called == "sunny in Oslo", called
    for bad in ({"name": "", "parameters": {}}, {"name": "nope", "parameters": {}}):
        with contextlib.suppress(ToolError):
            await proxy._handle_call(None, bad)
            raise AssertionError(f"{bad} was called")


async def cli() -> None:
    """What ``lk agent`` drives, short of a LiveKit server: the command line, the entrypoint's
    server found in a package, the log lines (JSON and colored), the dev channel to a stand-in
    CLI keeping a job across a reload, and the exit policy."""
    import os
    import struct
    import tempfile
    import traceback
    from pathlib import Path

    from livekit.agents import cli as agents_cli
    from livekit.agents._exceptions import CLIError
    from livekit.agents.job import JobAcceptArguments, RunningJobInfo
    from livekit.protocol import agent as agent_proto
    from livekit.protocol.agent_pb import agent_dev

    print("== cli")
    args = _core.parse_cli(["start", "agent.py", "--dev", "--reload-addr", "127.0.0.1:1"])
    assert (args["entrypoint"], args["cli_addr"], args["log_format"]) == (
        "agent.py",
        "127.0.0.1:1",
        "json",
    ), args
    for bad in (["start", "--cli-addr", "x"], ["console"], ["serve"]):
        with contextlib.suppress(CLIError):
            _core.parse_cli(bad)
            raise AssertionError(f"{bad} parsed")

    with tempfile.TemporaryDirectory() as root:
        package = Path(root, "myapp")
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "agent.py").write_text(
            "from livekit.agents import AgentServer\nserver = AgentServer()\n"
        )
        found = agents_cli._load_server(str(package / "agent.py"))
        assert type(found).__name__ == "AgentServer", found
        print(f"entrypoint {_core.entrypoint_module(str(package / 'agent.py'))[0]}")

    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("livekit.agents", logging.ERROR, __file__, 1, "x=%d", (3,), exc_info)
    record.job_id = "AJ_1"
    line = json.loads(agents_cli._LogFormatter("json", ensure_ascii=True).format(record))
    assert (line["message"], line["level"], line["job_id"]) == ("x=3", "ERROR", "AJ_1"), line
    assert "ValueError: boom" in line["exc_info"], line
    colored = agents_cli._LogFormatter("compact", ensure_ascii=False).format(record)
    assert "x=3" in colored and '{"job_id": "AJ_1"}' in colored and "boom" in colored, colored
    assert traceback  # the record carries a real traceback

    class Server:
        """The worker's side of the dev channel."""

        reloaded: list[RunningJobInfo] = []
        active_jobs = [
            RunningJobInfo(
                accept_arguments=JobAcceptArguments(name="n", identity="i", metadata="m"),
                job=agent_proto.Job(id="AJ_running"),
                url="wss://x",
                token="t",
                worker_id="w",
                fake_job=True,
            )
        ]

        class _jobs:  # noqa: N801
            @staticmethod
            async def launch(info: RunningJobInfo) -> None:
                Server.reloaded.append(info)

    async def read(reader: asyncio.StreamReader) -> agent_dev.AgentDevMessage:
        (length,) = struct.unpack("!I", await reader.readexactly(4))
        return agent_dev.AgentDevMessage.FromString(await reader.readexactly(length))

    def write(writer: asyncio.StreamWriter, msg: agent_dev.AgentDevMessage) -> None:
        data = msg.SerializeToString()
        writer.write(struct.pack("!I", len(data)) + data)

    seen: list[str] = []
    done = asyncio.Event()
    kept_token = (
        api.AccessToken("key", "secret_of_32_bytes_or_more_ok!!").with_identity("agent").to_jwt()
    )

    async def lk_cli(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append((await read(reader)).server_info.agent_name)
        assert (await read(reader)).HasField("get_running_jobs_request")
        kept = agent_dev.RunningAgentJobInfo(
            job=agent_proto.Job(id="AJ_kept").SerializeToString(), token=kept_token, url="ws://old"
        )
        write(
            writer,
            agent_dev.AgentDevMessage(
                get_running_jobs_response=agent_dev.GetRunningAgentJobsResponse(jobs=[kept])
            ),
        )
        write(
            writer,
            agent_dev.AgentDevMessage(
                get_running_jobs_request=agent_dev.GetRunningAgentJobsRequest()
            ),
        )
        await writer.drain()
        running = (await read(reader)).get_running_jobs_response.jobs
        seen.append(agent_proto.Job.FromString(running[0].job).id)
        writer.close()
        done.set()

    stand_in = await asyncio.start_server(lk_cli, "127.0.0.1", 0)
    port = stand_in.sockets[0].getsockname()[1]
    server = Server()
    await asyncio.wait_for(
        _core.dev_channel(
            f"127.0.0.1:{port}",
            "dev-agent",
            "wss://x",
            "key",
            "secret_of_32_bytes_or_more_ok!!",
            agents_cli._DevServer(server),
        ),  # type: ignore[arg-type]
        10,
    )
    await done.wait()
    stand_in.close()
    print(f"dev channel {seen}, reloaded {[j.job.id for j in server.reloaded]}")
    assert seen == ["dev-agent", "AJ_running"], seen
    assert [j.job.id for j in server.reloaded] == ["AJ_kept"], server.reloaded
    # on this server, with the token extended
    assert server.reloaded[0].url == "wss://x", server.reloaded
    assert server.reloaded[0].token not in ("", kept_token), server.reloaded

    exits = _core.ExitSignals(escalate=False)
    assert exits.signal(False) == "schedule"
    exits.served()
    assert exits.signal(True) == "force"
    assert os.getpid() > 0


async def warm_transfer() -> None:
    """The warm transfer the core runs over a host: dialled over SIP and connected (the caller
    leaving then deletes their room), declined, and dialled through Twilio, whose rejected caller
    ID is retried from the business number before the human agent's audio answers."""
    from livekit.agents.beta import workflows as wf
    from livekit.agents.llm import ChatContext, ToolError

    print("== warm transfer")
    history = ChatContext()
    history.add_message(role="user", content="my order is late")
    transfer = wf.WarmTransfer("+15550100", sip_trunk_id="ST_smoke", chat_ctx=history)
    instructions = transfer._agent.instructions
    assert "Caller: my order is late" in str(getattr(instructions, "common", instructions))
    assert [t.id for t in transfer._agent.tools] == [
        "connect_to_caller",
        "decline_transfer",
        "voicemail_detected",
    ], transfer._agent.tools

    caller = _core.FakeRoom(name="caller")
    caller.join("user")
    host = _FakeTransferHost(_core.FakeRoom(name="caller-human-agent"))
    running = asyncio.ensure_future(transfer._transfer.run(host, caller))
    await asyncio.sleep(0.05)
    await transfer._transfer.call_tool("connect_to_caller", {})
    identity = await running
    caller.leave("user")
    await asyncio.sleep(0.05)
    print(f"connected {identity}: {host.calls}")
    assert identity == "human-agent-sip"
    assert host.calls[:5] == [
        "hold",
        "io []",
        "connect caller-human-agent as agent",
        "start human",
        "sip +15550100 trunk=ST_smoke",
    ], host.calls
    assert host.calls[-1] == "delete caller room", host.calls
    assert "io ['audio_input', 'transcription_output']" in host.calls, "the io comes back as was"

    host = _FakeTransferHost(_core.FakeRoom(name="caller-human-agent"))
    running = asyncio.ensure_future(transfer._transfer.run(host, caller))
    await asyncio.sleep(0.05)
    await transfer._transfer.call_tool("decline_transfer", {"reason": "busy"})
    with contextlib.suppress(ToolError):
        await running
        raise AssertionError("a declined transfer connected")
    assert "close human drain=True" in host.calls, host.calls

    twilio = wf.TwilioConnectorWarmTransfer(
        "+15550111",
        twilio_from_number="+15550122",
        twilio_account_sid="AC_smoke",
        twilio_auth_token="token",
        twilio_call_token="CT_smoke",
        original_caller_number="+15550133",
    )
    human = _core.FakeRoom(name="caller-human-agent")
    host = _FakeTransferHost(human, http=[(400, '{"code": 21210}'), (201, '{"sid": "CA_smoke"}')])
    running = asyncio.ensure_future(twilio._transfer.run(host, caller))
    await asyncio.sleep(0.05)
    human.join("human-agent-sip")
    await asyncio.sleep(0.05)
    await twilio._transfer.call_tool("connect_to_caller", {})
    print(f"twilio {await running}: {host.calls}")
    posts = [c for c in host.calls if c.startswith("post")]
    assert posts == ["post Calls.json from=+15550133", "post Calls.json from=+15550122"], posts


async def decoder() -> None:
    """``AudioStreamDecoder``: a 16 kHz WAV comes out at its own rate with ``sample_rate=None``,
    as the reference's, and at the rate asked otherwise."""
    import io
    import wave

    from livekit.agents.utils.codecs import AudioStreamDecoder

    print("== decoder")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    for asked, expected in ((None, 16000), (24000, 24000)):
        decoder = AudioStreamDecoder(sample_rate=asked, format="audio/wav")
        decoder.push(wav.getvalue())
        decoder.end_input()
        rates = {frame.sample_rate async for frame in decoder}
        print(f"decoded at {rates} with sample_rate={asked}")
        assert rates == {expected}, (asked, rates)


QUICK_RETRIES = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)


class FlakyLLM(ScriptedLLM):
    """Answers once ``fails`` attempts have failed with a 503."""

    def __init__(self, fails: int) -> None:
        super().__init__()
        self.fails = fails
        self.calls = 0

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = QUICK_RETRIES,
        **_: Any,
    ) -> llm.LLMStream:
        return _FlakyStream(self, chat_ctx=chat_ctx, tools=[], conn_options=conn_options)


class _FlakyStream(_Stream):
    async def _run(self) -> None:
        flaky = self._llm
        assert isinstance(flaky, FlakyLLM)
        flaky.calls += 1
        if flaky.calls <= flaky.fails:
            raise APIStatusError("unavailable", status_code=503)
        await super()._run()


class FlakyTTS(SilentTTS):
    """Fails each sentence's first ``fails`` attempts with a 503; streams when ``streaming``."""

    def __init__(self, fails: int, *, streaming: bool = False) -> None:
        super().__init__()
        self._capabilities = tts.TTSCapabilities(streaming=streaming)
        self.fails = fails
        self.attempts: dict[str, int] = {}

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = QUICK_RETRIES
    ) -> tts.ChunkedStream:
        return _FlakyChunked(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = QUICK_RETRIES) -> tts.SynthesizeStream:
        return _FlakySynthesis(tts=self, conn_options=conn_options)

    def attempt(self, text: str) -> None:
        self.attempts[text] = self.attempts.get(text, 0) + 1
        if self.attempts[text] <= self.fails:
            raise APIStatusError("unavailable", status_code=503)


class _FlakyChunked(_Chunked):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, FlakyTTS)
        self._tts.attempt(self.input_text)
        await super()._run(output_emitter)


class _FlakySynthesis(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        text = "".join([t async for t in self.input if isinstance(t, str)])
        assert isinstance(self._tts, FlakyTTS)
        self._tts.attempt(text)
        output_emitter.start_segment(segment_id=shortuuid("seg_"))
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(text))))
        output_emitter.end_segment()


class FlakySTT(BatchSTT):
    def __init__(self, fails: int) -> None:
        super().__init__()
        self.fails = fails

    async def _recognize_impl(self, buffer: object, **kw: object) -> stt.SpeechEvent:
        if len(self.utterances) < self.fails:
            self.utterances.append(0.0)
            raise APIStatusError("unavailable", status_code=503)
        return await super()._recognize_impl(buffer, **kw)


def error_events(provider: Any) -> list[bool]:
    seen: list[bool] = []
    provider.on("error", lambda ev: seen.append(ev.recoverable))
    return seen


def metrics_events(provider: Any) -> list[Any]:
    seen: list[Any] = []
    provider.on("metrics_collected", seen.append)
    return seen


async def providers() -> None:
    """A provider called on its own retries per ``conn_options``, as the core's policy decides:
    each retried failure an ``error`` event marked recoverable, the last one not."""
    print("== standalone provider retries")
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    model = FlakyLLM(fails=2)
    seen = error_events(model)
    measured = metrics_events(model)
    reply = await model.chat(chat_ctx=chat_ctx).collect()
    assert (reply.text, model.calls, seen) == ("You said: hello", 3, [True, True]), (reply, seen)
    # a provider meters its own requests: one record for the request that answered
    assert [type(m).__name__ for m in measured] == ["LLMMetrics"], measured

    model = FlakyLLM(fails=9)
    seen = error_events(model)
    try:
        await model.chat(chat_ctx=chat_ctx).collect()
        raise AssertionError("an exhausted budget raises")
    except APIConnectionError as e:
        assert "after 3 attempts" in str(e), e
    assert (model.calls, seen) == (3, [True, True, False]), (model.calls, seen)

    voice = FlakyTTS(fails=1)
    seen = error_events(voice)
    measured = metrics_events(voice)
    frame = await voice.synthesize("hi").collect()
    assert frame.duration > 0 and voice.attempts == {"hi": 2} and seen == [True], seen
    assert [(type(m).__name__, m.characters_count) for m in measured] == [("TTSMetrics", 2)]

    # a retried stream is fed again what was pushed before the failure
    voice = FlakyTTS(fails=1, streaming=True)
    seen = error_events(voice)
    stream = voice.stream()
    stream.push_text("hello ")
    stream.push_text("world")
    stream.end_input()
    assert [ev async for ev in stream] and voice.attempts == {"hello world": 2}, voice.attempts
    assert seen == [True], seen

    ears = FlakySTT(fails=1)
    seen = error_events(ears)
    measured = metrics_events(ears)
    said = await ears.recognize([_square(0.0)], conn_options=QUICK_RETRIES)
    assert said.alternatives[0].text == "batch words" and seen == [True], seen
    assert [type(m).__name__ for m in measured] == ["STTMetrics"], measured

    # the core runs a gateway client as its own, each session with its job's identity, unless a
    # subclass overrides what the core would bypass
    from livekit.agents import _bindings, inference

    class Logged(inference.LLM):
        def chat(self, **kwargs: Any) -> llm.LLMStream:
            return super().chat(**kwargs)

    gateway = inference.LLM("openai/gpt-4.1-mini", api_key="k", api_secret="s")
    first, second = _bindings.core_leg(gateway), _bindings.core_leg(gateway)
    assert isinstance(first, _core.InferenceLlm) and first is not second, (first, second)
    logged = Logged("openai/gpt-4.1-mini", api_key="k", api_secret="s")
    assert _bindings.core_leg(logged) is logged

    # in a session, a voice that does not stream is retried sentence by sentence on the session's
    # tts_conn_options: each sentence's first attempt fails, and the whole reply is still spoken
    print("== a session's voice retries each sentence")
    from livekit.agents.voice.agent_session import SessionConnectOptions

    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    voice = FlakyTTS(fails=1)
    seen = error_events(voice)
    session = VoiceSession(
        stt=ScriptedSTT(),
        llm=ScriptedLLM(),
        tts=voice,
        vad=None,
        conn_options=SessionConnectOptions(tts_conn_options=QUICK_RETRIES),
    )
    await session.start(Agent(instructions="You retry.", name="retries"), room=room)
    await settle(room, 1)
    session.say("Hello there, this is the first sentence. And this is the second one.")
    await settle(room, 4)
    await session.aclose()
    assert len(voice.attempts) == 2 and set(voice.attempts.values()) == {2}, voice.attempts
    assert seen == [True, True], seen


async def llm_prewarm() -> None:
    """The gateway LLM is prewarmed as the session starts (``GET /models``, the reference's
    ``models.list()``), so a call's first reply finds a connection open."""
    from livekit.agents import inference

    print("== llm prewarm")
    seen: list[str] = []

    async def models(request: web.Request) -> web.Response:
        seen.append(f"{request.method} {request.path}")
        return web.json_response({"object": "list", "data": []})

    app = web.Application()
    app.router.add_get("/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    llm_ = inference.LLM(
        "openai/gpt-4.1-mini", base_url=f"http://127.0.0.1:{port}", api_key="k", api_secret="s"
    )
    session = VoiceSession(stt=ScriptedSTT(), llm=llm_, tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You answer."), room=room)
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.02)
    await session.aclose()
    await runner.cleanup()
    print(f"gateway saw {seen}")
    assert seen == ["GET /models"], seen


def agent_name(session: VoiceSession) -> str:
    return type(session.current_agent).__name__


_JOB_LOGS = """
import asyncio, logging, multiprocessing as mp, sys
from livekit.agents import Agent, cli
from livekit.agents.ipc.job_proc_executor import ProcJobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent, models


async def entrypoint(ctx):
    pii = Agent(instructions="x", name="pii")
    logging.getLogger("livekit.agents").warning("from the job", extra={"lk.pii.output": pii})
    ctx.shutdown("done")


async def main(style):
    cli.setup_logging("INFO", style)
    proc = ProcJobExecutor(
        job_entrypoint_fnc=entrypoint, session_end_fnc=None,
        simulation_end_fnc=None, inference_executor=None, initialize_timeout=60,
        close_timeout=10, session_end_timeout=10, memory_warn_mb=0, memory_limit_mb=0,
        ping_interval=2.5, ping_timeout=60, high_ping_threshold=1, http_proxy=None,
        mp_ctx=mp.get_context("spawn"), loop=asyncio.get_running_loop(),
    )
    await proc.start()
    await proc.initialize()
    room = models.Room(name="logs", sid="RM_logs")
    await proc.launch_job(RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=agent.Job(id="AJ_logs", room=room), url="", token="", worker_id="smoke",
        fake_job=True,
    ))
    await proc.join()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
"""


def job_logs() -> None:
    """A job process writes its own log lines to the stdout it inherits, as the worker does
    (JSON, and the dev layout), stamped with its pid and its job; an extra no JSON holds (an
    ``Agent``) as its string."""
    import os
    import tempfile

    from livekit.agents import Agent

    print("== a job process's own log lines")
    pii = str(Agent(instructions="x", name="pii")).split(" object at ")[0]
    with tempfile.TemporaryDirectory() as root:
        script = Path(root, "job_logs.py")
        script.write_text(_JOB_LOGS)
        for style in ("json", "dev"):
            run = subprocess.run(
                [sys.executable, str(script), style], capture_output=True, text=True, timeout=120
            )
            lines = [line for line in run.stdout.splitlines() if "from the job" in line]
            assert run.returncode == 0 and len(lines) == 1, (run.stdout, run.stderr)
            line = lines[0]
            if style == "json":
                record = json.loads(line)
                assert record["job_id"] == "AJ_logs" and record["room"] == "logs", record
                assert record["pid"] not in (os.getpid(), None), record
                assert record["lk.pii.output"].startswith(pii), record
            else:
                assert "AJ_logs" in line and pii in line, line
            print(f"{style}: the job's line, stamped with its job")


async def recording() -> None:
    """The session's audio recording, as a recorded job attaches it: written while the call runs
    and finished once the session closes."""
    print("== recording")
    import tempfile

    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(FrontDesk(), room=room)
    with tempfile.TemporaryDirectory() as directory:
        session._recording_path = Path(directory) / "audio.ogg"
        session._recorder = _core.SessionRecorder(session._recording_path)
        session._core.attach_recorder(session._recorder)
        session.say("Let me say something.")
        room.speak(frames=40)
        await settle(room, 6)
        await session.aclose()
        assert session._recording_finished is not None, "the close left the recording open"
        await session._recording_finished
        size = session._recording_path.stat().st_size
        assert size > 0 and session._recorder.started_at is not None, size
        print("recording: written and finished at the close")


async def noise_cancellation() -> None:
    """The room input's denoising, as the reference applies it: AGC is off when a processor is
    given directly, and the pre-connect head goes through the processor the live stream opened."""
    print("== noise cancellation")
    from livekit.agents.voice import room_io as _core_rtc

    class Halve(rtc.FrameProcessor[rtc.AudioFrame]):
        closed = False

        @property
        def enabled(self) -> bool:
            return True

        @enabled.setter
        def enabled(self, value: bool) -> None:
            pass

        def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
            data = bytes(b // 2 for b in bytes(frame.data.cast("B")))
            return rtc.AudioFrame(data, frame.sample_rate, 1, frame.samples_per_channel)

        def _close(self) -> None:
            self.closed = True

    def agc(nc: object) -> bool:
        options = room_io.RoomOptions(audio_input=room_io.AudioInputOptions(noise_cancellation=nc))
        return options._to_core()["audio_input"]["auto_gain_control"]  # type: ignore[no-any-return]

    assert agc(None) and agc(lambda params: None), "AGC is on without a direct processor"
    assert not agc(Halve()), "AGC is off with a processor given directly"

    class Publication:
        sid = "TR_mic"

    room = _core_rtc.Room(rtc.Room())
    processor = Halve()
    room._processors["TR_mic"] = processor
    head = [rtc.AudioFrame(b"\x10\x00" * 240, 24000, 1, 240)]
    publication = _core_rtc.TrackPublication(room, Publication())  # type: ignore[arg-type]
    out = await publication._core_process_audio(head)
    assert bytes(out[0].data.cast("B"))[:2] == b"\x08\x00", "the head skipped the processor"
    print("noise cancellation: AGC default and pre-connect head as the reference")


async def main() -> int:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)

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

    print("== chat from the room")
    assert room.send_chat("user", "typed hello", "chat_1"), "nobody reads lk.chat"
    await settle(room, 3)
    said = [i.text_content for i in session.history.items if getattr(i, "role", None) == "user"]
    assert "typed hello" in said, f"the chat message got no turn: {said}"

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
    await avatar_runner()
    await expressive()
    await background_audio()
    await amd()
    await filler()
    await tool_outcomes()
    await keyterms()
    await room_camera()
    await noise_cancellation()
    await recording()
    await console()
    await job()
    await workflows()
    await warm_transfer()
    await tools()
    await toolsets()
    await cli()
    await decoder()
    await llm_prewarm()
    await providers()
    job_logs()
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
    # the property-based seeds: random calls through the same session, on virtual time
    print("== fuzz.py")
    fuzz = subprocess.run([sys.executable, str(Path(__file__).with_name("fuzz.py"))])
    sys.exit(code or fuzz.returncode)
