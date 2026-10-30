"""Smoke run of the copy, with no LiveKit server and no room (room coverage is the DST's): a
session with ``room=None``, with a scripted LLM, a scripted STT and a silent TTS written as
plugins (so all three cross the core's provider views). It starts, says something, replies,
runs a tool, explicitly switches to a second agent whose ``on_enter`` speaks, and closes; then
``agent.run()``, the realtime legs, ``lk agent console``'s socket, a job run over the IPC
socket, the CLI, the standalone provider retries and a job process's own log lines.

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
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import api, rtc
from livekit.agents import (
    Agent,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    RunContext,
    VoiceSession,
    _core,
    function_tool,
    llm,
    stt,
    tts,
)
from livekit.agents.llm._realtime import RealtimeModel as OpenAIRealtimeModel
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid
from livekit.agents.voice import SpeechHandle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.trace_schema import check_trace, from_readable_spans  # noqa: E402

SAMPLE_RATE = 24000

# what the main session's trace holds, from the core's span records (with no room there is no
# audio: no user turn is heard and no reply is synthesized)
SESSION_SPANS = {
    "agent_session",
    "session_start",
    "start_agent_activity",
    "on_enter",
    "agent_turn",
    "agent_speaking",
    "llm_node",
    "llm_request",
    "llm_request_run",
    "function_tool",
    "update_agent",
    "drain_agent_activity",
    "on_exit",
    "session_close",
}

# (words in the user's message, the tool it calls, its arguments from the message)
TOOL_CALLS = [
    ("weather", "get_weather", lambda text: {"city": "Paris"}),
    ("billing", "to_billing", lambda text: {}),
    ("order", "look_up_order", lambda text: {}),
    ("keypad", "send_dtmf_events", lambda text: {"events": ["1", "#"]}),
    ("bye", "end_call", lambda text: {}),
    ("book", "book", lambda text: {"day": "Monday"}),
    ("table", "reserve_table", lambda text: {"guests": 2}),
    ("receipt", "change_receipt_email", lambda text: {}),
    ("@", "confirm_email", lambda text: {"email_address": text.split()[-1]}),
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


async def until(done: Callable[[], bool], what: str, timeout: float = 10) -> None:
    """Wait, in real time, until ``done()``."""
    try:
        async with asyncio.timeout(timeout):
            while not done():
                await asyncio.sleep(0.01)
    except TimeoutError:
        raise AssertionError(what) from None


def billing() -> Agent:
    agent = Agent(instructions="You handle billing.", name="Billing")

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        session.say("Billing here, how can I help?")

    return agent


def front_desk() -> Agent:
    agent = Agent(instructions="You are the front desk.", name="FrontDesk")

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        session.say("Hello, front desk.")

    @agent.tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @agent.tool
    async def to_billing(ctx: RunContext) -> None:
        """Transfer the caller to billing."""
        assert ctx.live_session is not None
        ctx.live_session.update_agent(billing())

    return agent


def print_item(ev: object) -> None:
    item = ev.item  # type: ignore[attr-defined]
    if item.type == "message":
        print(f"conversation_item {item.role}: {item.text_content!r}")
    else:
        print(f"conversation_item {item.type}: {item.old_agent_id} -> {item.new_agent_id}")


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
    session = VoiceSession(llm=ScriptedRealtime(), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="realtime"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the realtime reply did not finish")
    await session.aclose()


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
    primary, backup = DyingRealtime(), ScriptedRealtime()
    adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
    changes: list[tuple[bool, bool]] = []
    adapter.on(
        "realtime_availability_changed",
        lambda ev: changes.append((ev.realtime_model is primary, ev.available)),
    )
    session = VoiceSession(llm=adapter, vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="fallback"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the backup model did not answer")
    assert changes == [(True, False)], changes
    await session.aclose()
    print(f"availability {changes}")


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
    model = ScriptedDuplex()
    session = VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are duplex.", name="duplex"), room=None)
    handle = session.generate_reply()
    await until(handle.done, "the duplex reply did not finish")
    script = model.sessions[0]
    assert script._configured.is_set(), "the configuration never reached the model as one unit"
    assert script.configured_with == ["You are duplex."], script.configured_with
    await session.aclose()


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


async def openai_protocol_realtime() -> None:
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

    model = OpenAIRealtimeModel(
        api_key="sk-local", base_url=f"http://127.0.0.1:{port}/v1", voice="cedar"
    )
    session = VoiceSession(llm=model, vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="openai_realtime"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the OpenAI-protocol realtime reply did not finish")
    assert sessions[0]["audio"]["output"]["voice"] == "cedar", sessions[0]
    await session.aclose()
    await runner.cleanup()
    # the connection's record, accounted in the realtime model's usage
    usage = session.usage
    assert usage.settled and [u.model for u in usage.models] == ["gpt-realtime"], usage
    print(f"usage_requests={usage.requests}")


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
    await session.start(front_desk())
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
    from livekit.agents._generated.worker_ipc import StartJobFields
    from livekit.agents.ipc.child import _JobProc
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

    runner = _JobProc(
        entrypoint,
        None,
        session_end_timeout=5.0,
        executor_type=JobExecutorType.THREAD,
    )

    class Parent:
        """The worker's side, the core's supervisor's host, with what the supervisor logs."""

        logs: list[tuple[str, dict[str, Any]]] = []

        def _core_log(self, level: int, msg: str, extra: dict[str, Any], exc: Any = None) -> None:
            self.logs.append((msg, extra))

        def __init__(self) -> None:
            self.endpoint: asyncio.Future[int] = asyncio.get_running_loop().create_future()

        async def _core_launch(self, endpoint: int) -> None:
            self.endpoint.set_result(endpoint)

        async def _core_completed(self, completion: Any) -> None:
            pass

    # the worker's side is the core's supervisor of a job thread: no pings, nothing killed; the
    # "thread" is this loop serving the job's end
    parent = Parent()
    sup = _core.JobSupervisor(
        parent,
        thread=True,
        initialize_timeout=30,
        close_timeout=30,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=0,
        ping_timeout=0,
        high_ping_threshold=10,
    )
    cch = await parent.endpoint
    initialized = asyncio.ensure_future(sup.initialize(asyncio_debug=False, http_proxy=None))
    client = await asyncio.to_thread(_core.IpcClient, cch, runner.initialize)
    await initialized
    runner._client = client
    serving = asyncio.ensure_future(client.serve(runner))
    request = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent", metadata=""),
        job=agent_proto.Job(id="AJ_smoke", room=models.Room(name="console", sid="RM_smoke")),
        url="",
        token="",
        worker_id="smoke",
        fake_job=True,
    )
    await sup.start_job(
        StartJobFields(
            job=request.job.SerializeToString(),
            name=request.accept_arguments.name,
            identity=request.accept_arguments.identity,
            metadata=request.accept_arguments.metadata,
            url=request.url,
            token=request.token,
            worker_id=request.worker_id,
            fake_job=request.fake_job,
            received_at=request.received_at,
            accepted_at=request.accepted_at,
            assigned_at=request.assigned_at,
            launched_at=request.launched_at,
        )
    )
    await asyncio.wait_for(serving, 30)
    assert runner._job_task is not None
    await asyncio.wait_for(runner._job_task, 30)
    await client.aclose()
    sup._core_thread_exited()
    assert await sup.run()
    print(f"job {seen} told its parent {[m for m, _ in Parent.logs]}")
    # the participant's entrypoint runs beside the entrypoint, in no set order with it
    assert sorted(seen) == [
        "delete_room DeleteRoomResponse",
        "participant mock_user",
        "shutdown callback 'done'",
        "shutdown callback, no reason",
        "waited for mock_user",
    ], seen
    assert ("job thread exiting", {"reason": "done"}) in Parent.logs, Parent.logs
    events, attributes = _core.job_timeline(1.0, 2.0, 0.0, 4.0, 5.0, 3_000_000_000_000_000_000)
    assert [e[0] for e in events][-1] == "job_started_on_server" and len(events) == 5, events
    assert dict(attributes) == {
        "lk.job.accept_latency": 1.0,
        "lk.job.entrypoint_latency": 1.0,
        "lk.job.dispatch_latency": 4.0,
    }, attributes


async def workflows() -> None:
    """The workflow checks (the core's)."""
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
    from livekit.agents._generated.worker_cli import CliArgsStart
    from livekit.agents.job import JobAcceptArguments, RunningJobInfo
    from livekit.protocol import agent as agent_proto
    from livekit.protocol.agent_pb import agent_dev

    print("== cli")
    args = _core.parse_cli(["start", "agent.py", "--dev", "--reload-addr", "127.0.0.1:1"])
    assert isinstance(args, CliArgsStart)
    assert (args.entrypoint, args.cli_addr, args.log_style) == (
        "agent.py",
        "127.0.0.1:1",
        "dev",
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
        _active_jobs = [
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
    await asyncio.wait_for(done.wait(), 10)
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


QUICK_RETRIES = APIConnectOptions(max_retry=2, retry_interval=0.0)


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
    llm_ = inference.LLM(
        "openai/gpt-4.1-mini", base_url=f"http://127.0.0.1:{port}", api_key="k", api_secret="s"
    )
    session = VoiceSession(stt=ScriptedSTT(), llm=llm_, tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You answer."), room=None)
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.02)
    await session.aclose()
    await runner.cleanup()
    print(f"gateway saw {seen}")
    assert seen == ["GET /models"], seen


_JOB_LOGS = """
import asyncio, logging, multiprocessing as mp, sys
from livekit.agents import Agent, cli
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent, models


async def entrypoint(ctx):
    pii = Agent(instructions="x", name="pii")
    logging.getLogger("livekit.agents").warning("from the job", extra={"lk.pii.output": pii})
    ctx.shutdown("done")


async def main(style):
    if style != "none":
        cli.setup_logging("INFO", style)
    proc = JobExecutor(
        job_entrypoint_fnc=entrypoint, session_end_fnc=None,
        simulation_end_fnc=None, initialize_timeout=60,
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
        # "none": a worker that set up no logging, its processes still log (JSON off a terminal)
        for style in ("json", "dev", "none"):
            run = subprocess.run(
                [sys.executable, str(script), style], capture_output=True, text=True, timeout=120
            )
            lines = [line for line in run.stdout.splitlines() if "from the job" in line]
            assert run.returncode == 0 and len(lines) == 1, (run.stdout, run.stderr)
            line = lines[0]
            if style in ("json", "none"):
                record = json.loads(line)
                assert record["job_id"] == "AJ_logs" and record["room"] == "logs", record
                assert record["pid"] not in (os.getpid(), None), record
                assert record["lk.pii.output"].startswith(pii), record
            else:
                assert "AJ_logs" in line and pii in line, line
            print(f"{style}: the job's line, stamped with its job")


async def main() -> int:
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    set_tracer_provider(provider)
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)

    def on_tools(ev: object) -> None:
        calls = [(c.name, o.output) for c, o in ev.zipped()]  # type: ignore[attr-defined]
        print(f"function_tools_executed {calls}")

    def on_state(ev: object) -> None:
        # the session's state is the event's by the time a listener runs
        assert session.agent_state == ev.new_state  # type: ignore[attr-defined]
        print(f"agent_state {ev.new_state}")  # type: ignore[attr-defined]

    session.on("agent_state_changed", on_state)
    session.on("conversation_item_added", print_item)
    session.on("function_tools_executed", on_tools)
    session.on("speech_created", lambda ev: print(f"speech_created {ev.source}"))
    session.on("close", lambda ev: print(f"close {ev.reason.value}"))

    print("== start")
    await session.start(front_desk(), room=None)

    print("== say")
    registered = asyncio.Event()

    async def gated() -> AsyncIterator[str]:
        # With no audio output a say can end before the next line runs; its words end only
        # once the callbacks below are registered, so the removal reaches a pending callback.
        await registered.wait()
        yield "Let me say something."

    handle = session.say(gated())
    # set before the core created the speech: the speech is created with it
    handle.allow_interruptions = False
    calls: list[str] = []
    handle.add_done_callback(lambda h: calls.append(f"first {h.id}"))
    handle.add_done_callback(dropped := lambda h: calls.append("dropped"))
    handle.add_done_callback(lambda h: calls.append("second"))
    handle.remove_done_callback(dropped)
    registered.set()
    # the callbacks run when the session reports the speech done, after `done()` turns true
    await until(lambda: len(calls) >= 2, "say()'s done callbacks did not run")
    assert handle.done() and not handle.allow_interruptions, "the setter before creation was lost"
    assert calls == [f"first {handle.id}", "second"], calls
    assert handle.exception() is None
    late: list[str] = []
    handle.add_done_callback(lambda h: late.append("late"))
    await asyncio.sleep(0)
    assert late == ["late"], "a callback added after done runs soon"

    print("== wait_if_not_interrupted")

    async def unfinished() -> AsyncIterator[str]:
        # Words that never end, so the speech is still running when it is interrupted: with no
        # audio output a fixed sentence can end before `interrupt()`, which is then a no-op.
        yield "A long sentence the agent will not finish saying."
        await asyncio.Event().wait()

    handle = session.say(unfinished())
    work: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(handle.wait_if_not_interrupted([work]))
    handle.interrupt()
    await until(waiting.done, "an interrupt ends the wait")
    assert not work.done(), "an interrupt ends the wait, never the work"
    work.cancel()
    assert SpeechHandle.SPEECH_PRIORITY_LOW < SpeechHandle.SPEECH_PRIORITY_HIGH

    print("== say, streamed words")

    async def words() -> AsyncIterator[str]:
        for word in ("Streamed ", "words."):
            yield word

    handle = session.say(words())
    await until(handle.done, "say(AsyncIterable) did not finish")

    print("== generate_reply")
    handle = session.generate_reply(user_input="hello there")
    await until(handle.done, "generate_reply did not finish")

    print("== a reply, asserted")
    result = await asyncio.wait_for(session._run(user_input="what's the weather like?"), 10)
    assert result.status == "completed", result.status
    result.expect.next_event().is_function_call(name="get_weather")
    result.expect.next_event().is_function_call_output(output="sunny in Paris")
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()
    print(f"run items: {[item.type for item in result.new_messages()]}")

    print("== handoff")
    session.generate_reply(user_input="put me through to billing")
    await until(lambda: session.current_agent.name == "Billing", "no handoff to billing")

    print("== close")
    await session.aclose()

    print("== trace")
    finished = spans.get_finished_spans()
    names = {s.name for s in finished}
    print(f"spans: {sorted(names)}")
    violations = check_trace(from_readable_spans(finished))
    assert not violations, "trace shape violations:\n  " + "\n  ".join(violations)
    assert SESSION_SPANS <= names, f"spans never emitted: {sorted(SESSION_SPANS - names)}"
    spans.clear()

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

    print("== OpenAI-protocol realtime")
    await openai_protocol_realtime()

    await console()
    await job()
    await workflows()
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
    sys.exit(code)
