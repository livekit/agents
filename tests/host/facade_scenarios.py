"""One scenario per facade class for `test_facades_freed.py`: each builds an object, uses it,
closes it and returns it, so the test can drop it and check that it is freed."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    APIConnectOptions,
    RunContext,
    VoiceSession,
    _core,
    inference,
    llm,
    stt,
    tokenize,
    tts,
)
from livekit.agents.a2a import Credentials
from livekit.agents.llm import _realtime
from livekit.agents.llm.mcp import MCPServerHTTP, MCPServerStdio
from livekit.agents.storage import SQLiteStorage, Thread
from livekit.agents.storage.context import ContextStore, persisted
from livekit.agents.storage.session_state import SessionState
from livekit.agents.telemetry import _tracer
from livekit.agents.utils.aio import ConnectionPool
from livekit.agents.voice.background_audio import PlayHandle
from tests.fakes import (
    FakeDuplex,
    FakeLLM,
    FakeRealtime,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    audio_frame,
    tool_call,
)

from .mcp_peer import serve_http, stdio_command

# An address nothing listens on: an inference request fails at once.
UNREACHABLE = "http://127.0.0.1:9"
NO_RETRY = APIConnectOptions(max_retry=0)

Scenario = Callable[[], Awaitable[object]]


async def _drained(stream: AsyncIterator[Any]) -> None:
    with contextlib.suppress(Exception):
        async for _ in stream:
            pass


@contextlib.asynccontextmanager
async def _storage() -> AsyncIterator[SQLiteStorage]:
    with tempfile.TemporaryDirectory() as directory:
        storage = SQLiteStorage(
            directory=Path(directory),
            lease_ttl=timedelta(seconds=2),
            renew_interval=timedelta(seconds=0.5),
            holder="worker-a",
        )
        try:
            yield storage
        finally:
            await storage.aclose()


async def credentials() -> object:
    return Credentials("APItest", "secret-secret-secret-secret-secret")


async def inference_llm() -> object:
    model = inference.LLM("openai/gpt-4o-mini", base_url=UNREACHABLE, api_key="k", api_secret="s")
    await _drained(model.chat(chat_ctx=llm.ChatContext.empty(), conn_options=NO_RETRY))
    await model.aclose()
    return model


async def gateway_llm_stream() -> object:
    model = inference.LLM("openai/gpt-4o-mini", base_url=UNREACHABLE, api_key="k", api_secret="s")
    stream = model.chat(chat_ctx=llm.ChatContext.empty(), conn_options=NO_RETRY)
    await _drained(stream)
    await stream.aclose()
    await model.aclose()
    return stream


async def inference_stt() -> object:
    model = inference.STT("deepgram/nova-3", base_url=UNREACHABLE, api_key="k", api_secret="s")
    await speech_stream_of(model)
    await model.aclose()
    return model


async def speech_stream_of(model: stt.STT) -> object:
    stream = model.stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame())
    stream.end_input()
    await _drained(stream)
    await stream.aclose()
    return stream


async def speech_stream() -> object:
    model = inference.STT("deepgram/nova-3", base_url=UNREACHABLE, api_key="k", api_secret="s")
    stream = await speech_stream_of(model)
    await model.aclose()
    return stream


async def inference_tts() -> object:
    model = inference.TTS("cartesia/sonic-3", base_url=UNREACHABLE, api_key="k", api_secret="s")
    await _drained(model.synthesize("hello", conn_options=NO_RETRY))
    await model.aclose()
    return model


async def inference_vad() -> object:
    model = inference.VAD()
    await vad_stream_of(model)
    await model.aclose()
    return model


async def vad_stream_of(model: Any) -> object:
    stream = model.stream()
    stream.push_frame(audio_frame())
    stream.end_input()
    await _drained(stream)
    await stream.aclose()
    return stream


async def inference_vad_stream() -> object:
    model = inference.VAD()
    stream = await vad_stream_of(model)
    await model.aclose()
    return stream


async def llm_fallback() -> object:
    adapter = llm.FallbackAdapter([FakeLLM(["Hi."])])
    await _drained(adapter.chat(chat_ctx=llm.ChatContext.empty()))
    await adapter.aclose()
    return adapter


async def llm_fallback_stream() -> object:
    adapter = llm.FallbackAdapter([FakeLLM(["Hi."])])
    stream = adapter.chat(chat_ctx=llm.ChatContext.empty())
    await _drained(stream)
    await stream.aclose()
    await adapter.aclose()
    return stream


async def _talked(model: llm.RealtimeModel | llm.LLM) -> None:
    async with VoiceSession(llm=model, vad=None) as session:
        await session.start(Agent(instructions="Help."))
        await session.generate_reply(instructions="greet the user")


async def openai_realtime() -> object:
    model = _realtime.RealtimeModel(base_url=UNREACHABLE, api_key="k")
    await model.aclose()
    return model


async def duplex_adapter() -> object:
    adapter = llm.DuplexRealtimeAdapter(FakeDuplex())
    await _talked(adapter)
    await adapter.aclose()
    return adapter


async def realtime_fallback() -> object:
    adapter = llm.RealtimeModelFallbackAdapter([FakeRealtime(), FakeRealtime()])
    await _talked(adapter)
    await adapter.aclose()
    return adapter


async def mcp_http() -> object:
    with tempfile.TemporaryDirectory() as directory:
        async with serve_http(Path(directory)) as peer:
            server = MCPServerHTTP(peer.url)
            await server.initialize()
            await server.list_tools()
            await server.aclose()
    return server


async def mcp_stdio() -> object:
    with tempfile.TemporaryDirectory() as directory:
        command, args = stdio_command(Path(directory))
        server = MCPServerStdio(command, args)
        await server.initialize()
        await server.list_tools()
        await server.aclose()
    return server


async def thread() -> object:
    async with _storage() as storage:
        thread_id = (await storage.threads.create_thread()).thread_id
        handle = Thread(storage, thread_id)
        await handle.info()
    return handle


async def partition() -> object:
    async with _storage() as storage:
        thread_id = (await storage.threads.create_thread()).thread_id
        part = Thread(storage, thread_id).partition("desk")
        await part.history()
    return part


async def transaction() -> object:
    async with _storage() as storage:
        thread_id = (await storage.threads.create_thread()).thread_id
        activation = await storage.threads._core_context_store(thread_id, "desk")
        store = ContextStore(activation=activation, thread=Thread(storage, thread_id))
        async with store.transaction() as tx:
            await tx.fetch("SELECT 1")
        store._close()
    return tx


@dataclass
class _App:
    session: SessionState = persisted(default_factory=SessionState)


async def context_store() -> object:
    store = ContextStore()
    app = await store.load_context(_App())
    await store.save_context(app)
    store._close()
    return store


async def session_binding() -> object:
    store = ContextStore()
    app = await store.load_context(_App())
    binding = store._bind_session(app)
    binding.close()
    store._close()
    return binding


async def session_state() -> object:
    state = SessionState()
    assert state.kind == "empty"
    return state


async def stt_fallback() -> object:
    adapter = stt.FallbackAdapter([FakeSTT(["hi"])])
    await recognize_stream_of(adapter)
    await adapter.aclose()
    return adapter


async def recognize_stream_of(model: stt.STT) -> object:
    stream = model.stream()
    stream.push_frame(audio_frame())
    stream.end_input()
    await _drained(stream)
    await stream.aclose()
    return stream


async def stt_fallback_stream() -> object:
    adapter = stt.FallbackAdapter([FakeSTT(["hi"])])
    stream = await recognize_stream_of(adapter)
    await adapter.aclose()
    return stream


async def stt_stream_adapter() -> object:
    adapter = stt.StreamAdapter(stt=FakeSTT(["hi"], streaming=False), vad=FakeVAD())
    await recognize_stream_of(adapter)
    await adapter.aclose()
    return adapter


async def stt_stream_adapter_stream() -> object:
    adapter = stt.StreamAdapter(stt=FakeSTT(["hi"], streaming=False), vad=FakeVAD())
    stream = await recognize_stream_of(adapter)
    await adapter.aclose()
    return stream


async def span() -> object:
    with _tracer.start_as_current_span("freed") as current:
        current.set_attribute("used", True)
    return current


async def sentence_tokenizer() -> object:
    tokenizer = tokenize.SentenceTokenizer()
    tokenizer.tokenize("Hello there. How are you today?")
    return tokenizer


async def word_tokenizer() -> object:
    tokenizer = tokenize.WordTokenizer()
    tokenizer.tokenize("Hello there.")
    return tokenizer


async def tts_fallback() -> object:
    adapter = tts.FallbackAdapter([FakeTTS()])
    await _drained(adapter.synthesize("hello"))
    await adapter.aclose()
    return adapter


async def tts_stream_adapter() -> object:
    adapter = tts.StreamAdapter(tts=FakeTTS(streaming=False))
    stream = adapter.stream()
    stream.push_text("Hello there.")
    stream.end_input()
    await _drained(stream)
    await stream.aclose()
    await adapter.aclose()
    return adapter


async def connection_pool() -> object:
    async def connect(_: APIConnectOptions) -> object:
        return object()

    async def close(_: object) -> None:
        pass

    pool = ConnectionPool[object](connect_cb=connect, close_cb=close)
    async with pool.connection(conn_options=DEFAULT_API_CONNECT_OPTIONS):
        pass
    await pool.aclose()
    return pool


async def voice_session() -> object:
    session = VoiceSession(llm=FakeLLM(["Hi."]))
    await session.start(Agent(instructions="Help."))
    await session.generate_reply(user_input="hello")
    await session.aclose()
    return session


async def play_handle() -> object:
    # a player needs a connected room to play: the handle over its play state
    handle = PlayHandle(_core.PlayState())
    handle.stop()
    await handle.wait_for_playout()
    return handle


async def run_context() -> object:
    seen: list[RunContext[list[Any]]] = []
    agent = Agent[list[Any]](
        instructions="Help.", name="desk", llm=FakeLLM([tool_call("note"), "Noted."])
    )

    @agent.tool
    async def note(ctx: RunContext[list[Any]]) -> str:
        """Note it."""
        ctx.context.append(ctx)
        return "noted"

    await agent.run("go", context=seen)
    return seen[0]


async def speech_handle() -> object:
    async with VoiceSession(llm=FakeLLM(["Hi."])) as session:
        await session.start(Agent(instructions="Help."))
        handle = session.generate_reply(user_input="hello")
        await handle
    return handle


async def run_stream() -> object:
    agent = Agent(instructions="Help.", llm=FakeLLM(["Hi."]))
    async with agent.run_stream("hello") as stream:
        await stream.result()
    return stream


SCENARIOS: dict[str, Scenario] = {
    "livekit.agents.a2a.Credentials": credentials,
    "livekit.agents.inference.VAD": inference_vad,
    "livekit.agents.inference._VADStream": inference_vad_stream,
    "livekit.agents.inference.llm.GatewayLLMStream": gateway_llm_stream,
    "livekit.agents.inference.llm.LLM": inference_llm,
    "livekit.agents.inference.stt.STT": inference_stt,
    "livekit.agents.inference.stt.SpeechStream": speech_stream,
    "livekit.agents.inference.tts.TTS": inference_tts,
    "livekit.agents.llm.FallbackAdapter": llm_fallback,
    "livekit.agents.llm.FallbackLLMStream": llm_fallback_stream,
    "livekit.agents.llm._realtime.RealtimeModel": openai_realtime,
    "livekit.agents.llm.duplex.DuplexRealtimeAdapter": duplex_adapter,
    "livekit.agents.llm.mcp.MCPServerHTTP": mcp_http,
    "livekit.agents.llm.mcp.MCPServerStdio": mcp_stdio,
    "livekit.agents.llm.realtime.RealtimeModelFallbackAdapter": realtime_fallback,
    "livekit.agents.storage.Partition": partition,
    "livekit.agents.storage.Thread": thread,
    "livekit.agents.storage.Transaction": transaction,
    "livekit.agents.storage.context.ContextStore": context_store,
    "livekit.agents.storage.context._SessionBinding": session_binding,
    "livekit.agents.storage.session_state.SessionState": session_state,
    "livekit.agents.stt.FallbackAdapter": stt_fallback,
    "livekit.agents.stt.FallbackRecognizeStream": stt_fallback_stream,
    "livekit.agents.stt.StreamAdapter": stt_stream_adapter,
    "livekit.agents.stt.StreamAdapterWrapper": stt_stream_adapter_stream,
    "livekit.agents.telemetry._Span": span,
    "livekit.agents.tokenize.SentenceTokenizer": sentence_tokenizer,
    "livekit.agents.tokenize.WordTokenizer": word_tokenizer,
    "livekit.agents.tts.FallbackAdapter": tts_fallback,
    "livekit.agents.tts.StreamAdapter": tts_stream_adapter,
    "livekit.agents.utils.aio.ConnectionPool": connection_pool,
    "livekit.agents.voice.agent_session.VoiceSession": voice_session,
    "livekit.agents.voice.background_audio.PlayHandle": play_handle,
    "livekit.agents.voice.events.RunContext": run_context,
    "livekit.agents.voice.io.SpeechHandle": speech_handle,
    "livekit.agents.voice.run_result.RunStream": run_stream,
}
