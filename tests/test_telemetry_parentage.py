"""Application spans opened inside framework-invoked callbacks nest under the framework span of
that callback, as the Rust exporter sends them to an application OTLP endpoint.

One table: each row is an entry point and the framework span an application span opened in it
must be a direct child of (`docs/internals/telemetry-reference.md`, span trees; 1.8.10's parents).
Text-run tools and instructions are covered by `test_telemetry_invocations.py`, and the job
entrypoint by `conformance/test_telemetry_transport.py`."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import gzip
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import pytest
from aiohttp import web
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from livekit import rtc
from livekit.agents import (
    Agent,
    JobContext,
    ModelSettings,
    RunContext,
    VoiceSession,
    _core,
    llm,
    stt,
    telemetry,
)
from livekit.agents.a2a import AgentClient
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall
from livekit.protocol import agent as agent_proto

from .conformance.conftest import own_executor  # noqa: F401  (the fixture)
from .fake_io import FakeAudioInput, FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD
from .test_a2a_delegation import OPTIONS, Peer, peer  # noqa: F401  (the fixture)

pytestmark = pytest.mark.unit

# entry point -> the framework span its application span is a direct child of
EXPECTED_PARENT = {
    "on_enter": "on_enter",
    "on_user_turn_completed": "on_user_turn_completed",
    "llm_node": "llm_node",
    # opened after a `yield`, inside a span the generator holds across its yields
    "llm_node.after_yield": "app.llm_node",
    "tool": "function_tool",
    "tts_node": "tts_node",
    "on_exit": "on_exit",
}

# 1.8.10 runs the STT pump, and so `stt_node`, under `start_agent_activity`.
STT_NODE_PARENT = "start_agent_activity"

_app = telemetry.TracerProvider().get_tracer("app")
_marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")


class _Collector:
    """An OTLP/HTTP endpoint that keeps every span it receives, by hex span id, in the OTLP/JSON
    shape."""

    def __init__(self) -> None:
        self.spans: dict[str, dict[str, Any]] = {}

    async def traces(self, request: web.Request) -> web.Response:
        body = await request.read()
        if request.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        message = ExportTraceServiceRequest.FromString(body)
        for rs in MessageToDict(message, use_integers_for_enums=True).get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    for key in ("traceId", "spanId", "parentSpanId"):
                        if key in span:
                            span[key] = base64.b64decode(span[key]).hex()
                    self.spans[span["spanId"]] = span
        return web.json_response({})

    async def metrics(self, request: web.Request) -> web.Response:
        return web.json_response({})

    def named(self, name: str) -> list[dict[str, Any]]:
        return [s for s in self.spans.values() if s["name"] == name]

    def parent_name(self, span: dict[str, Any]) -> str | None:
        parent = self.spans.get(span.get("parentSpanId", ""))
        return parent["name"] if parent else None


@pytest.fixture
async def collector() -> AsyncIterator[_Collector]:
    sink = _Collector()
    app = web.Application()
    app.router.add_post("/v1/traces", sink.traces)
    app.router.add_post("/v1/metrics", sink.metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    telemetry.configure(endpoint=f"http://127.0.0.1:{port}")
    _core._core_telemetry_install("test")
    try:
        yield sink
    finally:
        telemetry.configure(endpoint="")
        await runner.cleanup()


def _opened(entry: str) -> Iterator[Any]:
    return _app.start_as_current_span(f"app.{entry}")  # type: ignore[return-value]


async def _settle(sink: _Collector, names: set[str], timeout: float = 8.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if all(sink.named(n) for n in names):
            return
        await asyncio.sleep(0.2)


async def test_application_spans_nest_under_their_callbacks_span(collector: _Collector) -> None:
    seen, seen_after_yield = await _run(collector)
    expected = {entry: [parent] for entry, parent in EXPECTED_PARENT.items()}
    assert {entry: seen[entry] for entry in EXPECTED_PARENT} == expected
    assert set(seen_after_yield) == {"before the first yield", "set by the tts_node call"}


async def test_an_stt_node_span_nests_under_the_activity_start(collector: _Collector) -> None:
    seen, _ = await _run(collector)
    assert seen["stt_node"] == [STT_NODE_PARENT]


async def test_a_tool_and_llm_node_under_a_fallback_llm(collector: _Collector) -> None:
    seen, _ = await _run(collector, wrap=lambda model: llm.FallbackAdapter([model]))
    assert (seen["tool"], seen["llm_node"]) == (["function_tool"], ["llm_node"])


async def test_a_delegation_from_a_tool_continues_under_its_function_tool(
    collector: _Collector,
    peer: Peer,  # noqa: F811  (the imported fixture)
) -> None:
    async def delegate() -> None:
        client = AgentClient(peer.url, conn_options=OPTIONS)
        try:
            reply = await client.send("Check my coverage")
            async for _ in reply:
                pass
        finally:
            await client.aclose()

    await _run(collector, delegate=delegate)
    sends = [headers for method, headers, rpc in peer.requests if rpc]
    assert sends, peer.requests
    _, trace_id, span_id, _ = sends[0]["traceparent"].split("-")
    tool = collector.named("function_tool")
    assert tool and all(t["traceId"] == trace_id for t in tool)
    # The request's span is the tool's, or a client span directly under it.
    caller = collector.spans.get(span_id)
    assert span_id in {t["spanId"] for t in tool} or (
        caller is not None and collector.parent_name(caller) == "function_tool"
    ), (span_id, caller)


async def _run(
    collector: _Collector,
    *,
    wrap: Callable[[llm.LLM], llm.LLM] = lambda model: model,
    delegate: Callable[[], Awaitable[None]] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """One call through every callback: the parents each entry's application spans got."""
    speech = [FakeUserSpeech(start_time=0.1, end_time=0.5, transcript="look it up", stt_delay=0.05)]
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="look it up",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="call")],
            ),
            FakeLLMResponse(input="found", content="Here it is.", ttft=0, duration=0),
        ]
    )
    session = VoiceSession(
        vad=FakeVAD(fake_user_speeches=speech, min_silence_duration=0.1),
        stt=FakeSTT(fake_user_speeches=speech),
        llm=wrap(model),
        tts=FakeTTS(fake_audio_duration=0.1),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    session.output.audio = FakeAudioOutput()
    agent = Agent(instructions="Help the caller.", name="desk")
    replied = asyncio.Event()
    seen_after_yield: list[str] = []

    @agent.on_enter
    async def entered(session: VoiceSession) -> None:
        with _opened("on_enter"):
            await asyncio.sleep(0)

    @agent.on_exit
    async def exited(session: VoiceSession) -> None:
        with _opened("on_exit"):
            await asyncio.sleep(0)

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the caller's record."""
        with _opened("tool"):
            await asyncio.sleep(0)
        if delegate is not None:
            await delegate()
        return "found"

    @session.on_user_turn_completed
    async def turn_done(ctx: RunContext, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        with _opened("on_user_turn_completed"):
            await asyncio.sleep(0)

    @session.stt_node
    async def hear(
        ctx: RunContext, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterator[stt.SpeechEvent]:
        async for event in session.default.stt_node(ctx, audio, model_settings):
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                with _opened("stt_node"):
                    pass
            yield event

    @session.llm_node
    async def think(
        ctx: RunContext,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[Any]:
        # A span and a contextvar held across the generator's yields: every pull and the close
        # run in one context, so the span's token resets and the variable survives.
        with _opened("llm_node"):
            _marker.set("before the first yield")
            yielded = False
            async for chunk in session.default.llm_node(ctx, chat_ctx, tools, model_settings):
                yield chunk
                if not yielded:
                    yielded = True
                    seen_after_yield.append(_marker.get())
                    with _opened("llm_node.after_yield"):
                        pass

    # A plain callback returning its generator: the generator is pulled in the context the call
    # ran in, so what the call set before returning is visible in its body.
    @session.tts_node
    def speak(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        _marker.set("set by the tts_node call")

        async def frames() -> AsyncIterator[rtc.AudioFrame]:
            seen_after_yield.append(_marker.get())
            with _opened("tts_node"):
                await asyncio.sleep(0)
            async for frame in session.default.tts_node(ctx, text, model_settings):
                yield frame

        return frames()

    session.on(
        "conversation_item_added",
        lambda ev: (
            ev.item.type == "message"
            and ev.item.role == "assistant"
            and "Here it is" in (ev.item.text_content or "")
            and replied.set()
        ),
    )
    async with session:
        await session.start(agent)
        audio_input.push(0.1)
        await asyncio.wait_for(replied.wait(), timeout=10)
    entries = [*EXPECTED_PARENT, "stt_node"]
    await _settle(collector, {f"app.{e}" for e in entries} | {"agent_session"})
    seen = {
        entry: sorted(
            {collector.parent_name(s) or "<none>" for s in collector.named(f"app.{entry}")}
        )
        for entry in entries
    }
    return seen, seen_after_yield


# The provider request spans and the node spans they nest under, by how the step's nodes run.
REQUEST_PARENTS = {
    # a node making its own output makes no provider request
    "custom": {},
    # a node running the default makes the request, under its node span
    "delegating": {"llm_request": "llm_node", "tts_request": "tts_node"},
    # with no node registered the pipeline makes it, under the same spans
    "default": {"llm_request": "llm_node", "tts_request": "tts_node"},
}
# Node spans describe the callback; the model and provider are their requests'.
REQUEST_ONLY_KEYS = {"gen_ai.request.model", "gen_ai.provider.name", "gen_ai.response.model"}


# Catches: a request span opened around a registered node (a custom node's output reported as a
# provider request), a node running the default losing its request or the step's identity, or a
# node span carrying its request's model.
@pytest.mark.parametrize("nodes", list(REQUEST_PARENTS))
async def test_a_provider_request_nests_under_the_node_that_makes_it(
    collector: _Collector, nodes: str
) -> None:
    model = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="Hi there.", ttft=0, duration=0)]
    )
    tts_model = FakeTTS(fake_audio_duration=0.1)
    session = VoiceSession(llm=model, tts=tts_model, aec_warmup_duration=None)
    session.output.audio = FakeAudioOutput()

    if nodes == "custom":

        @session.llm_node
        async def think(
            ctx: RunContext, chat_ctx: ChatContext, tools: list[llm.Tool], settings: ModelSettings
        ) -> AsyncIterator[str]:
            yield "Hi there."

        @session.tts_node
        async def speak(
            ctx: RunContext, text: AsyncIterable[str], settings: ModelSettings
        ) -> AsyncIterator[rtc.AudioFrame]:
            async for _ in text:
                pass
            yield rtc.AudioFrame.create(tts_model.sample_rate, 1, tts_model.sample_rate // 10)

    elif nodes == "delegating":

        @session.llm_node
        def think(
            ctx: RunContext, chat_ctx: ChatContext, tools: list[llm.Tool], settings: ModelSettings
        ) -> AsyncIterator[Any]:
            return session.default.llm_node(ctx, chat_ctx, tools, settings)

        @session.tts_node
        def speak(
            ctx: RunContext, text: AsyncIterable[str], settings: ModelSettings
        ) -> AsyncIterator[rtc.AudioFrame]:
            return session.default.tts_node(ctx, text, settings)

    async with session:
        await session.start(Agent(instructions="Greet the caller.", name="desk"))
        await session.generate_reply(user_input="hello")
    await _settle(collector, {"llm_node", "tts_node", "agent_session"})

    expected = REQUEST_PARENTS[nodes]
    for name in ("llm_request", "tts_request"):
        parents = {collector.parent_name(s) for s in collector.named(name)}
        assert parents == ({expected[name]} if name in expected else set()), (name, parents)
    for name in ("llm_node", "tts_node"):
        (node,) = collector.named(name)
        keys = {a["key"] for a in node.get("attributes", [])}
        assert not keys & REQUEST_ONLY_KEYS, (name, keys)
    for request in collector.named("llm_request") + collector.named("tts_request"):
        node = collector.spans[request["parentSpanId"]]
        conversation = _attr(node, "gen_ai.conversation.id")
        assert conversation and _attr(request, "gen_ai.conversation.id") == conversation
        assert _attr(request, "gen_ai.request.model") is not None


def _attr(span: dict[str, Any], key: str) -> Any:
    return next((a["value"] for a in span.get("attributes", []) if a["key"] == key), None)


async def test_job_callbacks_spans_nest_under_their_job_spans(
    collector: _Collector,
    own_executor: Callable[[JobExecutor], JobExecutor],  # noqa: F811  (the imported fixture)
) -> None:
    async def entrypoint(ctx: JobContext) -> None:
        with _opened("entrypoint"):
            await asyncio.sleep(0)
        ctx.shutdown("done")

    async def session_end(ctx: JobContext) -> None:
        with _opened("on_session_end"):
            await asyncio.sleep(0)

    executor = own_executor(
        JobExecutor(
            thread=True,
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=session_end,
            simulation_end_fnc=None,
            initialize_timeout=10,
            close_timeout=10,
            session_end_timeout=5,
            ping_interval=0,
            high_ping_threshold=1,
            http_proxy=None,
            loop=asyncio.get_running_loop(),
        )
    )
    try:
        await executor.start()
        await executor.initialize()
        await executor.launch_job(
            RunningJobInfo(
                JobAcceptArguments("", "", ""), agent_proto.Job(id="AJ_parent"), "", "", "", True
            )
        )
        await asyncio.wait_for(executor.join(), 10)
    finally:
        await executor.aclose()
    await _settle(collector, {"app.entrypoint", "app.on_session_end"})
    parents = {
        entry: sorted(
            {collector.parent_name(s) or "<none>" for s in collector.named(f"app.{entry}")}
        )
        for entry in ("entrypoint", "on_session_end")
    }
    assert parents == {"entrypoint": ["job_entrypoint"], "on_session_end": ["on_session_end"]}


async def test_spans_still_export_after_a_server_in_this_process_shut_down(
    collector: _Collector,
) -> None:
    # `AgentServer` flushes the process exporter when it stops; a server started afterwards in
    # the same process (tests, embedded use) installs a new one.
    await telemetry.shutdown()
    _core._core_telemetry_install("test")
    with _opened("after_shutdown"):
        pass
    await _settle(collector, {"app.after_shutdown"})
    assert collector.named("app.after_shutdown")
