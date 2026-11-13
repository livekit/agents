"""The context user Python runs in when the core calls it: the caller's event loop and
ContextVars, the application's own context object, and the span its own spans nest under."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import gzip
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TypedDict

import httpx
import pytest
from aiohttp import web
from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from livekit import rtc
from livekit.agents import (
    Agent,
    APIError,
    JobContext,
    ModelSettings,
    RunContext,
    SessionState,
    TextSession,
    VoiceSession,
    llm,
    persisted,
    telemetry,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.storage import ContextStore, Storage, Thread, TooLarge
from livekit.agents.types import FlushSentinel
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.protocol import agent as agent_proto
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    Hang,
    tool_call,
)

from .a2a_door import message_send
from .conftest import A2AHandler, RunJob

request_label: contextvars.ContextVar[str] = contextvars.ContextVar("request_label")


def _span(trace_id: int, span_id: int) -> trace.NonRecordingSpan:
    return trace.NonRecordingSpan(trace.SpanContext(trace_id, span_id, False, trace.TraceFlags(1)))


# -- the caller's loop and ContextVars


async def test_callbacks_run_on_the_callers_loop_with_its_context_vars() -> None:
    """Instructions and tool callbacks run on the loop that called `agent.run`, each run with
    the ContextVars its caller set."""
    loop = asyncio.get_running_loop()
    seen: list[tuple[str, bool, str]] = []
    agent = Agent(
        instructions="Help.",
        llm=FakeLLM(lambda ctx: tool_call("lookup") if len(ctx.items) < 3 else "ok"),
    )

    @agent.instructions
    def instructions(ctx: RunContext) -> str:
        seen.append(("instructions", asyncio.get_running_loop() is loop, request_label.get()))
        return "Look it up."

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look it up."""
        seen.append(("tool", asyncio.get_running_loop() is loop, request_label.get()))
        return "found"

    async def run(label: str) -> None:
        request_label.set(label)
        await agent.run("go")

    await asyncio.gather(run("first"), run("second"))

    assert sorted(seen) == sorted(
        (callback, True, label)
        for callback in ("instructions", "instructions", "tool")
        for label in ("first", "second")
    )


async def test_a_node_generator_is_pulled_in_the_context_its_node_ran_in() -> None:
    """A generator a `tts_node` returns is pulled on the session's loop, seeing what the node
    call set; an `llm_node` generator keeps its ContextVars across its yields."""
    loop = asyncio.get_running_loop()
    seen: list[tuple[str, bool, str]] = []
    session = VoiceSession(vad=None, llm=FakeLLM(["Hello there."]), tts=FakeTTS())
    session.output.audio = FakeAudioOutput()

    @session.llm_node
    async def think(
        ctx: RunContext, chat_ctx: ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        request_label.set("set before the first yield")
        async for chunk in session.default.llm_node(ctx, chat_ctx, tools, model_settings):
            yield chunk
        seen.append(("llm_node", asyncio.get_running_loop() is loop, request_label.get()))

    @session.tts_node
    def speak(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        request_label.set("set by the tts_node call")

        async def frames() -> AsyncIterator[rtc.AudioFrame]:
            seen.append(("tts_node", asyncio.get_running_loop() is loop, request_label.get()))
            async for frame in session.default.tts_node(ctx, text, model_settings):
                yield frame

        return frames()

    async with session:
        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="hello")

    assert sorted(seen) == [
        ("llm_node", True, "set before the first yield"),
        ("tts_node", True, "set by the tts_node call"),
    ]


async def test_each_a2a_request_runs_in_a_copy_of_the_servers_context(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    """A handler sees the ContextVars the server started with; what one request sets is not
    what the next one sees."""
    seen: list[str] = []

    async def handler(ctx: A2ASessionContext) -> str:
        seen.append(request_label.get("missing"))
        request_label.set(ctx.message.text_content or "")
        return "ok"

    token = request_label.set("server")
    try:
        async with a2a_door(handler) as door:
            await door.post("/v1", json=message_send("first", "m1"))
            await door.post("/v1", json=message_send("second", "m2"))
    finally:
        request_label.reset(token)

    assert seen == ["server", "server"]


# -- the application's context object


@dataclass
class Booking:
    nights: int = 0


async def test_a_calls_tools_get_the_sessions_context_object_and_thread(
    thread_storage: Storage,
) -> None:
    """A `VoiceSession`'s tools receive its very `context` object and its thread."""
    thread_id = (await thread_storage.threads.create_thread()).thread_id
    booking = Booking()
    seen: list[tuple[Booking, str]] = []
    agent = Agent(instructions="Help.", name="desk")

    @agent.tool
    async def note(ctx: RunContext[Booking]) -> str:
        """Note the booking."""
        assert ctx.thread is not None
        seen.append((ctx.context, ctx.thread.id))
        return "noted"

    model = FakeLLM([tool_call("note"), "ok"])
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    async with VoiceSession(llm=model, thread=thread, context=booking) as session:
        await session.start(agent)
        await session.generate_reply(user_input="go")

    [(context, seen_thread)] = seen
    assert context is booking
    assert seen_thread == thread_id


async def test_a_child_agent_shares_its_parents_context_object() -> None:
    """A child agent called as a tool works on the same context object as its parent."""
    nights = Booking()
    child = Agent(
        name="child",
        instructions="Record the nights.",
        output_type=str,
        llm=FakeLLM([tool_call("record", nights=3), "Recorded."]),
    )

    @child.tool
    async def record(ctx: RunContext[Booking], nights: int) -> str:
        """Record the nights."""
        ctx.context.nights = nights
        return "ok"

    parent = Agent(
        name="parent",
        instructions="Delegate.",
        tools=[child.as_tool()],
        llm=FakeLLM([tool_call("child"), "Booked for three nights."]),
    )

    await TextSession(agent=parent).run("Book three nights", context=nights)

    assert nights.nights == 3


@dataclass
class BoundApp:
    session: SessionState = persisted(default_factory=SessionState)
    count: int = persisted(default=0)


async def test_a_bound_run_updates_the_applications_own_session_state() -> None:
    """With `store.bind(app)`, tools get `app` itself and `app.session` follows the
    conversation in place; a snapshot loaded before does not."""
    store = ContextStore()
    app = await store.load_context(BoundApp())
    session_state = app.session
    snapshot = await store.load_session()
    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM([tool_call("inspect"), "done"]))

    @agent.tool
    async def inspect(ctx: RunContext[BoundApp]) -> str:
        """Inspect the app."""
        assert ctx.context is app
        app.count += 1
        return "seen"

    await TextSession(agent=agent).run("start", context=store.bind(app))

    assert app.count == 1
    assert app.session is session_state
    assert session_state.active_agent == "desk"
    assert snapshot.kind == "empty"


async def test_a_bound_run_that_fails_to_set_up_leaves_the_store_bindable() -> None:
    """A run that fails before it starts (no LLM) releases its binding: the next run binds the
    same store and completes."""
    store = ContextStore()
    app = await store.load_context(BoundApp())

    with pytest.raises(RuntimeError, match="has no LLM"):
        await TextSession(agent=Agent(name="desk", instructions="Help.")).run(
            "start", context=store.bind(app)
        )
    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM(["Hello."]))
    turn = await TextSession(agent=agent).run("start", context=store.bind(app))

    assert turn.result.status == "completed"
    assert app.session.active_agent == "desk"


async def test_a_bound_store_continues_one_conversation_across_runs() -> None:
    """Runs bound to the same store continue one conversation: the second run's model sees the
    first exchange."""
    store = ContextStore()
    app = await store.load_context(BoundApp())
    model = FakeLLM(["First.", "Second."])
    session = TextSession(agent=Agent(name="desk", instructions="Help.", llm=model))

    first = await session.run("hello", context=store.bind(app))
    second = await session.run("again", context=store.bind(app))

    assert second.state.session_id == first.state.session_id
    said = [item.text_content for item in model.requests[1].items if item.type == "message"]
    assert said[-3:] == ["hello", "First.", "again"]


async def test_a_plain_context_starts_a_new_conversation_each_run() -> None:
    """Without `store.bind`, each run starts its own conversation and the store's session is
    left untouched."""
    store = ContextStore()
    app = await store.load_context(BoundApp())
    session = TextSession(
        agent=Agent(name="desk", instructions="Help.", llm=FakeLLM(["One.", "Two."]))
    )

    first = await session.run("same", context=app)
    second = await session.run("same", context=app)

    assert first.state.session_id != second.state.session_id
    assert app.session.kind == "empty"
    assert (await store.load_session()).kind == "empty"


@pytest.mark.parametrize("application", [None, object()], ids=["no context", "plain object"])
async def test_a_binding_without_a_session_field_continues_the_conversation(
    application: object,
) -> None:
    """A store bound to no application object, or to one without a `SessionState` field, still
    carries the conversation from one run to the next."""
    store = ContextStore()
    session = TextSession(
        agent=Agent(name="desk", instructions="Help.", llm=FakeLLM(["One.", "Two."]))
    )
    binding = store.bind() if application is None else store.bind(application)

    first = await session.run("one", context=binding)
    second = await session.run("two", context=binding)

    assert first.state.session_id == second.state.session_id


async def test_binding_a_context_that_was_never_loaded_fails_before_the_run() -> None:
    """`store.bind(app)` of an object `load_context` never hydrated is refused, and the model is
    never called."""
    model = FakeLLM()
    session = TextSession(agent=Agent(name="desk", instructions="Help.", llm=model))

    with pytest.raises(ValueError, match="hydrate"):
        await session.run("hello", context=ContextStore().bind(BoundApp()))

    assert model.requests == []


async def test_a_second_binding_while_a_run_holds_the_store_is_refused() -> None:
    """While one bound run is in progress, another run binding the same store fails at once."""
    store = ContextStore()
    app = await store.load_context(BoundApp())
    hang = Hang()
    session = TextSession(agent=Agent(name="desk", instructions="Help.", llm=FakeLLM([hang])))

    async with session.run_stream("hello", context=store.bind(app)):
        await hang.started.wait()
        with pytest.raises(ValueError, match="active session binding"):
            await session.run("again", context=store.bind(app))


async def test_a_snapshot_taken_while_a_tool_runs_is_not_resumed() -> None:
    """A session read while a tool's outcome is still unknown cannot be resumed: the tool might
    have taken effect, so the run is refused rather than replayed."""
    store = ContextStore()
    app = await store.load_context(BoundApp())
    entered = asyncio.Event()
    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM([tool_call("charge")]))

    @agent.tool
    async def charge() -> str:
        """Charge the card."""
        entered.set()
        await asyncio.Future()
        return "unreachable"

    async with TextSession(agent=agent).run_stream("start", context=store.bind(app)):
        await entered.wait()
        unknown = await store.load_session()

    with pytest.raises(ValueError, match="reconcile"):
        await TextSession(agent=agent).run("continue", state=unknown)


async def test_a_failed_turn_saves_none_of_the_context_its_tools_saved(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    """A text turn that fails after a tool saved context fields commits none of them: the next
    request on the thread loads the fields as they were before."""
    model = FakeLLM([tool_call("increment"), APIError("the provider failed", retryable=False)])
    agent = Agent(name="desk", instructions="Help.", llm=model)
    loaded: list[int] = []

    @agent.tool
    async def increment(ctx: RunContext[BoundApp]) -> str:
        """Count one more."""
        ctx.context.count += 1
        await ctx.store.save_context(ctx.context)
        return "counted"

    async def handler(ctx: A2ASessionContext) -> object:
        app = await ctx.store.load_context(BoundApp())
        loaded.append(app.count)
        if len(loaded) > 1:
            return "loaded"
        turn = await TextSession(agent=agent).run(ctx.message, context=ctx.store.bind(app))
        return turn.result

    async with a2a_door(handler) as door:
        failed = await door.post("/v1", json=message_send("count", "m1"))
        context_id = failed.json()["result"]["task"]["contextId"]
        await door.post("/v1", json=message_send("again", "m2", context_id=context_id))

    assert failed.json()["result"]["task"]["status"]["state"] == "TASK_STATE_FAILED"
    assert loaded == [0, 0]


@dataclass
class NotesApp:
    notes: str = persisted(default="")


# Catches: a context save over 64 KiB committed, or failing the turn instead of raising TooLarge.
async def test_a_context_save_over_64_kib_is_refused_and_commits_nothing(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    refused: list[TooLarge] = []
    loaded: list[str] = []

    async def handler(ctx: A2ASessionContext) -> object:
        app = await ctx.store.load_context(NotesApp())
        loaded.append(app.notes)
        if len(loaded) > 1:
            return "loaded"
        app.notes = "small"
        await ctx.store.save_context(app)
        app.notes = "x" * (64 * 1024)
        try:
            await ctx.store.save_context(app)
        except TooLarge as error:
            refused.append(error)
        return "saved"

    async with a2a_door(handler) as door:
        saved = await door.post("/v1", json=message_send("save", "m1"))
        context_id = saved.json()["result"]["task"]["contextId"]
        await door.post("/v1", json=message_send("read", "m2", context_id=context_id))

    assert len(refused) == 1
    assert loaded == ["", "small"]


@dataclass
class ServiceApp:
    service: object = field(default_factory=object)
    seen: list[object] = field(default_factory=list)


class Reference(BaseModel):
    reference: str


async def test_a_resumed_session_gives_callbacks_the_new_requests_context() -> None:
    """A `TextSession` resumed from saved state calls the waiting child with the context of the
    request that resumed it, not the one that started it."""
    child = Agent(
        name="booking",
        instructions="Book.",
        output_type=Reference,
        llm=FakeLLM(["Which date?", tool_call("final_result", reference="R2")]),
    )

    @child.instructions
    def use_service(ctx: RunContext[ServiceApp]) -> str:
        ctx.context.seen.append(ctx.context.service)
        return "Use the supplied service."

    parent = Agent(
        name="desk",
        instructions="Help.",
        llm=FakeLLM([tool_call("book"), "Booked."]),
        tools=[child.as_tool(tool_name="book")],
    )
    first_app, second_app = ServiceApp(), ServiceApp()

    first = await TextSession(agent=parent).run("Book a room", context=first_app)
    second = await TextSession(agent=parent).run("Friday", state=first.state, context=second_app)

    assert first.result.status == "input_required"
    assert second.result.status == "completed" and second.result.output == "Booked."
    assert first_app.seen == [first_app.service]
    assert second_app.seen == [second_app.service]


# -- spans the application opens in callbacks


async def test_concurrent_runs_nest_callback_spans_under_their_own_callers_trace() -> None:
    """Spans a callback opens with the OpenTelemetry SDK continue the trace of the run's
    caller, per run, and the caller's current span is unchanged after the run."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("customer")
    model = FakeLLM({"begin": tool_call("lookup"), "found": "Done"})
    agent = Agent[str, str](instructions="Help.", llm=model)

    @agent.tool
    async def lookup(ctx: RunContext[str]) -> str:
        """Look up the customer's record."""
        with tracer.start_as_current_span("tool", attributes={"job": ctx.context}):
            await asyncio.sleep(0)
        return "found"

    async def run(job: str, caller: trace.Span) -> bool:
        with trace.use_span(caller):
            await agent.run("begin", context=job)
            return trace.get_current_span() is caller

    restored = await asyncio.gather(run("one", _span(101, 11)), run("two", _span(202, 22)))
    provider.shutdown()

    assert all(restored)
    trace_of = {
        span.attributes["job"]: span.context.trace_id
        for span in exporter.get_finished_spans()
        if span.attributes
    }
    assert trace_of == {"one": 101, "two": 202}


async def test_an_a2a_turn_continues_the_callers_trace(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    """A handler runs in the trace its request's `traceparent` names, so a slow brain's turn
    joins its fast brain's trace."""
    seen: list[int] = []

    async def handler(ctx: A2ASessionContext) -> str:
        seen.append(trace.get_current_span().get_span_context().trace_id)
        return "ok"

    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    async with a2a_door(handler) as door:
        await door.post("/v1", json=message_send("hello"), headers={"traceparent": traceparent})

    assert seen == [0x0AF7651916CD43DD8448EB211C80319C]


async def test_concurrent_jobs_continue_the_span_current_at_their_launch(run_job: RunJob) -> None:
    """Each job's entrypoint runs under a job root of its own, in the trace that was current
    when that job was launched."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("customer")

    async def entrypoint(ctx: JobContext) -> None:
        with tracer.start_as_current_span("customer", attributes={"job": ctx.job.id}):
            await asyncio.sleep(0)
        ctx.shutdown("done")

    async def launch(job_id: str, caller: trace.Span) -> None:
        with trace.use_span(caller):
            await run_job(entrypoint, agent_proto.Job(id=job_id))

    await asyncio.gather(launch("one", _span(101, 11)), launch("two", _span(202, 22)))
    provider.shutdown()

    spans = {s.attributes["job"]: s for s in exporter.get_finished_spans() if s.attributes}
    assert {job: span.context.trace_id for job, span in spans.items()} == {"one": 101, "two": 202}
    parents = {job: span.parent.span_id for job, span in spans.items() if span.parent}
    assert parents["one"] not in (11, parents["two"])


# -- application spans as the native exporter sends them to the application's endpoint


class OtlpAttribute(TypedDict):
    key: str
    value: dict[str, str]


class OtlpSpan(TypedDict, total=False):
    name: str
    traceId: str
    spanId: str
    parentSpanId: str
    attributes: list[OtlpAttribute]


class Collector:
    """An OTLP/HTTP endpoint keeping every span it receives, by hex span id."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self.tracer = tracer
        """The application's tracer, on the framework's provider."""
        self.endpoint = ""
        self.spans: dict[str, OtlpSpan] = {}

    async def traces(self, request: web.Request) -> web.Response:
        body = await request.read()
        if request.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        export = MessageToDict(ExportTraceServiceRequest.FromString(body))
        for resource in export.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    for key in ("spanId", "parentSpanId"):
                        if key in span:
                            span[key] = base64.b64decode(span[key]).hex()
                    self.spans[span["spanId"]] = span
        return web.json_response({})

    @staticmethod
    async def metrics(request: web.Request) -> web.Response:
        return web.json_response({})

    def parents_of(self, name: str) -> set[str]:
        """The names of the parents of every span named ``name``."""
        return {
            self.spans[span["parentSpanId"]]["name"]
            for span in self.spans.values()
            if span.get("name") == name and span.get("parentSpanId") in self.spans
        }

    async def exported(self, *names: str) -> None:
        """Flushes every span recorded so far to this endpoint; they include ``names``."""
        await telemetry._shutdown()
        missing = [n for n in names if not any(s.get("name") == n for s in self.spans.values())]
        assert not missing, f"spans never exported: {missing}"


@pytest.fixture
async def collector() -> AsyncIterator[Collector]:
    """The application's OTLP endpoint, with telemetry installed as a worker installs it."""
    app = web.Application()
    app.router.add_post(
        "/v1/traces", (sink := Collector(telemetry._TracerProvider().get_tracer("app"))).traces
    )
    app.router.add_post("/v1/metrics", Collector.metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    sink.endpoint = f"http://127.0.0.1:{runner.addresses[0][1]}"
    telemetry.configure(endpoint=sink.endpoint)
    telemetry._install()
    yield sink
    telemetry.configure()
    await telemetry._shutdown()
    await runner.cleanup()


@pytest.fixture
async def exported_call(collector: Collector) -> Collector:
    """One spoken call through every callback, each opening an `app.<callback>` span."""
    tracer = collector.tracer
    session = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT(["look it up"]),
        llm=FakeLLM([tool_call("lookup"), "Here it is."]),
        tts=FakeTTS(),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    audio = FakeAudioInput()
    session.input.audio = audio
    session.output.audio = FakeAudioOutput()
    agent = Agent(instructions="Help the caller.")
    replied = asyncio.Event()

    @agent.on_enter
    async def entered(session: VoiceSession) -> None:
        with tracer.start_as_current_span("app.on_enter"):
            await asyncio.sleep(0)

    @agent.on_exit
    async def exited(session: VoiceSession) -> None:
        with tracer.start_as_current_span("app.on_exit"):
            await asyncio.sleep(0)

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the caller's record."""
        with tracer.start_as_current_span("app.tool"):
            await asyncio.sleep(0)
        return "found"

    @session.on_user_turn_completed
    async def turn_done(ctx: RunContext, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        with tracer.start_as_current_span("app.on_user_turn_completed"):
            await asyncio.sleep(0)

    @session.llm_node
    async def think(
        ctx: RunContext, chat_ctx: ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        # a span held across the generator's yields: every pull runs in one context
        with tracer.start_as_current_span("app.llm_node"):
            first = True
            async for chunk in session.default.llm_node(ctx, chat_ctx, tools, model_settings):
                yield chunk
                if first:
                    first = False
                    with tracer.start_as_current_span("app.llm_node.after_yield"):
                        pass

    @session.tts_node
    async def speak(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        with tracer.start_as_current_span("app.tts_node"):
            await asyncio.sleep(0)
        async for frame in session.default.tts_node(ctx, text, model_settings):
            yield frame

    @session.on("conversation_item_added")
    def on_item(event: ConversationItemAddedEvent) -> None:
        if event.item.type == "message" and event.item.text_content == "Here it is.":
            replied.set()

    async with session:
        await session.start(agent)
        audio.speak(0.4)
        await replied.wait()
    await collector.exported("app.on_exit", "app.tts_node", "user_turn", "agent_session")
    return collector


@pytest.mark.parametrize(
    ("span", "parent"),
    [
        ("app.on_enter", "on_enter"),
        ("app.on_user_turn_completed", "on_user_turn_completed"),
        ("app.llm_node", "llm_node"),
        ("app.llm_node.after_yield", "app.llm_node"),
        ("app.tool", "function_tool"),
        ("app.tts_node", "tts_node"),
        ("app.on_exit", "on_exit"),
    ],
)
async def test_a_callbacks_spans_nest_under_its_framework_span(
    exported_call: Collector, span: str, parent: str
) -> None:
    """A span a callback opens is exported as a child of the framework span of that callback."""
    assert exported_call.parents_of(span) == {parent}


async def test_stt_telemetry_is_exported_under_the_session(exported_call: Collector) -> None:
    """The user turn carrying the STT's events and model is exported inside the session's
    trace. https://github.com/livekit/agents/issues/3322"""
    turns = [s for s in exported_call.spans.values() if s.get("name") == "user_turn"]
    attributes = [{a["key"]: a["value"] for a in turn.get("attributes", [])} for turn in turns]
    events = [
        json.loads(a["lk.stt.events"]["stringValue"]) for a in attributes if "lk.stt.events" in a
    ]

    assert any(event["type"] == "final_transcript" for batch in events for event in batch)
    for turn in turns:
        ancestor = turn
        while ancestor.get("name") != "agent_session":
            assert ancestor.get("parentSpanId") in exported_call.spans, "no exported session"
            ancestor = exported_call.spans[ancestor["parentSpanId"]]


async def test_a_reply_asked_for_right_after_start_is_exported_in_the_sessions_trace(
    collector: Collector,
) -> None:
    """A greeting requested before any user turn belongs to the session's trace, not its own.

    https://github.com/livekit/agents/issues/3770
    """
    session = VoiceSession(llm=FakeLLM(["Welcome."]), tts=FakeTTS())
    session.output.audio = FakeAudioOutput()

    async with session:
        await session.start(Agent(instructions="Greet the caller."))
        await session.generate_reply(instructions="greet the caller")
    await collector.exported("agent_turn", "agent_session")

    traces = {s["name"]: s["traceId"] for s in collector.spans.values()}
    assert traces["agent_turn"] == traces["agent_session"]


async def test_a_jobs_callback_spans_nest_under_their_job_spans(
    collector: Collector, run_job: RunJob
) -> None:
    """Spans the entrypoint and `on_session_end` open are children of their job spans.

    https://github.com/livekit/agents/pull/3092 https://github.com/livekit/agents/issues/4416
    """

    async def entrypoint(ctx: JobContext) -> None:
        with collector.tracer.start_as_current_span("app.entrypoint"):
            await asyncio.sleep(0)
        ctx.shutdown("done")

    async def session_end(ctx: JobContext) -> None:
        with collector.tracer.start_as_current_span("app.on_session_end"):
            await asyncio.sleep(0)

    await run_job(entrypoint, agent_proto.Job(id="AJ_parent"), session_end=session_end)
    await collector.exported("app.entrypoint", "app.on_session_end")

    assert collector.parents_of("app.entrypoint") == {"job_entrypoint"}
    assert collector.parents_of("app.on_session_end") == {"on_session_end"}


async def test_framework_shutdown_leaves_the_applications_tracer_provider_open(
    collector: Collector, run_job: RunJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The framework exports through an application's own SDK provider without closing it,
    and adds the configured metadata to each job's spans.
    https://github.com/livekit/agents/pull/7428"""
    closed: list[bool] = []

    class ApplicationProvider(TracerProvider):
        def shutdown(self) -> None:
            closed.append(True)
            super().shutdown()

    provider = ApplicationProvider(shutdown_on_exit=False)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(telemetry, "_provider_adopted", False)
    telemetry.configure(endpoint=collector.endpoint, metadata={"customer.label": "kept"})
    telemetry._install()

    async def entrypoint(ctx: JobContext) -> None:
        with provider.get_tracer("customer").start_as_current_span("customer.job"):
            pass
        ctx.shutdown("done")

    await run_job(entrypoint, agent_proto.Job(id="AJ_first"))
    await run_job(entrypoint, agent_proto.Job(id="AJ_second"))
    await collector.exported("customer.job")
    provider.shutdown()

    assert closed == [True], "the framework closed the application's provider"
    jobs = [s for s in collector.spans.values() if s.get("name") == "customer.job"]
    labels = [{a["key"]: a["value"] for a in span.get("attributes", [])} for span in jobs]
    assert len(jobs) == 2
    assert all(label["customer.label"]["stringValue"] == "kept" for label in labels)
