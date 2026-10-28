"""The Agents 2.0 agent surface: name inference, registration decorators, transitions and runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import (
    Agent,
    AgentHandler,
    AgentServer,
    InputRequired,
    ModelRetry,
    RunContext,
    RunResult,
    VoiceSession,
    function_tool,
    mock_tools,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext, FunctionToolCall, Tool, Toolset
from livekit.agents.storage import Storage
from livekit.agents.storage.resolve import Thread
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMResponse
from .storage_fixture import new_session
from .test_a2a_server import AGENT, API_KEY, API_SECRET, CREDENTIALS, http_client, wire
from .test_doors import stream

pytestmark = [pytest.mark.unit]


module_agent = Agent(instructions="named after the variable")


class RecordingLLM(FakeLLM):
    """A fake LLM that keeps every chat context it was asked to complete."""

    def __init__(self, *, fake_responses: list[FakeLLMResponse] | None = None) -> None:
        super().__init__(fake_responses=fake_responses)
        self.requests: list[ChatContext] = []

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ):  # type: ignore[override]
        self.requests.append(chat_ctx.copy())
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)


def test_agent_name_inference() -> None:
    assert module_agent.name == "module_agent"
    assert Agent(instructions="x", name="named").name == "named"
    assert Agent(instructions="unassigned").name == "default_agent"


def test_agent_tool_decorator() -> None:
    agent = Agent(instructions="x")

    @agent.tool
    async def lookup(ctx: RunContext, key: str) -> str:
        """Look a key up.

        Args:
            key: the key
        """
        return key

    @agent.tool(name="renamed")
    async def other(ctx: RunContext) -> str:
        """Another one."""
        return "x"

    names = sorted(t.info.name for t in agent.tools)  # type: ignore[union-attr]
    assert names == ["lookup", "renamed"]

    class WithMethod(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="x")

        @function_tool
        async def on_method(self, ctx: RunContext) -> str:
            """A method tool."""
            return "m"

    assert [t.info.name for t in WithMethod().tools] == ["on_method"]  # type: ignore[union-attr]


async def test_agent_instructions_and_on_enter() -> None:
    llm = RecordingLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi there", ttft=0, duration=0)]
    )
    agent = Agent(instructions="static part")
    entered: list[VoiceSession] = []

    @agent.instructions
    def dynamic(ctx: RunContext) -> str:
        return "dynamic part"

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        entered.append(session)

    async with VoiceSession(llm=llm) as session:
        await session.start(agent)
        assert entered == [session]
        assert agent._frozen

        with pytest.raises(RuntimeError):

            @agent.tool
            async def late(ctx: RunContext) -> str:
                """Too late."""
                return "x"

        result = await session.run(user_input="hello")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    system = [
        item for item in llm.requests[-1].items if item.type == "message" and item.role == "system"
    ]
    assert any(
        "static part" in (m.text_content or "") and "dynamic part" in (m.text_content or "")
        for m in system
    )


async def test_on_exit_receives_the_session_on_replacement_and_close() -> None:
    first = Agent(instructions="first", name="first")
    second = Agent(instructions="second", name="second")
    exited: list[tuple[str, VoiceSession]] = []
    replaced = asyncio.Event()

    @first.on_exit
    async def leave_first(session: VoiceSession) -> None:
        exited.append(("first", session))
        replaced.set()

    @second.on_exit
    async def leave_second(session: VoiceSession) -> None:
        exited.append(("second", session))

    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first)
        session.update_agent(second)
        await asyncio.wait_for(replaced.wait(), timeout=5)
        assert exited == [("first", session)]
    assert exited == [("first", session), ("second", session)]


async def test_agent_chat_ctx_is_a_read_only_view_inside_its_hooks() -> None:
    agent = Agent(instructions="x")
    refused: list[bool] = []

    @agent.on_enter
    async def enter(session: VoiceSession) -> None:
        with pytest.raises(RuntimeError):
            agent.chat_ctx.add_message(role="user", content="injected")
        refused.append(True)

    with pytest.raises(RuntimeError):
        agent.chat_ctx  # noqa: B018 (outside its hooks and tools)
    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(agent)
        assert refused == [True]
        assert not [i for i in session.history.items if i.type == "message"]


async def test_mock_tools_replace_a_tool_only_in_the_runs_started_inside_it() -> None:
    """``mock_tools`` selects mocks for the invocations started in its block; a concurrent run
    outside it calls the real tool."""
    calls: list[str] = []
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="weather?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="lookup", arguments='{"city": "Paris"}', call_id="l1")
                ],
            ),
            FakeLLMResponse(input="real: Paris", content="real", ttft=0, duration=0),
            FakeLLMResponse(input="mock: Paris", content="mocked", ttft=0, duration=0),
        ]
    )
    agent = Agent(instructions="x", llm=llm)

    @agent.tool
    async def lookup(ctx: RunContext, city: str) -> str:
        """Look the weather up.

        Args:
            city: the city
        """
        calls.append(city)
        return f"real: {city}"

    async def mocked() -> str:
        result = await agent.run("weather?")
        return result.output if result.status == "completed" else ""

    with mock_tools(agent, {"lookup": lambda city: f"mock: {city}"}):
        mocked_run = asyncio.create_task(mocked())
    real = await agent.run("weather?")

    assert await mocked_run == "mocked"
    assert real.status == "completed" and real.output == "real"
    assert calls == ["Paris"]


async def test_session_mock_tools_replace_a_tool_on_the_call() -> None:
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="weather?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="lookup", arguments='{"city": "Paris"}', call_id="l1")
                ],
            ),
            FakeLLMResponse(input="mock: Paris", content="mocked", ttft=0, duration=0),
        ]
    )
    agent = Agent(instructions="x")

    @agent.tool
    async def lookup(ctx: RunContext, city: str) -> str:
        """Look the weather up.

        Args:
            city: the city
        """
        raise AssertionError("the real tool ran")

    async with VoiceSession(llm=llm) as session:
        mock_tools(agent, {"lookup": lambda city: f"mock: {city}"}, session=session)
        await session.start(agent)
        result = await session.run(user_input="weather?")
        result.expect.next_event().is_function_call(name="lookup", arguments={"city": "Paris"})
        result.expect.next_event().is_function_call_output(output="mock: Paris")


async def test_input_required_pauses_the_turn_until_the_next_message() -> None:
    """A verbatim ``InputRequired`` speaks the question as written; the next user message is
    the call's output."""
    fnol = Agent(instructions="claims")

    @fnol.tool
    async def request_photo(ctx: RunContext) -> InputRequired:
        """Ask for a photo."""
        return InputRequired("Could you send a photo of the damage?", verbatim=True)

    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="my car is damaged",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="request_photo", arguments="{}", call_id="p1")],
            ),
            FakeLLMResponse(input="here it is", content="got it, thanks", ttft=0, duration=0),
        ]
    )
    async with VoiceSession(llm=llm) as session:
        await session.start(fnol)
        result = await session.run(user_input="my car is damaged")
        result.expect.next_event().is_function_call(name="request_photo")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")  # the question, word for word
        result.expect.no_more_events()  # no tool reply: the turn waits

        result = await session.run(user_input="here it is")
        result.expect.next_event().is_message(role="assistant")

    history = session.history.items
    question = [i for i in history if i.type == "message" and i.role == "assistant"][0]
    assert question.text_content == "Could you send a photo of the damage?"
    output = [i for i in history if i.type == "function_call_output" and i.call_id == "p1"][0]
    assert output.output == "here it is"
    # the model ran on the call, its answer and the question
    assert [i.call_id for i in llm.requests[-1].items if i.type == "function_call_output"] == ["p1"]


@dataclass
class Deps:
    greeting: str


@dataclass
class Patient:
    looked_up: int = 0


@dataclass
class AppContext:
    settings: Deps
    patient: Patient = persisted(default_factory=Patient)


@dataclass
class Answer:
    city: str
    temperature: int


async def test_agent_run_returns_a_typed_result() -> None:
    """``agent.run()`` loops over the model and the tools, with deps and state on the context."""
    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="weather in tokyo?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="lookup", arguments='{"city": "Tokyo"}', call_id="c1")
                ],
            ),
            FakeLLMResponse(
                input="Tokyo: 21",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(
                        name="final_result",
                        arguments='{"city": "Tokyo", "temperature": 12}',
                        call_id="c2",
                    )
                ],
            ),
            FakeLLMResponse(
                input="invalid final_result: the temperature must be at least 20",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(
                        name="final_result",
                        arguments='{"city": "Tokyo", "temperature": 21}',
                        call_id="c3",
                    )
                ],
            ),
        ]
    )
    weather = Agent[AppContext, Answer](instructions="weather", output_type=Answer, llm=llm)

    @weather.instructions
    def known(ctx: RunContext[AppContext]) -> str:
        return f"Greeting: {ctx.context.settings.greeting}"

    @weather.tool
    async def lookup(ctx: RunContext[AppContext], city: str) -> str:
        """Look the weather up.

        Args:
            city: the city
        """
        ctx.context.patient.looked_up += 1
        assert ctx.task_id is None and ctx.live_session is None  # no A2A turn, no call
        return f"{city}: 21"

    retries: list[str] = []

    @weather.output_validator
    def warm_enough(ctx: RunContext[AppContext], output: Answer) -> Answer:
        if output.temperature < 20:
            retries.append("too cold")
            raise ModelRetry("the temperature must be at least 20")
        return output

    state = Patient()
    result = await weather.run(
        "weather in tokyo?",
        context=AppContext(settings=Deps(greeting="hello"), patient=state),
    )
    assert result.status == "completed"
    assert result.output == Answer(city="Tokyo", temperature=21)
    assert state.looked_up == 1
    assert retries == ["too cold"]
    assert result.usage.requests == 3 and result.usage.tool_calls == 3

    new = result.new_messages()
    assert new[0].type == "message" and new[0].role == "user"
    assert [i.type for i in new[1:3]] == ["function_call", "function_call_output"]
    assert all(i.type != "message" or i.role != "system" for i in result.all_messages())
    system = [i for i in llm.requests[0].items if i.type == "message" and i.role == "system"]
    assert "Greeting: hello" in (system[0].text_content or "")

    # the next run continues from all_messages()
    llm.fake_response_map["and osaka?"] = FakeLLMResponse(
        input="and osaka?",
        content="",
        ttft=0,
        duration=0,
        tool_calls=[
            FunctionToolCall(
                name="final_result", arguments='{"city": "Osaka", "temperature": 23}', call_id="c4"
            )
        ],
    )
    second = await weather.run(
        "and osaka?",
        message_history=result.all_messages(),
        context=AppContext(settings=Deps(greeting="hi"), patient=state),
    )
    assert second.status == "completed" and second.output == Answer(city="Osaka", temperature=23)
    assert len(second.all_messages()) > len(second.new_messages())


async def test_agent_run_stops_at_input_required() -> None:
    fnol = Agent(
        instructions="claims",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="my car is damaged",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="request_photo", arguments="{}", call_id="p1")
                    ],
                ),
                FakeLLMResponse(input="here it is", content="thanks", ttft=0, duration=0),
            ]
        ),
    )

    @fnol.tool
    async def request_photo(ctx: RunContext) -> InputRequired:
        """Ask for a photo."""
        return InputRequired("Could you send a photo?")

    result = await fnol.run("my car is damaged")
    assert result.status == "input_required"
    assert result.question == "Could you send a photo?"
    assert result.new_messages()[-1].type == "message"
    assert result.new_messages()[-1].text_content == "Could you send a photo?"

    second = await fnol.run("here it is", message_history=result.all_messages())
    assert second.status == "completed" and second.output == "thanks"


async def test_live_session_context_reaches_tools_and_is_explicitly_saved(
    thread_storage: Storage,
) -> None:
    seen: list[tuple[Deps, Patient, str]] = []
    agent = Agent[AppContext](instructions="x")

    @agent.tool
    async def note(ctx: RunContext[AppContext]) -> str:
        """Note the state."""
        ctx.context.patient.looked_up += 1
        seen.append((ctx.context.settings, ctx.context.patient, ctx.thread.id))
        await ctx.live_session.store.save_context(ctx.context)
        return "noted"

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="note", arguments="{}", call_id="n1")],
            ),
            FakeLLMResponse(input="noted", content="ok", ttft=0, duration=0),
        ]
    )
    state = Patient()
    deps = Deps(greeting="hey")
    storage = thread_storage
    sid = await new_session(storage, agent.name)
    thread = Thread(storage, sid, agent_name=agent.name)
    app = AppContext(settings=deps, patient=state)
    async with VoiceSession(llm=llm, context=app, thread=thread) as session:
        assert session.context is app
        await session.start(agent)
        await session.store.load_context(app)
        result = await session.run(user_input="go")
        result.expect.contains_function_call(name="note")
        assert result.output == "ok"

    assert seen[0][0] is deps and seen[0][1] is state and seen[0][2] == sid
    assert state.looked_up == 1
    # the call's turn was committed to the agent's lane: history and state
    history = await thread.chat_history()
    items = [i for i in history.items if i.type != "agent_handoff"]
    assert [i.type for i in items] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    db = await storage.open(sid)
    row = await db.lane_info(agent.name)
    db.close()
    assert row is not None
    from livekit.agents.storage.context_store import ContextStore

    restored = AppContext(settings=deps)
    activation = await storage.threads._core_context_store(sid, agent.name)
    await ContextStore(scope_id=f"root:{agent.name}", activation=activation).load_context(restored)
    assert restored.patient == Patient(looked_up=1)


async def test_run_stream_yields_the_reply_as_it_is_written() -> None:
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there, friend", ttft=0, duration=0)
        ]
    )
    agent = Agent(instructions="say hi", llm=llm)

    async with agent.run_stream("hello") as stream:
        deltas = [delta async for delta in stream.stream_text(delta=True)]
        result = await stream.result()

    assert len(deltas) > 1, "the reply arrives in pieces"
    assert "".join(deltas) == "hi there, friend"
    assert result.status == "completed" and result.output == "hi there, friend"


class CountingToolset(Toolset):
    def __init__(self) -> None:
        super().__init__(id="counting")
        self.setups = self.closes = self.calls = 0
        self.release = asyncio.Event()

    async def setup(self) -> CountingToolset:
        self.setups += 1
        return await super().setup()

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()

    @function_tool
    async def wait(self, ctx: RunContext) -> str:
        """Wait until released."""
        self.calls += 1
        await self.release.wait()
        return "done"


async def test_a_run_sets_up_and_closes_its_toolsets_once() -> None:
    """A toolset connects per run (tools.md "Toolsets"): concurrent runs share one setup and
    one close, and a run left early still closes it."""
    toolset = CountingToolset()
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="wait", arguments="{}", call_id="w1")],
            ),
            FakeLLMResponse(input="done", content="finished", ttft=0, duration=0),
        ]
    )
    agent = Agent(instructions="x", llm=llm, tools=[toolset])

    runs = [asyncio.create_task(agent.run("go")) for _ in range(2)]
    while toolset.calls < 2:
        await asyncio.sleep(0.01)
    toolset.release.set()
    for result in await asyncio.gather(*runs):
        assert result.status == "completed" and result.output == "finished"
    assert (toolset.setups, toolset.closes) == (1, 1)

    toolset.release.clear()
    async with agent.run_stream("go") as stream:
        async for event in stream.stream_events():
            if event.type == "tool_call":
                break
    assert (toolset.setups, toolset.closes) == (2, 2)


async def test_a_tool_in_an_a2a_turn_reads_its_task_and_thread(thread_storage: Storage) -> None:
    """``ctx.task_id`` is the turn's A2A task and ``ctx.thread`` its thread (agents.md
    "RunContext"); a text turn has no live session."""
    seen: list[tuple[str | None, str, VoiceSession | None]] = []
    agent = Agent(
        instructions="x",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="hi",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="note", arguments="{}", call_id="n1")],
                ),
                FakeLLMResponse(input="noted", content="ok", ttft=0, duration=0),
            ]
        ),
    )

    @agent.tool
    async def note(ctx: RunContext) -> str:
        """Note the turn."""
        seen.append((ctx.task_id, ctx.thread.id, ctx.live_session))
        return "noted"

    handler = AgentHandler()

    @handler.a2a_session(endpoint="frontend", description="the front desk")
    async def text(ctx: A2ASessionContext) -> RunResult[str]:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    server = AgentServer(
        storage=thread_storage, agent_name=AGENT, api_key=API_KEY, api_secret=API_SECRET
    )
    server.add(handler)
    async with http_client(server, Authorization=f"Bearer {CREDENTIALS.project_token()}") as http:
        events = await stream(http, "/frontend/v1", wire("hi", "m1"))

    task = events[0]["task"]
    assert seen == [(task["id"], task["contextId"], None)]
