"""The Agents 2.0 agent surface: name inference, registration decorators, transitions and runs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from livekit.agents import (
    Agent,
    AgentTask,
    Completed,
    Failed,
    InputRequired,
    LiveSession,
    ModelRetry,
    RunContext,
    function_tool,
)
from livekit.agents.llm import ChatContext, FunctionToolCall, Tool
from livekit.agents.storage import AgentDBStorage
from livekit.agents.storage.protocol import load_state
from livekit.agents.storage.resolve import Thread
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .agentdb_fixture import new_session
from .fake_llm import FakeLLM, FakeLLMResponse

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
    assert module_agent.id == "module_agent"
    assert Agent(instructions="x", name="named").id == "named"

    class Sub(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="x")

    assert Sub().id == "sub"
    assert Agent(instructions="unassigned").id == "default_agent"


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
    entered: list[LiveSession] = []

    @agent.instructions
    def dynamic(ctx: RunContext) -> str:
        return "dynamic part"

    @agent.on_enter
    async def greet(session: LiveSession) -> None:
        entered.append(session)

    async with LiveSession(llm=llm) as session:
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


async def test_completed_ends_the_task_with_its_value() -> None:
    """A task tool returning ``Completed(value)`` ends the task; the awaiting tool gets the value."""

    class Capture(AgentTask[int]):
        def __init__(self) -> None:
            super().__init__(instructions="capture a number")

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="capture_greeting")

        @function_tool
        async def record(self, ctx: RunContext, value: int) -> Completed[int]:
            """Record the number.

            Args:
                value: the number
            """
            return Completed(value)

    class Decline(AgentTask[int]):
        def __init__(self) -> None:
            super().__init__(instructions="declined")

        async def on_enter(self) -> Failed:
            return Failed("the caller declined")

    outcomes: list[object] = []
    root = Agent(instructions="root")

    @root.tool
    async def capture(ctx: RunContext) -> str:
        """Start the capture."""
        outcomes.append(await Capture())
        try:
            await Decline()
        except Exception as e:  # the Failed error surfaces to the awaiting tool
            outcomes.append(e)
        return "captured"

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="capture", arguments="{}", call_id="c1")],
            ),
            FakeLLMResponse(input="capture_greeting", content="number?", ttft=0, duration=0),
            FakeLLMResponse(
                input="seven",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="record", arguments='{"value": 7}', call_id="c2")
                ],
            ),
            FakeLLMResponse(input="captured", content="thanks", ttft=0, duration=0),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(root)
        await session.run(user_input="go")
        assert isinstance(session.current_agent, Capture)
        await session.run(user_input="seven")
        assert session.current_agent is root

    assert outcomes[0] == 7
    assert "the caller declined" in str(outcomes[1])


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
    async with LiveSession(llm=llm) as session:
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
class Answer:
    city: str
    temperature: int


async def test_agent_run_returns_a_typed_result() -> None:
    """``agent.run()`` loops over the model and the tools, with deps and state on the context."""
    weather = Agent(instructions="weather", output_type=Answer, deps_type=Deps, state_type=Patient)

    @weather.instructions
    def known(ctx: RunContext[Deps, Patient]) -> str:
        return f"Greeting: {ctx.deps.greeting}"

    @weather.tool
    async def lookup(ctx: RunContext[Deps, Patient], city: str) -> str:
        """Look the weather up.

        Args:
            city: the city
        """
        ctx.state.looked_up += 1
        assert ctx.task_id.startswith("run_")
        return f"{city}: 21"

    retries: list[str] = []

    @weather.output_validator
    def warm_enough(ctx: RunContext[Deps, Patient], output: Answer) -> Answer:
        if output.temperature < 20:
            retries.append("too cold")
            raise ModelRetry("the temperature must be at least 20")
        return output

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
                input="Validation failed, fix the output and try again: the temperature must be at least 20",
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
    weather.update_options(llm=llm)

    state = Patient()
    result = await weather.run(
        "weather in tokyo?", deps=Deps(greeting="hello"), state=state, max_steps=5
    )
    assert result.output == Answer(city="Tokyo", temperature=21)
    assert result.input_required is None
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
        "and osaka?", message_history=result.all_messages(), deps=Deps(greeting="hi"), state=state
    )
    assert second.output == Answer(city="Osaka", temperature=23)
    assert len(second.all_messages()) > len(second.new_messages())


async def test_agent_run_stops_at_input_required() -> None:
    fnol = Agent(instructions="claims")

    @fnol.tool
    async def request_photo(ctx: RunContext) -> InputRequired:
        """Ask for a photo."""
        return InputRequired("Could you send a photo?")

    fnol.update_options(
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
        )
    )
    result = await fnol.run("my car is damaged")
    assert result.output is None
    assert result.input_required == "Could you send a photo?"
    assert result.new_messages()[-1].type == "message"
    assert result.new_messages()[-1].text_content == "Could you send a photo?"

    second = await fnol.run("here it is", message_history=result.all_messages())
    assert second.output == "thanks"


async def test_live_session_state_and_deps_reach_the_run_context(
    agentdb_storage: AgentDBStorage,
) -> None:
    seen: list[tuple[Deps, Patient, str]] = []
    agent = Agent(instructions="x", deps_type=Deps, state_type=Patient)

    @agent.tool
    async def note(ctx: RunContext[Deps, Patient]) -> str:
        """Note the state."""
        ctx.state.looked_up += 1
        seen.append((ctx.deps, ctx.state, ctx.task_id))
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
    storage = agentdb_storage
    sid = await new_session(storage, agent.id)
    thread = Thread(storage, sid, agent_name=agent.id)
    async with LiveSession(llm=llm, state=state, deps=deps, thread=thread) as session:
        assert session.state is state and session.deps is deps
        await session.start(agent)
        result = await session.run(user_input="go")
        result.expect.contains_function_call(name="note")
        assert result.output == "ok"

    assert seen[0][0] is deps and seen[0][1] is state and seen[0][2]
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
    row = await db.lane(agent.id)
    db.close()
    assert row is not None and load_state(row.state, Patient) == Patient(looked_up=1)


def test_agent_session_alias_is_deprecated() -> None:
    import warnings

    import livekit.agents

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert livekit.agents.AgentSession is LiveSession
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


async def test_stream_yields_the_reply_as_it_is_written() -> None:
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there, friend", ttft=0, duration=0)
        ]
    )
    agent = Agent(instructions="say hi", llm=llm)

    run = agent.stream("hello")
    deltas = [delta async for delta in run]

    assert len(deltas) > 1, "the reply arrives in pieces"
    assert "".join(deltas) == "hi there, friend"
    assert run.result.output == "hi there, friend"
