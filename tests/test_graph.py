"""``livekit.agents.graph``: nodes, edges from annotations, run and iter, the diagram, and the
waiting form: a graph awaited from a tool asks the person and runs the asking node again on the
reply."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, Completed, Failed, InputRequired, RunContext, RunResult
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.graph import (
    STALE_GRAPH,
    BaseNode,
    Edge,
    End,
    Graph,
    GraphRunContext,
    GraphRuntimeError,
    GraphSetupError,
)
from livekit.agents.llm import ChatMessage, FunctionToolCall
from livekit.agents.storage import AgentDBStorage

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import (
    http_client,
    make_server,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    payloads,
    send,
    stock_client,
    user_message,
)

pytestmark = pytest.mark.unit


@dataclass
class Draft:
    limit: int
    drafts: list[str] = field(default_factory=list)


@dataclass
class Write(BaseNode[Draft]):
    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[Draft]) -> Review:
        ctx.state.drafts.append(f"draft {len(ctx.state.drafts) + 1}: {self.feedback or 'fresh'}")
        return Review(ctx.state.drafts[-1])


@dataclass
class Review(BaseNode[Draft, None, str]):
    text: str

    async def run(
        self, ctx: GraphRunContext[Draft]
    ) -> Annotated[Write, Edge(label="again")] | End[str]:
        if len(ctx.state.drafts) >= ctx.state.limit:
            return End(self.text)
        return Write(feedback="shorter")


class Renewal(Graph[str, Draft]):
    nodes = [Write, Review]


async def test_run_loops_until_end() -> None:
    result = await Renewal.run(Write(), state=Draft(limit=3))
    assert result.output == "draft 3: shorter"
    assert result.state.drafts == ["draft 1: fresh", "draft 2: shorter", "draft 3: shorter"]


async def test_iter_yields_every_node_and_the_end() -> None:
    seen: list[str] = []
    async with Renewal.iter(Write(), state=Draft(limit=2)) as run:
        assert run.result is None
        async for node in run:
            seen.append(type(node).__name__)
    assert seen == ["Write", "Review", "Write", "Review", "End"]
    assert run.result is not None and run.result.output == "draft 2: shorter"


async def test_next_runs_one_node_by_hand() -> None:
    async with Renewal.iter(Write(), state=Draft(limit=1)) as run:
        out = await run.next()
        assert isinstance(out, Review)
        assert isinstance(await run.next(), End)
        with pytest.raises(GraphRuntimeError):
            await run.next()


def test_edges_from_the_annotations() -> None:
    assert set(Renewal.node_defs) == {"Write", "Review"}
    assert list(Renewal.node_defs["Write"].next_node_edges) == ["Review"]
    assert Renewal.node_defs["Review"].returns_end
    assert Renewal.node_defs["Review"].next_node_edges["Write"].label == "again"


def test_mermaid() -> None:
    assert Renewal.mermaid_code(start_node=Write) == "\n".join(
        [
            "stateDiagram-v2",
            "  [*] --> Write",
            "  Write --> Review",
            "  Review --> Write: again",
            "  Review --> [*]",
        ]
    )


def test_an_edge_to_a_node_outside_the_graph_is_a_setup_error() -> None:
    with pytest.raises(GraphSetupError, match="Review"):

        class Half(Graph[str, Draft]):
            nodes = [Write]


def test_a_return_that_is_not_a_node_is_a_setup_error() -> None:
    @dataclass
    class Bad(BaseNode[None]):
        async def run(self, ctx: GraphRunContext[None]) -> str:  # type: ignore[override]
            return "no"

    with pytest.raises(GraphSetupError, match="neither a node"):

        class Broken(Graph[None, None]):
            nodes = [Bad]


async def test_returning_a_node_outside_the_graph_fails_the_run() -> None:
    @dataclass
    class Outside(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext[None]) -> End[int]:
            return End(1)

    @dataclass
    class Start(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext[None]) -> End[int]:  # lies about its edge
            return Outside()  # type: ignore[return-value]

    class Lying(Graph[int, None]):
        nodes = [Start]

    with pytest.raises(GraphRuntimeError, match="Outside"):
        await Lying.run(Start(), state=None)


async def test_deps_reach_the_nodes() -> None:
    @dataclass
    class Add(BaseNode[int, int, int]):
        async def run(self, ctx: GraphRunContext[int, int]) -> End[int]:
            return End(ctx.state + ctx.deps)

    class Sum(Graph[int, int, int]):
        nodes = [Add]

    result = await Sum.run(Add(), state=2, deps=40)
    assert result.output == 42


@dataclass
class Form:
    name: str | None = None
    asked: int = 0


@dataclass
class AskName(BaseNode[Form]):
    greeting: str

    async def run(self, ctx: GraphRunContext[Form]) -> Greet | InputRequired:
        if ctx.reply:
            ctx.state.name = ctx.reply.strip()
        if not ctx.state.name:
            ctx.state.asked += 1
            return InputRequired("What is your name?")
        return Greet()


@dataclass
class Greet(BaseNode[Form, None, str]):
    async def run(self, ctx: GraphRunContext[Form]) -> End[str]:
        return End(f"hello {ctx.state.name}, asked {ctx.state.asked} time(s)")


class Welcome(Graph[str, Form]):
    nodes = [AskName, Greet]


async def test_inline_run_of_a_node_that_waits_is_an_error() -> None:
    with pytest.raises(GraphRuntimeError, match="await Welcome"):
        await Welcome.run(AskName("hi"), state=Form())


greeter = Agent(
    instructions="front desk",
    llm=FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hi there",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="welcome", arguments="{}", call_id="c1")],
            ),
            FakeLLMResponse(
                input="hello Ada, asked 1 time(s)", content="Welcome, Ada.", ttft=0, duration=0
            ),
        ]
    ),
)


@greeter.tool
async def welcome(ctx: RunContext) -> str:
    """Welcome the person."""
    return await Welcome(AskName("hi"), state=Form())


async def desk(ctx: A2ASessionContext[Any]) -> RunResult[Any]:
    return await greeter.run(ctx.message, message_history=ctx.message_history)


async def test_an_awaited_graph_asks_and_runs_the_asking_node_again_on_the_reply(
    agentdb_storage: AgentDBStorage,
) -> None:
    server, _ = make_server(agentdb_storage, desk)
    async with http_client(server) as http:
        client = await stock_client(http)
        first = payloads(await send(client, user_message("hi there", "m1")))
        ctx_id = first[0].context_id
        asked = [p for p in first if hasattr(p, "status")][-1].status
        assert asked.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
        assert asked.message.parts[0].text == "What is your name?"

        second = payloads(await send(client, user_message("Ada", "m2", context_id=ctx_id)))
        done = [p for p in second if hasattr(p, "status")][-1].status
        await client.close()

    assert done.state == pb.TaskState.TASK_STATE_COMPLETED
    answer = "".join(
        part.text for p in second if hasattr(p, "artifact") for part in p.artifact.parts
    )
    assert answer == "Welcome, Ada."  # streamed in pieces as the model writes it


async def test_a_saved_node_that_no_longer_rebuilds_fails_the_graph() -> None:
    saved = Welcome(AskName("hi"), state=Form())
    assert saved._arguments() == {"node": "AskName", "fields": {"greeting": "hi"}}

    for args in ({"node": "Gone", "fields": {}}, {"node": "AskName", "fields": {"hello": "hi"}}):
        graph = Welcome._rebuild(args)
        graph._restore({"name": None, "asked": 1}, call_id="c1")
        out = await graph._take(ChatMessage(role="user", content=["Ada"]))
        assert isinstance(out, Failed) and out.error.startswith(STALE_GRAPH)

    graph = Welcome._rebuild(saved._arguments())
    graph._restore({"name": None, "asked": 1}, call_id="c1")
    assert await graph._take(ChatMessage(role="user", content=["Ada"])) == Completed(
        "hello Ada, asked 1 time(s)"
    )
