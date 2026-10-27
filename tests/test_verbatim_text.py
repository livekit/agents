"""Verbatim text over A2A, front desk to expert: a say or a verbatim result during the turn is a
flagged ``working`` status sent at once; the answer is flagged only when the turn ends on one."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ChatContext, ToolResult
from livekit.agents.llm.tool_context import DELIVERED
from livekit.agents.storage import AgentDBStorage

from .a2a_driver import (
    AGENT,
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    sse_results,
)

pytestmark = pytest.mark.unit

NOTICE = "Calls are recorded for quality."
CODE = "Your code is HTL-7A3K."
POLICY = "Cancellations are free until noon."
VERBATIM = "livekit/verbatim"

Frames = list[dict[str, Any]]


def is_verbatim(part: dict[str, Any]) -> bool:
    return metadata_from_wire(part.get("metadata")).get(VERBATIM) is True


def working(frames: Frames) -> list[tuple[str, bool]]:
    """Each ``working`` status's text parts, in order, with their flag."""
    statuses = [f["statusUpdate"]["status"] for f in frames if "statusUpdate" in f]
    return [
        (p["text"], is_verbatim(p))
        for s in statuses
        if s["state"] == "TASK_STATE_WORKING"
        for p in s.get("message", {}).get("parts", [])
        if "text" in p
    ]


def answer(frames: Frames) -> list[tuple[str, bool]]:
    """The answer artifact's text parts, its chunks joined, with their flag."""
    parts = [
        p
        for f in frames
        if "artifactUpdate" in f
        for p in f["artifactUpdate"]["artifact"].get("parts", [])
        if p.get("text")
    ]
    return [(p["text"], is_verbatim(p)) for p in parts]


def serving(agent: Agent) -> Callable[[A2ASessionContext[Any]], Any]:
    async def handler(ctx: A2ASessionContext[Any]) -> object:
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required if result.input_required is not None else result.output

    return handler


class _Recording(httpx.AsyncBaseTransport):
    """The expert's door, keeping the frames of every stream it answered."""

    def __init__(self, app: Any) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self.frames: Frames = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response
        body = httpx.Response(200, headers=response.headers, content=await response.aread())
        self.frames.extend(r["result"] for r in sse_results(body) if "result" in r)
        return body


@pytest.fixture
def delegate_to(door: Callable[..., Door], agentdb_storage: AgentDBStorage) -> Callable[..., Any]:
    """The person's text to a front desk delegating to ``expert``: both wires' frames."""

    async def run(expert: Agent, front_model: ScriptedLLM, text: str) -> tuple[Frames, Frames]:
        async def expert_door(ctx: A2ASessionContext[Any]) -> object:
            assert ctx.caller_lane == AGENT  # the delegation names the front desk's lane
            return await serving(expert)(ctx)

        server = AgentServer(storage=agentdb_storage, agent_name="expert")
        server.a2a_session(endpoint="expert", description="Answers.")(expert_door)
        transport = _Recording(server.http.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://experts.test") as http:
            delegate = A2ADelegate("http://experts.test/expert", httpx_client=http)
            front = Agent(instructions="front desk", tools=[delegate], llm=front_model)
            sent = await door(serving(front)).send(text)
            await delegate.aclose()
        return transport.frames, sent.frames

    return run


def tool_outputs(request: ChatContext) -> list[str]:
    return [i.output for i in request.items if i.type == "function_call_output"]


@pytest.mark.rule("R18", "R22", "R23", "B12")
async def test_verbatim_text_said_during_a_turn_goes_out_at_once_and_the_answer_is_worded(
    delegate_to: Callable[..., Any],
) -> None:
    """A say, a verbatim result asking for a reply and one that would end the turn are flagged
    ``working`` statuses on both wires, and the front desk's model words only the answer."""
    expert_model = ScriptedLLM(
        Reply(tool_calls=[call("book", "e1"), call("cancel_policy", "e2")]),
        Reply("Booked for Tuesday.", when=output_of("e2")),
    )
    expert = Agent(instructions="scheduling", llm=expert_model)

    @expert.tool
    async def book(ctx: RunContext) -> ToolResult:
        """Book."""
        await ctx.update("checking the calendar")
        await ctx.say(NOTICE)  # type: ignore[attr-defined]
        return ToolResult(CODE, verbatim=True)

    @expert.tool
    async def cancel_policy(ctx: RunContext) -> ToolResult:
        """The cancellation policy."""
        return ToolResult(POLICY, verbatim=True, reply_required=False)

    front_model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request="book me")]),
        Reply("You are booked for Tuesday.", when=output_of("c1")),
    )
    expert_frames, desk_frames = await delegate_to(expert, front_model, "book me")

    verbatim = [(NOTICE, True), (CODE, True), (POLICY, True)]
    assert working(expert_frames) == [("checking the calendar", False), *verbatim]
    assert answer(expert_frames) == [("Booked for Tuesday.", False)]
    # the expert's model reads the request alone, nothing of the caller injected
    first, last = expert_model.requests[0], expert_model.requests[-1]
    assert [i.text_content for i in first.items if i.type == "message"] == ["scheduling", "book me"]
    said = [i.text_content for i in last.items if i.type == "message" and i.role == "assistant"]
    assert said == [NOTICE]
    assert tool_outputs(last) == [DELIVERED.format(text=CODE), DELIVERED.format(text=POLICY)]

    assert working(desk_frames) == [("checking the calendar", False), *verbatim]
    assert answer(desk_frames) == [("You are booked for Tuesday.", False)]
    assert tool_outputs(front_model.requests[-1]) == ["Booked for Tuesday."]


@pytest.mark.rule("R23", "B12")
async def test_a_turn_ending_on_verbatim_results_is_one_flagged_answer_said_as_written(
    delegate_to: Callable[..., Any],
) -> None:
    """Three verbatim results end the expert's step: one flagged answer in call order, given
    as written by the front desk with no model step."""
    expert_model = ScriptedLLM(
        Reply(tool_calls=[call("first", "e1"), call("second", "e2"), call("third", "e3")])
    )
    expert = Agent(instructions="policy", llm=expert_model)
    for name, text, delay in (
        ("first", "one", 0.2),
        ("second", "two", 0.0),
        ("third", "three", 0.1),
    ):

        async def ending(ctx: RunContext, text: str = text, delay: float = delay) -> ToolResult:
            await asyncio.sleep(delay)
            return ToolResult(text, verbatim=True, reply_required=False)

        expert.tool(name=name, description=name)(ending)

    front_model = ScriptedLLM(Reply(tool_calls=[call("ask", "c1", request="the policy")]))
    expert_frames, desk_frames = await delegate_to(expert, front_model, "the policy?")

    for frames in (expert_frames, desk_frames):
        assert working(frames) == []
        assert answer(frames) == [("one\n\ntwo\n\nthree", True)]
    assert len(expert_model.requests) == 1
    assert len(front_model.requests) == 1


# ---------------------------------------------------------------------------------------------
# the calls of one step run together
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R22", "R23")
async def test_the_calls_of_one_step_run_together_and_say_as_they_go(
    door: Callable[..., Door],
) -> None:
    """Two tools of one step both start before either ends, and each says its text as it goes."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("slow", "c1", delay=0.4), call("slow", "c2", delay=0.2)]),
        Reply("Done."),
    )
    desk = Agent(instructions="desk", llm=model)
    events: list[str] = []

    @desk.tool
    async def slow(ctx: RunContext, delay: float) -> str:
        """Slow."""
        events.append("start")
        await asyncio.sleep(delay)
        events.append("end")
        await ctx.say(f"after {delay}")  # type: ignore[attr-defined]
        return "done"

    sent = await door(serving(desk)).send("go")

    assert events == ["start", "start", "end", "end"]
    assert working(sent.frames) == [("after 0.2", True), ("after 0.4", True)]
    assert answer(sent.frames) == [("Done.", False)]


async def test_a_cancel_during_a_step_stops_every_call() -> None:
    """The run is cancelled while its step's two calls run: both are cancelled."""
    model = ScriptedLLM(Reply(tool_calls=[call("wait", "c1"), call("wait", "c2")]))
    desk = Agent(instructions="desk", llm=model)
    started: list[str] = []
    stopped: list[str] = []

    @desk.tool
    async def wait(ctx: RunContext) -> str:
        """Wait."""
        started.append(ctx.function_call.call_id)
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopped.append(ctx.function_call.call_id)
            raise
        return "done"

    run = asyncio.ensure_future(desk.run("go"))
    for _ in range(100):
        if len(started) == 2:
            break
        await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert sorted(stopped) == ["c1", "c2"]


@pytest.mark.rule("B46", "R23")
async def test_a_verbatim_result_on_a_typed_agent_is_said_and_the_output_keeps_its_type() -> None:
    """On a typed agent a verbatim result that would end the turn is said, and the model goes on."""

    class Booking(BaseModel):
        day: str

    model = ScriptedLLM(
        Reply(tool_calls=[call("disclose", "c1")]),
        Reply(tool_calls=[call("final_result", "c2", day="Tuesday")], when=output_of("c1")),
    )
    desk = Agent(instructions="desk", llm=model, output_type=Booking)

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True, reply_required=False)

    result = await desk.run("book me")

    assert result.output == Booking(day="Tuesday")
    assert tool_outputs(model.requests[-1]) == [DELIVERED.format(text=NOTICE)]
