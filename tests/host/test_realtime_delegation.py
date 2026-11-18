"""A realtime frontend delegating to a remote A2A agent: the same contract as a pipeline call."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, APIConnectOptions, DelegationOptions, VoiceSession, _core
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.storage import ContextStore
from tests.fakes import (
    FakeLLM,
    FakeRealtime,
    Hang,
    RealtimeReply,
    last_input,
    tool_call,
    until,
)

from .a2a_peer import Peer

ASK = "Ask the person this, in your own words, as your whole reply, and add nothing else: "
PROGRESS = "Tell the person how their request with the expert is going"
OUTCOME = "Present this remote tool outcome to the person"


class Desk:
    """A frontend on the session's realtime ``model`` whose one tool is the remote expert: each
    new human message becomes one call carrying its words, and each framework instruction is
    spoken as asked."""

    def __init__(self, peer: Peer) -> None:
        self.remote = RemoteAgent(
            peer.url,
            name="coverage_expert",
            description="Checks account coverage.",
            conn_options=APIConnectOptions(max_retry=0),
        )
        self.model = FakeRealtime(self.reply)
        self.agent = Agent(name="desk", instructions="Help.", tools=[self.remote.as_tool()])
        self.forwarded: set[str] = set()
        self.outcomes: list[dict[str, Any]] = []

    def reply(self, history: ChatContext, instructions: str | None) -> RealtimeReply:
        if instructions is not None and instructions.startswith(ASK):
            return instructions.removeprefix(ASK)
        if instructions is not None and instructions.startswith(PROGRESS):
            return "Still checking."
        if instructions is not None and instructions.startswith(OUTCOME):
            self.outcomes.append(json.loads(instructions[instructions.index("{") :]))
            return f"Result: {self.outcomes[-1]['output']}"
        latest = latest_human(history)
        if latest is not None and latest not in self.forwarded:
            self.forwarded.add(latest)
            return tool_call("coverage_expert", request=latest)
        return "I'm on it."

    def instructions(self, prefix: str) -> list[str]:
        return [i for _, i in self.model.requests if i is not None and i.startswith(prefix)]


def latest_human(history: ChatContext) -> str | None:
    return next(
        (
            item.text_content
            for item in reversed(history.items)
            if item.type == "message" and item.role == "user"
        ),
        None,
    )


def said(session: VoiceSession, text: str) -> int:
    return sum(
        item.type == "message" and item.role == "assistant" and item.text_content == text
        for item in session.history.items
    )


def route(history: ChatContext) -> FunctionToolCall:
    """Routes ``Friday`` to the open question and a call carrying it to that response."""
    value = last_input(history)
    assert value is not None
    prompt = json.loads(value)
    if "proposed_call" in prompt:
        [candidate] = prompt["candidates"]
        [response] = candidate["responses"]
        assert candidate["task_id"] == "remote-task"
        assert response["held_response"] == "Friday"
        return tool_call(
            "route_remote_call",
            action="respond",
            candidate=candidate["index"],
            response=response["index"],
        )
    [candidate] = prompt["candidates"]
    assert candidate["question"] == "Which date?"
    return tool_call(
        "route_input_response",
        candidates=[candidate["index"]] if prompt["latest_message"] == "Friday" else [],
        unclear=False,
    )


async def open_requests(store: ContextStore, peer: Peer) -> list[str]:
    state = await store.load_session()
    task = state.remote_agents[peer.url].tasks["remote-task"]
    return [request.state for request in task.input_requests.values()]


# Catches: a realtime frontend that never presents the expert's question, or that sends the
# person's answer as new work instead of continuing the task that asked.
async def test_a_realtime_call_presents_the_remote_question_and_answers_the_same_task(
    peer: Peer,
) -> None:
    peer.question = "Which date?"
    desk = Desk(peer)
    store = ContextStore()
    async with VoiceSession(
        llm=desk.model,
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Which date?"), "the remote question was not presented")
        assert await open_requests(store, peer) == ["open"]

        session.generate_reply(user_input="Friday")
        await until(lambda: said(session, "Result: Confirmed"), "the result was not presented")
        assert len(peer.messages) == 2
        assert peer.messages[1]["taskId"] == "remote-task"
        assert peer.messages[1]["parts"] == [{"text": "Friday"}]
        assert await open_requests(store, peer) == ["responded"]


# Catches: an expert's flagged progress dropped on a realtime call because only a pipeline
# reply can word it.
async def test_a_realtime_call_words_the_remote_progress_while_the_task_works(peer: Peer) -> None:
    peer.hold = True
    desk = Desk(peer)
    async with VoiceSession(llm=desk.model, vad=None) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Still checking."), "the progress was not presented")
        [progress] = desk.instructions(PROGRESS)
        assert progress.endswith("Checking")
        assert not desk.outcomes


# Catches: a realtime call presenting a remote result twice, or leaving a second output
# beside the call it resolves.
async def test_a_realtime_call_delivers_the_remote_result_once(peer: Peer) -> None:
    desk = Desk(peer)
    async with VoiceSession(llm=desk.model, vad=None) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Result: Confirmed"), "the result was not presented")
        await session.wait_for_idle()
        call = next(i for i in session.history.items if i.type == "function_call")
        outputs = [i for i in session.history.items if i.type == "function_call_output"]
        assert [(o.id, o.output, o.is_error) for o in outputs] == [
            (f"{call.id}_out", "Confirmed", False)
        ]
        assert desk.outcomes == [
            {"tool": "coverage_expert", "is_error": False, "output": "Confirmed"}
        ]
        assert said(session, "Result: Confirmed") == 1
        assert len(peer.messages) == 1


# Catches: a realtime agent replacement dropping the expert's pending question, so the answer
# given once the desk returns is lost or opens new work.
async def test_replacing_the_realtime_agent_keeps_the_pending_remote_question(
    peer: Peer,
) -> None:
    peer.question = "Which date?"
    desk = Desk(peer)
    helper = Agent(name="helper", instructions="Help.")
    desk_entered, helper_entered = asyncio.Event(), asyncio.Event()

    @desk.agent.on_enter
    async def returned_to_desk(session: VoiceSession) -> None:
        desk_entered.set()

    @helper.on_enter
    async def entered_helper(session: VoiceSession) -> None:
        helper_entered.set()

    store = ContextStore()
    async with VoiceSession(
        llm=desk.model,
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Which date?"), "the remote question was not presented")

        session.update_agent(helper)
        await until(helper_entered.is_set, "the helper never took over")
        await session.wait_for_idle()
        assert await open_requests(store, peer) == ["open"]
        desk_entered.clear()
        session.update_agent(desk.agent)
        await until(desk_entered.is_set, "the desk never returned")
        await session.wait_for_idle()
        assert await open_requests(store, peer) == ["open"]

        session.generate_reply(user_input="Friday")
        await until(lambda: said(session, "Result: Confirmed"), "the result was not presented")
        assert len(peer.messages) == 2
        assert peer.messages[1]["taskId"] == "remote-task"
        assert peer.messages[1]["parts"] == [{"text": "Friday"}]


# Catches: switching to an agent on another realtime model, and back, dropping the expert's
# pending question with the replaced connection, so the answer is lost or opens new work.
async def test_switching_realtime_models_keeps_the_pending_remote_question(peer: Peer) -> None:
    peer.question = "Which date?"
    desk = Desk(peer)
    helper_model = FakeRealtime("Helper here.")
    helper = Agent(name="helper", instructions="Help.", llm=helper_model)
    desk_entered, helper_entered = asyncio.Event(), asyncio.Event()

    @desk.agent.on_enter
    async def returned_to_desk(session: VoiceSession) -> None:
        desk_entered.set()

    @helper.on_enter
    async def entered_helper(session: VoiceSession) -> None:
        helper_entered.set()

    store = ContextStore()
    async with VoiceSession(
        llm=desk.model,
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Which date?"), "the remote question was not presented")

        session.update_agent(helper)
        await until(helper_entered.is_set, "the helper never took over")
        await session.wait_for_idle()
        assert desk.model.sessions[0].closed
        assert await open_requests(store, peer) == ["open"]
        desk_entered.clear()
        session.update_agent(desk.agent)
        await until(desk_entered.is_set, "the desk never returned")
        await session.wait_for_idle()
        assert helper_model.sessions[0].closed
        assert len(desk.model.sessions) == 2
        assert await open_requests(store, peer) == ["open"]

        session.generate_reply(user_input="Friday")
        await until(lambda: said(session, "Result: Confirmed"), "the result was not presented")
        assert len(peer.messages) == 2
        assert peer.messages[1]["taskId"] == "remote-task"
        assert peer.messages[1]["parts"] == [{"text": "Friday"}]


# Catches: closing a realtime call leaving a host awaiting a reply forever, or cancelling the
# remote business task the call only observed.
async def test_closing_a_realtime_call_settles_its_receipts_and_leaves_the_remote_task(
    peer: Peer,
) -> None:
    peer.hold = True
    desk = Desk(peer)
    store = ContextStore()
    session = VoiceSession(llm=desk.model, context=store.bind(), vad=None)
    async with session:
        await session.start(desk.agent)
        first = session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Still checking."), "the remote task never started")
        later = session.generate_reply(user_input="Any news?")
        idle = asyncio.ensure_future(session.wait_for_idle())
    async with asyncio.timeout(2):
        await first
        await later
        with pytest.raises(_core.SessionClosing):
            await idle
    assert peer.cancelled == [], "closing the call must not cancel the remote task"
    state = await store.load_session()
    task = state.remote_agents[peer.url].tasks["remote-task"]
    assert task.task.status.state == pb.TASK_STATE_WORKING
    assert not task.cancel_requested


# Catches: a realtime call the model proposed while the person's words were still being routed
# keeping its interrupted reply, and the session's close, waiting on the routing model.
@pytest.mark.parametrize("cut", ["interrupt", "close"])
async def test_cutting_a_realtime_reply_during_response_routing_drops_the_unadmitted_call(
    peer: Peer, cut: str
) -> None:
    peer.question = "Which date?"
    desk = Desk(peer)
    detector = Hang()
    store = ContextStore()
    session = VoiceSession(
        llm=desk.model,
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(lambda history: detector)),
    )
    async with session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(lambda: said(session, "Which date?"), "the remote question was not presented")
        speech = session.generate_reply(user_input="Friday")
        await until(detector.started.is_set, "the response routing never started")
        await until(lambda: len(desk.model.calls_read) == 2, "the model never proposed its call")
        async with asyncio.timeout(2):
            await (session.interrupt() if cut == "interrupt" else session.aclose())
            await speech
        await until(detector.ended.is_set, "the cut reply left the routing model running")
        assert len(peer.messages) == 1, "an unadmitted call reached the remote task"
        assert await open_requests(store, peer) == ["open"]
    assert peer.cancelled == []


# Catches: a failed remote task on a realtime call not becoming the calling tool's error.
async def test_a_remote_failure_is_the_realtime_tools_error(peer: Peer) -> None:
    peer.fail = "coverage unavailable"
    desk = Desk(peer)
    async with VoiceSession(llm=desk.model, vad=None) as session:
        await session.start(desk.agent)
        session.generate_reply(user_input="Check coverage")
        await until(
            lambda: said(session, "Result: coverage unavailable"), "the failure was not presented"
        )
        [output] = [i for i in session.history.items if i.type == "function_call_output"]
        assert output.is_error and output.output == "coverage unavailable"
        assert desk.outcomes == [
            {"tool": "coverage_expert", "is_error": True, "output": "coverage unavailable"}
        ]
