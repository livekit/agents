"""A person corrects an answer or cancels one of several remote tasks during a voice call."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import (
    Agent,
    DelegationOptions,
    RunContext,
    SessionState,
    VoiceSession,
    persisted,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.storage import ContextStore
from tests.fakes import (
    FakeAudioInput,
    FakeLLM,
    FakeSTT,
    FakeVAD,
    Hang,
    last_input,
    tool_call,
    until,
)

from .a2a_peer import Peer

QUESTION = "Normal or high priority?"


@dataclass
class Desk:
    session: SessionState = persisted(default_factory=SessionState)


def task(
    message: dict[str, Any], task_id: str, state: str, text: str | None = None
) -> dict[str, Any]:
    status: dict[str, Any] = {"state": f"TASK_STATE_{state}"}
    if text is not None:
        status["message"] = {
            "messageId": f"{task_id}-{state.lower()}-{len(text)}",
            "role": "ROLE_AGENT",
            "parts": [{"text": text}],
        }
    return {
        "task": {"id": task_id, "contextId": message.get("contextId", "desk"), "status": status}
    }


def words(message: dict[str, Any]) -> str:
    return str(message["parts"][0]["text"])


def latest_human(history: ChatContext) -> str | None:
    return next(
        (
            item.text_content
            for item in reversed(history.items)
            if item.type == "message" and item.role == "user"
        ),
        None,
    )


def calls(history: ChatContext) -> list[Any]:
    return [item for item in history.items if item.type == "function_call"]


def tool_output(history: ChatContext, text: str) -> bool:
    return any(
        item.type == "function_call_output" and text in item.output for item in history.items
    )


def said(session: VoiceSession, text: str) -> bool:
    return any(
        item.type == "message" and item.role == "assistant" and item.text_content == text
        for item in session.history.items
    )


def heard(session: VoiceSession, text: str) -> bool:
    return any(
        item.type == "message" and item.role == "user" and item.text_content == text
        for item in session.history.items
    )


def ticket_desk(peer: Peer, frontend: FakeLLM) -> Agent:
    return Agent(
        name="desk",
        instructions="Open support tickets.",
        tools=[RemoteAgent(peer.url, name="tickets").as_tool()],
        llm=frontend,
    )


async def test_a_correction_before_dispatch_replaces_the_held_answer(peer: Peer) -> None:
    """The person answers "high", then speaks again with a correction while the frontend is
    still planning: the ticket service receives one answer, the corrected one."""

    def tickets(message: dict[str, Any]) -> list[dict[str, Any] | asyncio.Event]:
        if "taskId" not in message:
            return [task(message, "ticket", "INPUT_REQUIRED", QUESTION)]
        return [task(message, "ticket", "COMPLETED", f"Ticket opened: {words(message)}")]

    peer.events = tickets
    thinking = Hang()

    def frontend(history: ChatContext) -> str | FunctionToolCall | list[Any]:
        if (last_input(history) or "").startswith("Ask the person this"):
            return QUESTION
        human = latest_human(history)
        forwarded = len(calls(history)) > 1
        if human == "High":
            if not thinking.started.is_set():
                return [thinking]  # still planning when the person speaks again
            return "Got it." if forwarded else tool_call("tickets", request="High")
        if human == "Actually make it normal":
            if not forwarded:
                return tool_call("tickets", request="Normal priority")
            if tool_output(history, "Ticket opened"):
                return "Your ticket is open at normal priority."
            return "Updating that."
        if not calls(history):
            return tool_call("tickets", request="Open a ticket for my broken login")
        return "One moment."

    held: list[str | None] = []

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = json.loads(last_input(history) or "{}")
        if "proposed_call" in proposal:
            (candidate,) = proposal["candidates"]
            (response,) = candidate["responses"]
            return tool_call(
                "route_remote_call",
                action="respond",
                candidate=candidate["index"],
                response=response["index"],
            )
        (candidate,) = proposal["candidates"]
        held.append(candidate["held_response"])
        return tool_call("route_input_response", candidates=[0], unclear=False)

    store = ContextStore()
    microphone = FakeAudioInput()
    async with VoiceSession(
        context=store.bind(),
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
        vad=FakeVAD(),
        stt=FakeSTT(["Open a ticket for my broken login", "High", "Actually make it normal"]),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    ) as session:
        session.input.audio = microphone
        await session.start(ticket_desk(peer, FakeLLM(frontend)))
        microphone.speak(0.4)
        await until(lambda: said(session, QUESTION), "the priority question was not asked")

        microphone.speak(0.4)
        await until(thinking.started.is_set, "the frontend did not start planning")
        microphone.speak(0.4)
        await until(
            lambda: said(session, "Your ticket is open at normal priority."),
            "the corrected answer did not complete the ticket",
        )

        assert held == [None, "High"], "the router saw the held answer it may replace"
        assert [words(message) for message in peer.messages] == [
            "Open a ticket for my broken login",
            "Actually make it normal",
        ], "only the correction reaches the task, once"
        assert peer.messages[1]["taskId"] == "ticket"
        assert heard(session, "High"), "the earlier words stay in the conversation"
        saved = await store.load_session()
        (request,) = saved.remote_agents[peer.url].tasks["ticket"].input_requests.values()
        assert request.state == "responded"
    assert peer.cancelled == []


@pytest.mark.parametrize("acknowledged", [False, True])
async def test_a_correction_after_dispatch_is_sent_to_the_same_task(
    peer: Peer, acknowledged: bool
) -> None:
    """Once "high" was sent, the correction travels as another message to that task, after the
    peer admitted the answer it may already be acting on; the sent answer stays on record."""
    high_admitted = asyncio.Event()
    still_working = asyncio.Event()  # the peer keeps acting on "high" until the call ends
    if acknowledged:
        high_admitted.set()

    def tickets(message: dict[str, Any]) -> list[dict[str, Any] | asyncio.Event]:
        if "taskId" not in message:
            return [task(message, "ticket", "INPUT_REQUIRED", QUESTION)]
        if words(message) == "High":
            working = task(message, "ticket", "WORKING", "Raising the priority to high.")
            return [high_admitted, working, still_working]
        return [
            task(
                message, "ticket", "COMPLETED", "Ticket open; priority changed from high to normal."
            )
        ]

    peer.events = tickets

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        if (last_input(history) or "").startswith("Ask the person this"):
            return QUESTION
        if (last_input(history) or "").startswith("Present this remote tool outcome"):
            assert tool_output(history, "changed from high to normal")
            return "Done: the ticket is now normal priority."
        human = latest_human(history)
        if human == "High":
            if len(calls(history)) == 1:
                return tool_call("tickets", request="High")
            return "Got it, high priority."
        if human == "Actually make it normal":
            if len(calls(history)) == 2:
                correcting.set()
                return tool_call("tickets", request="Please change the priority to normal")
            return "I've asked them to change it to normal."
        if not calls(history):
            return tool_call("tickets", request="Open a ticket for my broken login")
        return "One moment."

    correcting = asyncio.Event()

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = json.loads(last_input(history) or "{}")
        if "proposed_call" in proposal:
            (candidate,) = proposal["candidates"]
            assert candidate["task_id"] == "ticket"
            if proposal["proposed_call"] == {"request": "High"}:
                return tool_call(
                    "route_remote_call", action="respond", candidate=candidate["index"], response=0
                )
            assert candidate["responses"] == [], "a sent answer is not offered for replacement"
            return tool_call(
                "route_remote_call", action="continue", candidate=candidate["index"], response=None
            )
        selected = [0] if proposal["latest_message"] == "High" else []
        return tool_call("route_input_response", candidates=selected, unclear=False)

    store = ContextStore()
    app = await store.load_context(Desk())
    microphone = FakeAudioInput()
    async with VoiceSession(
        context=store.bind(app),
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
        vad=FakeVAD(),
        stt=FakeSTT(["Open a ticket for my broken login", "High", "Actually make it normal"]),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    ) as session:
        session.input.audio = microphone
        await session.start(ticket_desk(peer, FakeLLM(frontend)))
        microphone.speak(0.4)
        await until(lambda: said(session, QUESTION), "the priority question was not asked")
        original = calls(session.history)[0]

        microphone.speak(0.4)
        await until(lambda: len(peer.messages) == 2, "the answer was not sent")
        ticket = app.session.remote_agents[peer.url].tasks["ticket"]
        if acknowledged:
            await until(
                lambda: ticket.task.status.state == pb.TASK_STATE_WORKING,
                "the peer's acknowledgement was not observed",
            )
        microphone.speak(0.4)
        # While "high" is unanswered, the correction waits for it rather than racing it.
        await until(correcting.is_set, "the frontend did not take up the correction")
        high_admitted.set()
        await until(
            lambda: said(session, "Done: the ticket is now normal priority."),
            "the correction's outcome was not presented",
        )
        assert [words(message) for message in peer.messages] == [
            "Open a ticket for my broken login",
            "High",
            "Please change the priority to normal",
        ]
        assert [message.get("taskId") for message in peer.messages[1:]] == ["ticket", "ticket"]
        outputs = [item for item in session.history.items if item.type == "function_call_output"]
        final = [item for item in outputs if "changed from high to normal" in item.output]
        assert [item.id for item in final] == [f"{original.id}_out"], (
            "the original call owns the one result"
        )
        (request,) = ticket.input_requests.values()
        assert request.state == "responded", "the sent answer is not rewritten"
        assert heard(session, "High")
        assert ticket.task.status.state == pb.TASK_STATE_COMPLETED
    assert peer.cancelled == []


@pytest.mark.parametrize("via", ["cancel_tool", "task_message"])
async def test_cancelling_one_task_keeps_the_others_running_and_delivers_their_results(
    peer: Peer, via: str
) -> None:
    """ "Cancel the export" stops only the export; the report still arrives, and ending the call
    leaves the archive running at the peer."""
    report_ready = asyncio.Event()
    never = asyncio.Event()

    def jobs(message: dict[str, Any]) -> list[dict[str, Any] | asyncio.Event]:
        request = words(message)
        if message.get("taskId") == "export":
            return [task(message, "export", "CANCELED", "Export cancelled.")]
        if request == "Export my data":
            return [task(message, "export", "WORKING"), never]
        if request == "Build the sales report":
            return [
                task(message, "report", "WORKING"),
                report_ready,
                task(message, "report", "COMPLETED", "The sales report is ready."),
            ]
        assert request == "Archive old files"
        return [task(message, "archive", "WORKING"), never]

    peer.events = jobs
    started = {
        "Export my data": "Exporting.",
        "Build the sales report": "Building it.",
        "Archive old files": "Archiving.",
    }

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        human = latest_human(history) or ""
        if tool_output(history, "The sales report is ready.") and not any(
            item.type == "message" and item.text_content == "Your sales report is ready."
            for item in history.items
        ):
            return "Your sales report is ready."
        if human in started:
            if not any(human in item.arguments for item in calls(history)):
                return tool_call("jobs", request=human)
            return started[human]
        if human == "Cancel the export":
            if not any("export" in item.arguments for item in calls(history)):
                if via == "cancel_tool":
                    return tool_call("cancel_job", task_id="export")
                return tool_call("jobs", request="Cancel the export", task_id="export")
            return "The export is cancelled."
        return "Okay."

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = json.loads(last_input(history) or "{}")
        if "proposed_call" not in proposal:
            return tool_call("route_input_response", candidates=[], unclear=False)
        if proposal["selected_task"] is not None:
            (candidate,) = proposal["candidates"]
            return tool_call(
                "route_remote_call", action="continue", candidate=candidate["index"], response=None
            )
        return tool_call("route_remote_call", action="new", candidate=None, response=None)

    desk = Agent(
        name="desk",
        instructions="Run back-office jobs.",
        tools=[RemoteAgent(peer.url, name="jobs").as_tool()],
        llm=FakeLLM(frontend),
    )

    @desk.tool
    async def cancel_job(ctx: RunContext[Desk], task_id: str) -> str:
        """Cancel one running job the person no longer wants."""
        await ctx.context.session.remote_agents[peer.url].tasks[task_id].cancel()
        return "Cancelled."

    store = ContextStore()
    app = await store.load_context(Desk())
    async with VoiceSession(
        context=store.bind(app),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(desk)
        for request in started:
            await session.generate_reply(user_input=request)
        await until(
            lambda: all(said(session, reply) for reply in started.values()),
            "the three jobs were not started",
        )
        tasks = app.session.remote_agents[peer.url].tasks
        assert set(tasks) == {"export", "report", "archive"}

        await session.generate_reply(user_input="Cancel the export")
        await until(
            lambda: said(session, "The export is cancelled."), "the cancel was not confirmed"
        )
        assert tasks["export"].task.status.state == pb.TASK_STATE_CANCELED
        assert peer.cancelled == (["export"] if via == "cancel_tool" else [])

        report_ready.set()
        await until(
            lambda: said(session, "Your sales report is ready."),
            "the report's result was not delivered after the export was cancelled",
        )
        assert tasks["report"].task.status.state == pb.TASK_STATE_COMPLETED
        assert tasks["archive"].task.status.state == pb.TASK_STATE_WORKING

    assert peer.cancelled == (["export"] if via == "cancel_tool" else []), (
        "ending the call must not cancel remote work"
    )
    saved = await store.load_session()
    archive = saved.remote_agents[peer.url].tasks["archive"]
    assert archive.task.status.state == pb.TASK_STATE_WORKING
    assert not archive.cancel_requested
