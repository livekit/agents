"""Real frontend/routing models using public sessions and a stock A2A peer process."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, DelegationOptions, SessionState, TextSession, persisted
from livekit.agents.a2a import RemoteAgent
from livekit.agents.storage import ContextStore

from .framework.app import E2E
from .framework.models import live_models, write_live_report

pytestmark = pytest.mark.live_models


@dataclass
class Conversation:
    session: SessionState = persisted(default_factory=SessionState)


@pytest.mark.parametrize(
    ("human", "responds"),
    [
        ("Friday", True),
        ("Why do you need the date?", False),
        ("What can you help me with?", False),
        ("Yes.", False),
    ],
)
async def test_a_live_frontend_and_gemma_route_the_persons_intent(
    e2e: E2E, tmp_path: Path, human: str, responds: bool
) -> None:
    model_ids = {
        "frontend": os.environ.get("E2E_FRONTEND_MODEL", "google/gemma-4-31b-it"),
        "routing": os.environ.get("E2E_ROUTING_MODEL", "google/gemma-4-31b-it"),
    }
    peer = await e2e.start_app("tests.e2e.apps.stock_a2a")
    store = ContextStore()
    app = await store.load_context(Conversation())
    turns: list[dict[str, object]] = []
    async with live_models(**model_ids) as (frontend, routing):
        agent = Agent(
            name="desk",
            instructions=(
                "You are the conversational front desk; travel owns flight work, required details "
                "and follow-up questions. Delegate new flight requests to travel immediately, "
                "including requests with missing information. Do not collect booking fields "
                "yourself before asking travel. Present the questions travel actually asks. "
                "The travel agent needs the date to check availability; you can explain that "
                "yourself. When the person answers a pending travel question, call travel to "
                "pass on their answer. Do not claim confirmation before the tool confirms. "
                "Handle side questions yourself and clarify ambiguous responses naturally."
            ),
            llm=frontend,
            tools=[RemoteAgent(peer.url, name="travel").as_tool()],
        )
        session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))

        async def say(text: str) -> None:
            started = perf_counter()
            observed: dict[str, object] = {"human": text}
            turns.append(observed)
            try:
                result = await session.run(text, context=store.bind(app))
                observed.update(
                    status=result.status,
                    assistant=result.output if result.status == "completed" else result.question,
                )
            except Exception as error:
                observed["error_type"] = type(error).__name__
                raise
            finally:
                observed["seconds"] = perf_counter() - started

        try:
            await say("Please book a flight for me.")
            assert peer.url in app.session.remote_agents, (
                "the frontend did not delegate the flight request"
            )
            remote = app.session.remote_agents[peer.url]
            [task_id] = remote.tasks
            assert remote.tasks[task_id].task.status.state == pb.TASK_STATE_INPUT_REQUIRED
            [request_id] = remote.tasks[task_id].input_requests
            assert remote.tasks[task_id].input_requests[request_id].state == "open"

            await say(human)
            remote = app.session.remote_agents[peer.url]
            request = remote.tasks[task_id].input_requests[request_id]
            sent = [
                message
                for message in remote.messages.values()
                if message.role == pb.ROLE_USER and message.task_id == task_id
            ]
            if responds:
                assert len(sent) == 1, "the recognized response never reached its actual task"
                assert "".join(part.text for part in sent[0].parts) == human
                assert request.response_message_id == sent[0].message_id
                assert remote.tasks[task_id].task.status.state == pb.TASK_STATE_COMPLETED
            else:
                assert sent == [], "an unanswered side question was forwarded as the task's answer"
                assert request.state == "open" and request.response_message_id is None
        finally:
            write_live_report(tmp_path, models=model_ids, turns=turns, state=app.session)


@pytest.mark.parametrize("partial", [False, True])
async def test_gemma_groups_related_remote_questions_and_routes_full_or_partial_names(
    e2e: E2E,
    tmp_path: Path,
    partial: bool,
) -> None:
    model_ids = {
        "frontend": os.environ.get("E2E_FRONTEND_MODEL", "google/gemma-4-31b-it"),
        "routing": os.environ.get("E2E_ROUTING_MODEL", "google/gemma-4-31b-it"),
    }
    peers = [
        await e2e.start_app(
            "tests.e2e.apps.stock_a2a",
            env={
                "STOCK_QUESTION": f"What is your {field} name?",
                "STOCK_DESCRIPTION": f"Collect and confirm the person's {field} name.",
                "STOCK_RESULT_PREFIX": f"Confirmed {field} name: ",
            },
        )
        for field in ("first", "last")
    ]
    store = ContextStore()
    app = await store.load_context(Conversation())
    turns: list[dict[str, object]] = []
    async with live_models(**model_ids) as (frontend, routing):
        agent = Agent(
            name="intake",
            llm=frontend,
            instructions=(
                "You are the conversational check-in desk. first_name and last_name own name "
                "collection and confirmation. For a check-in request, immediately call BOTH "
                "tools in parallel in the same response to start their work; do not collect "
                "fields before delegation. Present related questions together naturally. "
                "When a person answers, call the tool for each field they actually supplied. "
                "If they supplied only one name, pass it on and naturally ask for the missing "
                "field. Never guess a missing field or claim it has been confirmed."
            ),
            tools=[
                RemoteAgent(peer.url, name=f"{field}_name").as_tool()
                for peer, field in zip(peers, ("first", "last"), strict=True)
            ],
        )
        session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))

        async def say(text: str) -> None:
            started = perf_counter()
            observed: dict[str, object] = {"human": text}
            turns.append(observed)
            try:
                result = await session.run(text, context=store.bind(app))
                observed.update(
                    status=result.status,
                    assistant=result.output if result.status == "completed" else result.question,
                )
            except Exception as error:
                observed["error_type"] = type(error).__name__
                raise
            finally:
                observed["seconds"] = perf_counter() - started

        try:
            await say("Please check me in. Start collecting my first and last name.")
            references = []
            for peer in peers:
                remote = app.session.remote_agents[peer.url]
                [task_id] = remote.tasks
                [request_id] = remote.tasks[task_id].input_requests
                references.append((peer.url, task_id, request_id))

            answer = "My first name is Ada." if partial else "Ada Lovelace"
            await say(answer)
            for index, (remote_id, task_id, request_id) in enumerate(references):
                remote = app.session.remote_agents[remote_id]
                request = remote.tasks[task_id].input_requests[request_id]
                if partial and index == 1:
                    assert request.state == "open" and request.response_message_id is None
                else:
                    assert request.response_message_id is not None
                    sent = remote.messages[request.response_message_id]
                    assert sent.task_id == task_id
                    assert "".join(part.text for part in sent.parts) == answer
                    assert remote.tasks[task_id].task.status.state == pb.TASK_STATE_COMPLETED
            if partial:
                await say("My last name is Lovelace.")
                remote_id, task_id, request_id = references[1]
                remote = app.session.remote_agents[remote_id]
                request = remote.tasks[task_id].input_requests[request_id]
                assert request.response_message_id is not None
                sent = remote.messages[request.response_message_id]
                assert sent.task_id == task_id
                assert "".join(part.text for part in sent.parts) == "My last name is Lovelace."
                assert remote.tasks[task_id].task.status.state == pb.TASK_STATE_COMPLETED
        finally:
            write_live_report(tmp_path, models=model_ids, turns=turns, state=app.session)
