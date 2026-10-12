"""The twelve checks of plan 10.1 that run locally, against the live stack of examples/.

Gated by ``V0_STACK=1``; skipped otherwise, and when the stack or the workers are not up. Needs
the fast brain and the slow brain, livekit-server and agent-db as examples/README.md starts
them, with the same environment. Not here: 6's crash (the storage crate's DST tests), 10's two
regions, 11's SIP call.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

STACK = bool(os.environ.get("V0_STACK"))
pytestmark = [
    pytest.mark.unit,  # the category; every test skips unless V0_STACK=1 names the live stack
    pytest.mark.skipif(not STACK, reason="V0_STACK=1 runs these against the local stack"),
]

LIVEKIT_HTTP = os.environ.get("LIVEKIT_HTTP", "http://127.0.0.1:7880")
AGENT = "dentist"
BASE = f"{LIVEKIT_HTTP}/agents/{AGENT}/default"
EP = f"{BASE}/frontend/v1"
SLOW_BRAIN = "backend"  # the back office's lane
SLOW_BASE = BASE  # the back office is a door of the same worker, on its own lane
SLOW_EP = f"{SLOW_BASE}/backend/v1"
EXT = "https://livekit.io/a2a/ext/agents/v1"
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED"}


def project_token() -> str:
    from livekit.agents.a2a.auth import Credentials

    return Credentials(
        os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ).project_token()


def message(
    text: str, *, context_id: str | None = None, message_id: str | None = None, **meta: Any
) -> dict:
    m: dict[str, Any] = {
        "messageId": message_id or f"m_{uuid.uuid4().hex[:10]}",
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id:
        m["contextId"] = context_id
    if meta:
        m["metadata"] = {f"{EXT}/{k}": v for k, v in meta.items()}
    return m


class Dentist:
    """A client of the front-desk endpoint under the project credential. In v0 a thread grant
    is for the threads API of the platform; through the node it is stripped like any LiveKit
    bearer."""

    def __init__(self) -> None:
        self.token = project_token()
        self.http = httpx.AsyncClient(timeout=120.0)

    @property
    def headers(self) -> dict[str, str]:
        return {"A2A-Extensions": EXT, "Authorization": f"Bearer {self.token}"}

    def body(self, method: str, params: dict) -> dict:
        return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}

    async def rpc(
        self, method: str, params: dict, *, token: str | None = None, url: str = EP
    ) -> dict:
        headers = {**self.headers, "Authorization": f"Bearer {token}"} if token else self.headers
        r = await self.http.post(url, json=self.body(method, params), headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    async def stream(self, method: str, params: dict) -> list[dict]:
        frames: list[dict] = []
        async with self.http.stream(
            "POST", EP, json=self.body(method, params), headers=self.headers
        ) as r:
            assert r.status_code == 200, await r.aread()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[6:]))
        return frames

    async def send(self, text: str, *, token: str | None = None, url: str = EP, **kw: Any) -> dict:
        body = await self.rpc(
            "message/send", {"message": message(text, **kw)}, token=token, url=url
        )
        assert "result" in body, body
        return body["result"]["task"]

    async def tasks(self, context_id: str, *, token: str | None = None) -> list[dict]:
        body = await self.rpc("tasks/list", {"contextId": context_id}, token=token)
        return body["result"]["tasks"]

    async def history(self, context_id: str, *, agent: str = AGENT) -> list[dict]:
        """One lane's items, read from the thread's database."""
        from livekit.agents.storage import AgentDBStorage
        from livekit.agents.storage.resolve import Thread

        history = await Thread(AgentDBStorage(), context_id, agent_name=agent).chat_history()
        return [item.model_dump(mode="json") for item in history.items]

    async def aclose(self) -> None:
        await self.http.aclose()


async def stack_is_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            card = await http.get(f"{BASE}/frontend/.well-known/agent-card.json")
            slow = await http.get(
                f"{SLOW_BASE}/backend/.well-known/agent-card.json",
                headers={"Authorization": f"Bearer {project_token()}"},
            )
        return card.status_code == 200 and slow.status_code == 200
    except (httpx.HTTPError, KeyError):
        return False


@pytest.fixture
async def dentist() -> AsyncIterator[Dentist]:
    if not await stack_is_up():
        pytest.skip("the fast and slow brains are not reachable through livekit-server")
    c = Dentist()
    yield c
    await c.aclose()


def artifact_text(task: dict) -> str | None:
    for a in task.get("artifacts") or []:
        for p in a.get("parts") or []:
            if "text" in p:
                return p["text"]
    return None


def status_updates(frames: list[dict]) -> list[dict]:
    return [f["result"]["statusUpdate"] for f in frames if "statusUpdate" in f.get("result", {})]


def final_status(frames: list[dict]) -> dict:
    return status_updates(frames)[-1]


def progress(frames: list[dict]) -> list[str]:
    """The texts of the working frames: the slow brain's ``ctx.update()`` relayed."""
    return [
        p["text"]
        for u in status_updates(frames)
        if u["status"]["state"] == "TASK_STATE_WORKING" and u["status"].get("message")
        for p in u["status"]["message"].get("parts") or []
        if "text" in p
    ]


# -- 1 thin slice, 5 idempotency ----------------------------------------------------------


async def test_01_thin_slice_and_05_idempotency(dentist: Dentist) -> None:
    first_id = f"m_{uuid.uuid4().hex[:10]}"
    task_1 = await dentist.send("Hello, I have a question about the practice", message_id=first_id)
    thread_id = task_1["contextId"]
    assert thread_id.startswith("AT_")
    assert task_1["status"]["state"] == "TASK_STATE_COMPLETED"

    task_2 = await dentist.send("Repeat my first message word for word.", context_id=thread_id)
    assert task_2["status"]["state"] == "TASK_STATE_COMPLETED"
    # answered from the lane's history, not from the request: the first message is in it
    assert "question about the practice" in (artifact_text(task_2) or "").lower()
    listed = {t["id"] for t in await dentist.tasks(thread_id)}
    assert {task_1["id"], task_2["id"]} <= listed

    # 5: a resent messageId returns the existing task and makes no second one
    before = len(await dentist.tasks(thread_id))
    again = await dentist.send(
        "Hello, I have a question about the practice", context_id=thread_id, message_id=first_id
    )
    assert again["id"] == task_1["id"]
    assert len(await dentist.tasks(thread_id)) == before


# -- 2 resume ----------------------------------------------------------------------------


async def test_02_resubscribe_replays_the_task_and_every_frame(dentist: Dentist) -> None:
    task = await dentist.send("Hello there")
    frames = await dentist.stream(
        "tasks/resubscribe", {"id": task["id"], "contextId": task["contextId"]}
    )
    kinds = [next(iter(f["result"])) for f in frames]
    assert kinds[0] == "task" and kinds[-1] == "statusUpdate"
    assert "artifactUpdate" in kinds
    assert final_status(frames)["status"]["state"] == "TASK_STATE_COMPLETED"


# -- 3 interrupt, 4 inbox -----------------------------------------------------------------


async def test_03_interrupt_two_streams_within_a_second(dentist: Dentist) -> None:
    task = await dentist.send("Hello")
    thread_id = task["contextId"]
    first = asyncio.create_task(
        dentist.stream(
            "message/stream",
            {
                "message": message(
                    "What openings do you have in the next two weeks?", context_id=thread_id
                )
            },
        )
    )
    await asyncio.sleep(0.8)
    second = asyncio.create_task(
        dentist.stream(
            "message/stream",
            {
                "message": message(
                    "Actually, what openings do you have in the next month?", context_id=thread_id
                )
            },
        )
    )
    frames_1, frames_2 = await asyncio.gather(first, second)
    final_1, final_2 = final_status(frames_1), final_status(frames_2)
    assert final_1["status"]["state"] == "TASK_STATE_CANCELED"
    assert final_1["metadata"][f"{EXT}/interrupted"] is True
    assert final_1["metadata"][f"{EXT}/interruptionReason"] == "new-message"
    assert not any("artifactUpdate" in f["result"] for f in frames_1)
    assert final_2["status"]["state"] == "TASK_STATE_COMPLETED"
    assert any("artifactUpdate" in f["result"] for f in frames_2)
    # the delegation's progress reaches the patient while the slow brain works
    assert any("calendar" in p for p in progress(frames_2)), progress(frames_2)
    items = await dentist.history(thread_id)
    assert any(i["type"] == "function_call" and i["name"] == "ask" for i in items)
    # the slow brain did the check, on its own lane of the same session
    slow = await dentist.history(thread_id, agent=SLOW_BRAIN)
    assert any(i["type"] == "function_call" and i["name"] == "list_available_slots" for i in slow)


async def test_04_inbox_burst_is_one_reply_and_a_merge(dentist: Dentist) -> None:
    task = await dentist.send("Hello")
    thread_id = task["contextId"]
    first = asyncio.create_task(
        dentist.send("Where are you located", context_id=thread_id, channel="sms")
    )
    await asyncio.sleep(0.3)
    second = asyncio.create_task(
        dentist.send("and when do you open", context_id=thread_id, channel="sms")
    )
    task_1, task_2 = await asyncio.gather(first, second)
    assert task_1["status"]["state"] == "TASK_STATE_COMPLETED" and "artifacts" not in task_1
    assert task_1["metadata"][f"{EXT}/mergedInto"] == task_2["id"]
    assert artifact_text(task_2)


# -- 7 human in the loop --------------------------------------------------------------------


async def test_07_input_required_from_a_tool_then_the_answer(dentist: Dentist) -> None:
    # the question is the slow brain's: its schedule_appointment asks for the email, on its own
    # lane of the patient's session, and the front desk relays it; the front desk's own turn
    # completes
    task_1 = await dentist.send("Book me the earliest opening you have, whatever it is.")
    thread_id = task_1["contextId"]
    assert task_1["status"]["state"] == "TASK_STATE_COMPLETED"
    asked = next(
        t
        for t in await dentist.tasks(thread_id)
        if t["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    )
    assert (
        asked["status"]["message"]["parts"][0]["text"]
        == "What email address should the confirmation go to?"
    )

    # the next request on the slow brain's lane is that call's output; its model continues
    task_2 = await dentist.send("jane@example.com", context_id=thread_id)
    assert task_2["status"]["state"] == "TASK_STATE_COMPLETED"
    assert artifact_text(task_2), "the turn after the answer carries a reply"
    items = await dentist.history(thread_id, agent=SLOW_BRAIN)
    outputs = [
        i
        for i in items
        if i["type"] == "function_call_output" and i["name"] == "schedule_appointment"
    ]
    # the answer is the pending call's output, as the front desk relayed it; a model that
    # then calls the tool again with the email is asked to confirm, not for the email again
    assert "jane@example.com" in outputs[0]["output"]
    assert all("What email" not in o["output"] for o in outputs[1:])
    asked = next(t for t in await dentist.tasks(thread_id) if t["id"] == asked["id"])
    assert asked["status"]["state"] == "TASK_STATE_COMPLETED"
    assert f"{EXT}/mergedInto" in asked["metadata"]


@pytest.mark.parametrize("answer", ["yes", "no"])
async def test_07_requires_approval_asks_yes_or_no(dentist: Dentist, answer: str) -> None:
    # the approval is the slow brain's schedule_appointment; asked the way the fast brain
    # delegates, on a lane of the patient's session
    thread_id = (await dentist.send("Hello"))["contextId"]
    token = project_token()
    listed = await dentist.send(
        "What openings do you have in the next three months?",
        context_id=thread_id,
        token=token,
        url=SLOW_EP,
    )
    assert listed["status"]["state"] == "TASK_STATE_COMPLETED"
    items = await dentist.history(thread_id, agent=SLOW_BRAIN)
    slot = next(
        i["output"].split()[0]
        for i in items
        if i["type"] == "function_call_output" and i["name"] == "list_available_slots"
    )
    task_1 = await dentist.send(
        f"book slot {slot} for jane@example.com",
        context_id=thread_id,
        token=token,
        url=SLOW_EP,
    )
    assert task_1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    question = task_1["status"]["message"]["parts"][0]["text"]
    assert question.startswith("Please confirm: an appointment on")
    assert "jane@example.com" in question and question.endswith("Yes or no?")

    task_2 = await dentist.send(answer, context_id=thread_id, token=token, url=SLOW_EP)
    assert task_2["status"]["state"] == "TASK_STATE_COMPLETED"
    assert artifact_text(task_2), "the turn after the answer carries a reply"
    items = await dentist.history(thread_id, agent=SLOW_BRAIN)
    calls = [
        i for i in items if i["type"] == "function_call" and i["name"] == "schedule_appointment"
    ]
    ran = [
        i
        for i in items
        if i["type"] == "function_call_output"
        and i["name"] == "schedule_appointment"
        and i["output"].startswith("booked ")  # the body's return, not the model's words
    ]
    if answer == "yes":
        # the body ran on the model's second call, the same call repeated after the answer
        assert len(calls) == 2 and calls[0]["arguments"] == calls[1]["arguments"]
        assert "approved" not in json.loads(calls[1]["arguments"])
        assert [c["extra"]["lk.approval"] for c in calls] == ["asked", "granted"]
        assert len(ran) == 1
    else:
        assert len(calls) == 1 and not ran


# -- 8 the stateless Agent ---------------------------------------------------------------


async def test_08_one_agent_two_sessions_at_once_and_no_mutable_api(dentist: Dentist) -> None:
    from livekit.agents import Agent

    with pytest.raises(TypeError):
        Agent(instructions="x", chat_ctx=object())  # type: ignore[call-arg]
    assert not hasattr(Agent, "update_chat_ctx") and not hasattr(Agent, "update_instructions")
    # the fast brain's module-level front_desk serves two sessions at once with nothing shared:
    # each answers from its own history
    a, b = await asyncio.gather(dentist.send("I am A"), dentist.send("I am B"))
    assert a["contextId"] != b["contextId"]
    ask = "Repeat my first message word for word."
    a2, b2 = await asyncio.gather(
        dentist.send(ask, context_id=a["contextId"]), dentist.send(ask, context_id=b["contextId"])
    )
    said_a, said_b = (artifact_text(a2) or "").lower(), (artifact_text(b2) or "").lower()
    assert "i am a" in said_a and "i am b" in said_b
    assert "i am b" not in said_a and "i am a" not in said_b


# -- 9 the threads service lists a scope ----------------------------------------------------


async def test_09_threads_list_a_scope_newest_first() -> None:
    if not await stack_is_up():
        pytest.skip("agent-db is not reachable")
    from livekit.agents.storage import AgentDBStorage

    threads = AgentDBStorage().threads
    scoped = await threads.create_thread(scope="v0-checks")
    page = await threads.list_threads("v0-checks", limit=30)
    await threads.delete_thread(scoped.thread_id)
    assert page.threads[0].thread_id == scoped.thread_id
    stamps = [t.last_active for t in page.threads]
    assert stamps == sorted(stamps, reverse=True)


# -- 10 the threads service ---------------------------------------------------------------------


async def test_10_threads_create_update_delete() -> None:
    if not await stack_is_up():
        pytest.skip("agent-db is not reachable")
    from livekit.agents.storage import AgentDBStorage, ThreadNotFound

    threads = AgentDBStorage().threads
    created = await threads.create_thread()
    assert created.database_id.startswith("DB_"), "CreateThread creates the database"
    await asyncio.sleep(1.1)
    updated = await threads.update_thread(created.thread_id, attributes={"title": "v0"})
    assert updated.expires_at > created.expires_at, "an update extends expires_at"
    await threads.delete_thread(created.thread_id)
    with pytest.raises(ThreadNotFound):
        await threads.get_thread(created.thread_id)
