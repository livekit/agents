"""An ``AgentHandler`` a module defines with no server, served by ``server.add()``; one with
``lane=`` writes its own lane of the thread (docs/lanes.md)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import AgentHandler, AgentServer
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage
from livekit.agents.worker import rtc_session

from .test_a2a_delegation import lanes_of
from .test_a2a_server import (
    AGENT,
    API_KEY,
    API_SECRET,
    CREDENTIALS,
    http_client,
    sse_results,
    wire,
)

pytestmark = [pytest.mark.unit]


async def echo(ctx: A2ASessionContext[Any]) -> str:
    return f"heard {ctx.message.text_content}"


async def stream(http: Any, path: str, message: dict[str, Any]) -> list[dict[str, Any]]:
    body = {
        "jsonrpc": "2.0",
        "id": "message/stream",
        "method": "message/stream",
        "params": {"message": message},
    }
    return [r["result"] for r in sse_results(await http.post(path, json=body))]


async def test_a_handler_with_a_lane_of_its_own(agentdb_storage: AgentDBStorage) -> None:
    front_desk = AgentHandler()
    front_desk.a2a_session(endpoint="frontend", description="the front desk")(echo)
    back_office = AgentHandler(lane="back-office")
    back_office.a2a_session(endpoint="backend", description="the back office", public=False)(echo)
    front = AgentServer(
        storage=agentdb_storage, agent_name=AGENT, api_key=API_KEY, api_secret=API_SECRET
    )
    front.add(front_desk, back_office)
    assert set(front.a2a_endpoints) == {"frontend", "backend"}

    # the back office's door is not public: the project's token opens it, as a delegate's does
    async with http_client(front, Authorization=f"Bearer {CREDENTIALS.project_token()}") as http:
        first = await stream(http, "/frontend/v1", wire("hi", "m1"))
        thread_id = first[0]["task"]["contextId"]
        second = await stream(http, "/backend/v1", wire("hello", "m2", context_id=thread_id))
    assert second[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    # the same thread, two lanes: the worker's for one door, the other door's own
    assert await lanes_of(agentdb_storage, thread_id) == {AGENT, "back-office"}


@pytest.mark.parametrize(
    ("inbox", "holds"),
    [(None, True), ("live", True), ("queue", False), ({"sms": "batch", "a2a": "interrupt"}, False)],
)
def test_the_calls_mode_follows_the_text_door_of_its_lane(inbox: Any, holds: bool) -> None:
    """A call holds its lane for the whole call when the person's text door on that lane is
    ``live``, or when there is none; otherwise it takes the lane per voice turn. A door on
    another lane does not count."""

    async def call(ctx: Any) -> None: ...

    front_desk = AgentHandler()
    front_desk.rtc_session(call)
    front_desk.a2a_session(endpoint="frontend", description="the front desk", inbox=inbox)(echo)
    back_office = AgentHandler(lane="back-office")
    back_office.a2a_session(endpoint="backend", description="the back office", inbox="queue")(echo)
    server = AgentServer(agent_name=AGENT, storage=MagicMock(spec=AgentDBStorage))
    server.add(front_desk, back_office)
    assert server._call_holds_lane is holds

    alone = AgentServer(agent_name=AGENT)
    alone.add(rtc_session()(call))
    assert alone._call_holds_lane is True, "no text door: the call holds the lane"
