from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ToolError, function_tool
from livekit.agents.llm import ToolContext
from livekit.agents.llm.tool_context import get_fnc_tool_names
from livekit.agents.voice import ClientActionSet
from livekit.agents.voice.client_actions import ACTION_DECLINED_CODE, ACTIONS_ATTRIBUTE
from livekit.rtc.event_emitter import EventEmitter

pytestmark = pytest.mark.unit

OPEN_DOOR = {
    "name": "open_door",
    "description": "Open a door.",
    "parameters": {
        "type": "object",
        "properties": {"door": {"type": "string"}},
        "required": ["door"],
    },
}
DIM_LIGHTS = {
    "name": "dim_lights",
    "parameters": {"type": "object", "properties": {}},
}


@function_tool
async def agent_tool() -> str:
    """The agent's own tool."""
    return "mine"


class _FakeParticipant(rtc.RemoteParticipant):
    def __init__(self, identity: str, actions: list[dict[str, Any]] | str | None = None) -> None:
        attributes: dict[str, str] = {}
        if actions is not None:
            attributes[ACTIONS_ATTRIBUTE] = (
                actions if isinstance(actions, str) else json.dumps(actions)
            )
        self._info = SimpleNamespace(identity=identity, attributes=attributes)

    def advertise(self, actions: list[dict[str, Any]]) -> dict[str, str]:
        changed = {ACTIONS_ATTRIBUTE: json.dumps(actions)}
        self._info.attributes.update(changed)
        return changed


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = "{}"
        self.error: rtc.RpcError | None = None

    async def perform_rpc(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeRoom(EventEmitter[str]):
    def __init__(self, *participants: _FakeParticipant) -> None:
        super().__init__()
        self.remote_participants = {p.identity: p for p in participants}
        self.local_participant = _FakeLocalParticipant()


async def _start(
    room: _FakeRoom, *, scope: str = "session", agent_tools: list[Any] | None = None
) -> tuple[ClientActionSet, Agent, AgentSession]:
    toolset = ClientActionSet()
    own = list(agent_tools or [])
    session = AgentSession(tools=[toolset] if scope == "session" else [])
    agent = Agent(instructions="x", tools=[*own, toolset] if scope == "agent" else own)
    session._agent = agent
    session._room_io = SimpleNamespace(room=room)
    toolset._attach_activity(activity=None, session=session)
    await toolset.setup()
    return toolset, agent, session


def _by_name(toolset: ClientActionSet) -> dict[str, Any]:
    return {t.info.name: t for t in toolset.tools}


async def _wait_for(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met within 2s")


async def test_one_tool_per_action_calls_that_participant() -> None:
    room = _FakeRoom(
        _FakeParticipant("a", [{**OPEN_DOOR, "consent": "confirm"}]),
        _FakeParticipant("b", [DIM_LIGHTS]),
    )
    toolset, _, _ = await _start(room)
    try:
        tools = _by_name(toolset)
        assert list(tools) == ["dim_lights", "open_door"]
        schema = tools["open_door"].info.raw_schema
        assert schema["parameters"] == OPEN_DOOR["parameters"]
        assert schema["description"].endswith("asked to confirm before it runs.")
        assert tools["dim_lights"].info.raw_schema["description"] == ""

        room.local_participant.response = json.dumps({"opened": True})
        result = await tools["open_door"](raw_arguments={"door": "front"})
        assert result == {"opened": True}
        (call,) = room.local_participant.calls
        assert call["destination_identity"] == "a"
        assert call["method"] == "action:open_door"
        assert json.loads(call["payload"]) == {"door": "front"}

        room.local_participant.response = "plain text"
        assert await tools["dim_lights"](raw_arguments={}) == "plain text"
        room.local_participant.response = ""
        assert await tools["dim_lights"](raw_arguments={}) is None
    finally:
        await toolset.aclose()


async def test_shared_name_routes_by_participant() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]), _FakeParticipant("b", [OPEN_DOOR]))
    toolset, _, _ = await _start(room)
    try:
        tools = _by_name(toolset)
        assert list(tools) == ["open_door"]
        schema = tools["open_door"].info.raw_schema["parameters"]
        assert schema["properties"]["participant"]["enum"] == ["a", "b"]
        assert schema["required"] == ["door", "participant"]
        assert schema["properties"]["door"] == {"type": "string"}

        await tools["open_door"](raw_arguments={"door": "back", "participant": "b"})
        (call,) = room.local_participant.calls
        assert call["destination_identity"] == "b"
        assert json.loads(call["payload"]) == {"door": "back"}

        with pytest.raises(ToolError, match="participant must be one of: a, b"):
            await tools["open_door"](raw_arguments={"door": "back", "participant": "zzz"})
        assert len(room.local_participant.calls) == 1
    finally:
        await toolset.aclose()


async def test_decline_is_relayed_and_other_rpc_errors_raise() -> None:
    room = _FakeRoom(_FakeParticipant("a", [DIM_LIGHTS]))
    toolset, _, _ = await _start(room)
    try:
        tool = _by_name(toolset)["dim_lights"]

        room.local_participant.error = rtc.RpcError(ACTION_DECLINED_CODE, "not now")
        result = await tool(raw_arguments={})
        assert result == "The participant declined this action. They said: not now"

        room.local_participant.error = rtc.RpcError(ACTION_DECLINED_CODE, "")
        assert await tool(raw_arguments={}) == "The participant declined this action."

        room.local_participant.error = rtc.RpcError(1500, "boom")
        with pytest.raises(ToolError, match="Action dim_lights failed: boom"):
            await tool(raw_arguments={})
    finally:
        await toolset.aclose()


@pytest.mark.parametrize("scope", ["session", "agent"])
async def test_rebinds_as_participants_come_and_go(scope: str) -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]))
    toolset, agent, _ = await _start(room, scope=scope, agent_tools=[agent_tool])
    try:
        assert list(_by_name(toolset)) == ["open_door"]

        newcomer = _FakeParticipant("b", [DIM_LIGHTS])
        room.remote_participants["b"] = newcomer
        room.emit("participant_connected", newcomer)
        await _wait_for(lambda: "dim_lights" in _by_name(toolset))
        assert "agent_tool" in get_fnc_tool_names(agent.tools)

        changed = room.remote_participants["a"].advertise([OPEN_DOOR, {**DIM_LIGHTS, "name": "x"}])
        room.emit("participant_attributes_changed", changed, room.remote_participants["a"])
        await _wait_for(lambda: "x" in _by_name(toolset))

        del room.remote_participants["b"]
        room.emit("participant_disconnected", newcomer)
        await _wait_for(lambda: "dim_lights" not in _by_name(toolset))
        assert list(_by_name(toolset)) == ["open_door", "x"]
        assert "agent_tool" in get_fnc_tool_names(agent.tools)
        assert (toolset in agent.tools) == (scope == "agent")
    finally:
        await toolset.aclose()


async def test_attribute_changes_only_wake_for_remote_catalog_updates() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]))
    toolset, _, _ = await _start(room)
    try:
        room.emit("participant_attributes_changed", {"other": "1"}, room.remote_participants["a"])
        assert not toolset._changed.is_set()

        local = SimpleNamespace(identity="local", attributes={ACTIONS_ATTRIBUTE: "[]"})
        room.emit("participant_attributes_changed", {ACTIONS_ATTRIBUTE: "[]"}, local)
        assert not toolset._changed.is_set()

        changed = room.remote_participants["a"].advertise([DIM_LIGHTS])
        room.emit("participant_attributes_changed", changed, room.remote_participants["a"])
        await _wait_for(lambda: list(_by_name(toolset)) == ["dim_lights"])
    finally:
        await toolset.aclose()


async def test_action_named_like_an_agent_tool_is_skipped() -> None:
    room = _FakeRoom(_FakeParticipant("a", [{**DIM_LIGHTS, "name": "agent_tool"}, OPEN_DOOR]))
    toolset, agent, session = await _start(room, agent_tools=[agent_tool])
    try:
        assert list(_by_name(toolset)) == ["open_door"]
        assert get_fnc_tool_names(agent.tools) == ["agent_tool"]
        ctx = ToolContext([*session.tools, *agent.tools])
        assert set(ctx.function_tools) == {"agent_tool", "open_door"}
    finally:
        await toolset.aclose()


async def test_malformed_catalogs_yield_no_tools() -> None:
    room = _FakeRoom(
        _FakeParticipant("a", "not json"),
        _FakeParticipant("b", '{"name": "obj"}'),
        _FakeParticipant(
            "c",
            [
                {"description": "no name", "parameters": {}},
                {"name": "", "parameters": {}},
                {"name": "bad_params", "parameters": "nope"},
                "junk",
                42,
            ],
        ),
        _FakeParticipant("d"),
    )
    toolset, _, _ = await _start(room)
    try:
        assert toolset.tools == []
    finally:
        await toolset.aclose()


async def test_aclose_unsubscribes_and_is_safe_to_repeat() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]))
    toolset, _, _ = await _start(room)
    assert list(_by_name(toolset)) == ["open_door"]

    await toolset.aclose()
    assert toolset.tools == []
    assert toolset._rebind_atask is None

    newcomer = _FakeParticipant("b", [DIM_LIGHTS])
    room.remote_participants["b"] = newcomer
    room.emit("participant_connected", newcomer)
    await asyncio.sleep(0.05)
    assert toolset.tools == []
    await toolset.aclose()

    never_setup = ClientActionSet()
    await never_setup.aclose()

    no_room = ClientActionSet()
    session = AgentSession(tools=[no_room])
    no_room._attach_activity(activity=None, session=session)
    await no_room.setup()
    assert no_room.tools == []
    assert no_room._rebind_atask is None
    await no_room.aclose()
