from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ToolError, function_tool
from livekit.agents.llm import FunctionToolCall, ToolContext
from livekit.agents.llm.tool_context import get_fnc_tool_names
from livekit.agents.voice import ClientActionSet
from livekit.agents.voice.client_actions import (
    ACTION_DECLINED_CODE,
    ACTIONS_ATTRIBUTE,
    DESCRIBE_METHOD,
)
from livekit.rtc.event_emitter import EventEmitter

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = pytest.mark.unit

OPEN_DOOR = {
    "name": "open_door",
    "summary": "Open a door.",
    "description": "Open a door. Doors are named front or back.",
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
PLACEHOLDER = {"type": "object", "additionalProperties": True}


@function_tool
async def agent_tool() -> str:
    """The agent's own tool."""
    return "mine"


class _FakeParticipant(rtc.RemoteParticipant):
    """Advertises name + summary in the attribute and answers describe with full entries."""

    def __init__(self, identity: str, actions: list[dict[str, Any]] | str | None = None) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        attributes: dict[str, str] = {}
        if isinstance(actions, str):
            attributes[ACTIONS_ATTRIBUTE] = actions
        elif actions is not None:
            attributes.update(self._catalog(actions))
        self._info = SimpleNamespace(identity=identity, attributes=attributes)

    def _catalog(self, actions: list[dict[str, Any]]) -> dict[str, str]:
        self.entries = {a["name"]: a for a in actions}
        summaries = [
            {"name": a["name"], **({"summary": a["summary"]} if a.get("summary") else {})}
            for a in actions
        ]
        return {ACTIONS_ATTRIBUTE: json.dumps(summaries)}

    def advertise(self, actions: list[dict[str, Any]]) -> dict[str, str]:
        changed = self._catalog(actions)
        self._info.attributes.update(changed)
        return changed


class _FakeLocalParticipant:
    def __init__(self, room: _FakeRoom) -> None:
        self._room = room
        self.calls: list[dict[str, Any]] = []
        self.describes: list[dict[str, Any]] = []
        self.response = "{}"
        self.error: rtc.RpcError | None = None
        self.describe_error: rtc.RpcError | None = None

    async def perform_rpc(self, **kwargs: Any) -> str:
        if kwargs["method"] == DESCRIBE_METHOD:
            self.describes.append(kwargs)
            if self.describe_error is not None:
                raise self.describe_error
            entries = self._room.remote_participants[kwargs["destination_identity"]].entries
            names = json.loads(kwargs["payload"])["names"]
            actions = [
                {k: v for k, v in entries[n].items() if k != "summary"}
                for n in names
                if n in entries
            ]
            return json.dumps({"actions": actions})

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def described(self) -> dict[str, list[str]]:
        """Names requested per participant, in request order."""
        out: dict[str, list[str]] = {}
        for d in self.describes:
            out.setdefault(d["destination_identity"], []).extend(json.loads(d["payload"])["names"])
        return out


class _FakeRoom(EventEmitter[str]):
    def __init__(self, *participants: _FakeParticipant) -> None:
        super().__init__()
        self.remote_participants = {p.identity: p for p in participants}
        self.local_participant = _FakeLocalParticipant(self)


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


def _params(toolset: ClientActionSet, name: str) -> dict[str, Any]:
    return _by_name(toolset)[name].info.raw_schema["parameters"]


async def _wait_for(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met within 2s")


async def _wait_described(toolset: ClientActionSet, *names: str) -> None:
    await _wait_for(
        lambda: all(n in _by_name(toolset) and _params(toolset, n) != PLACEHOLDER for n in names)
    )


async def test_actions_register_at_once_then_fill_in_schemas() -> None:
    room = _FakeRoom(
        _FakeParticipant("a", [{**OPEN_DOOR, "consent": "confirm"}]),
        _FakeParticipant("b", [DIM_LIGHTS]),
    )
    toolset, _, _ = await _start(room)
    try:
        # registered synchronously in setup, before any describe completes
        tools = _by_name(toolset)
        assert list(tools) == ["dim_lights", "open_door"]
        assert tools["open_door"].info.raw_schema["description"] == "Open a door."
        assert tools["open_door"].info.raw_schema["parameters"] == PLACEHOLDER

        await _wait_described(toolset, "dim_lights", "open_door")
        assert room.local_participant.described() == {"a": ["open_door"], "b": ["dim_lights"]}
        schema = _by_name(toolset)["open_door"].info.raw_schema
        assert schema["parameters"] == OPEN_DOOR["parameters"]
        assert schema["description"] == (
            "Open a door. Doors are named front or back."
            " The participant will be asked to confirm before it runs."
        )

        room.local_participant.response = json.dumps({"opened": True})
        result = await _by_name(toolset)["open_door"](raw_arguments={"door": "front"})
        assert result == {"opened": True}
        (call,) = room.local_participant.calls
        assert call["destination_identity"] == "a"
        assert call["method"] == "action:open_door"
        assert json.loads(call["payload"]) == {"door": "front"}

        tool = _by_name(toolset)["dim_lights"]
        room.local_participant.response = "plain text"
        assert await tool(raw_arguments={}) == "plain text"
        room.local_participant.response = ""
        assert await tool(raw_arguments={}) is None
    finally:
        await toolset.aclose()


async def test_descriptions_are_batched_per_participant_and_cached() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR, DIM_LIGHTS]))
    toolset, _, _ = await _start(room)
    try:
        await _wait_described(toolset, "open_door", "dim_lights")
        assert len(room.local_participant.describes) == 1
        assert room.local_participant.described() == {"a": ["open_door", "dim_lights"]}

        # a new action is described on its own; cached ones are not requested again
        a = room.remote_participants["a"]
        changed = a.advertise([OPEN_DOOR, DIM_LIGHTS, {**DIM_LIGHTS, "name": "x"}])
        room.emit("participant_attributes_changed", changed, a)
        await _wait_described(toolset, "x")
        assert room.local_participant.described() == {"a": ["open_door", "dim_lights", "x"]}

        # a changed summary invalidates that action's cached entry
        updated = {**OPEN_DOOR, "summary": "Open any door.", "description": "Open any door."}
        changed = a.advertise([updated, DIM_LIGHTS, {**DIM_LIGHTS, "name": "x"}])
        room.emit("participant_attributes_changed", changed, a)
        await _wait_for(
            lambda: (
                _by_name(toolset)["open_door"].info.raw_schema["description"] == "Open any door."
            )
        )
        assert room.local_participant.described()["a"][-1:] == ["open_door"]
        assert len(room.local_participant.describes) == 3
    finally:
        await toolset.aclose()


async def test_failed_describe_keeps_the_placeholder_callable() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]))
    room.local_participant.describe_error = rtc.RpcError(1500, "boom")
    toolset, _, _ = await _start(room)
    try:
        await _wait_for(lambda: len(room.local_participant.describes) == 1)
        await asyncio.sleep(0.05)
        assert _params(toolset, "open_door") == PLACEHOLDER

        room.local_participant.response = json.dumps("ok")
        assert await _by_name(toolset)["open_door"](raw_arguments={"door": "back"}) == "ok"
        (call,) = room.local_participant.calls
        assert json.loads(call["payload"]) == {"door": "back"}
    finally:
        await toolset.aclose()


async def test_shared_name_routes_by_participant() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]), _FakeParticipant("b", [OPEN_DOOR]))
    toolset, _, _ = await _start(room)
    try:
        await _wait_for(lambda: len(room.local_participant.describes) == 2)
        await _wait_described(toolset, "open_door")
        assert room.local_participant.described() == {"a": ["open_door"], "b": ["open_door"]}

        tool = _by_name(toolset)["open_door"]
        schema = tool.info.raw_schema["parameters"]
        assert schema["properties"]["participant"]["enum"] == ["a", "b"]
        assert schema["required"] == ["door", "participant"]
        assert schema["properties"]["door"] == {"type": "string"}

        await tool(raw_arguments={"door": "back", "participant": "b"})
        (call,) = room.local_participant.calls
        assert call["destination_identity"] == "b"
        assert json.loads(call["payload"]) == {"door": "back"}

        with pytest.raises(ToolError, match="participant must be one of: a, b"):
            await tool(raw_arguments={"door": "back", "participant": "zzz"})
        assert len(room.local_participant.calls) == 1
    finally:
        await toolset.aclose()


async def test_decline_is_relayed_and_other_rpc_errors_raise() -> None:
    room = _FakeRoom(_FakeParticipant("a", [DIM_LIGHTS]))
    toolset, _, _ = await _start(room)
    try:
        await _wait_described(toolset, "dim_lights")
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
        await _wait_described(toolset, "open_door")

        newcomer = _FakeParticipant("b", [DIM_LIGHTS])
        room.remote_participants["b"] = newcomer
        room.emit("participant_connected", newcomer)
        await _wait_described(toolset, "dim_lights")
        assert "agent_tool" in get_fnc_tool_names(agent.tools)

        del room.remote_participants["b"]
        room.emit("participant_disconnected", newcomer)
        await _wait_for(lambda: "dim_lights" not in _by_name(toolset))
        assert list(_by_name(toolset)) == ["open_door"]
        assert "agent_tool" in get_fnc_tool_names(agent.tools)
        assert (toolset in agent.tools) == (scope == "agent")

        # rejoining describes again: the cache does not outlive the catalog
        room.remote_participants["b"] = newcomer
        room.emit("participant_connected", newcomer)
        await _wait_described(toolset, "dim_lights")
        assert room.local_participant.described()["b"] == ["dim_lights", "dim_lights"]
    finally:
        await toolset.aclose()


async def test_attribute_changes_only_wake_for_remote_catalog_updates() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR]))
    toolset, _, _ = await _start(room)
    try:
        await _wait_described(toolset, "open_door")
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
        await _wait_described(toolset, "open_door")
        assert list(_by_name(toolset)) == ["open_door"]
        assert room.local_participant.described() == {"a": ["open_door"]}
        assert get_fnc_tool_names(agent.tools) == ["agent_tool"]
        ctx = ToolContext([*session.tools, *agent.tools])
        assert set(ctx.function_tools) == {"agent_tool", "open_door"}
    finally:
        await toolset.aclose()


async def test_malformed_catalogs_and_descriptions() -> None:
    room = _FakeRoom(
        _FakeParticipant("a", "not json"),
        _FakeParticipant("b", '{"name": "obj"}'),
        _FakeParticipant("c", '[{"summary": "no name"}, {"name": ""}, "junk", 42]'),
        _FakeParticipant("d"),
    )
    toolset, _, _ = await _start(room)
    try:
        await asyncio.sleep(0.05)
        assert toolset.tools == []
        assert room.local_participant.describes == []
    finally:
        await toolset.aclose()

    room = _FakeRoom(_FakeParticipant("a", [{"name": "bad_params", "parameters": "nope"}]))
    toolset, _, _ = await _start(room)
    try:
        await _wait_for(lambda: len(room.local_participant.describes) == 1)
        await asyncio.sleep(0.05)
        assert _params(toolset, "bad_params") == PLACEHOLDER
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


class _RecordingLLM(FakeLLM):
    def __init__(self, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.offered: list[dict[str, Any]] = []

    def chat(self, *, chat_ctx: Any, tools: Any = None, **kwargs: Any) -> FakeLLMStream:
        self.offered.append({t.info.name: t.info.raw_schema["parameters"] for t in tools or []})
        return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)  # type: ignore[return-value]


def _step(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


async def test_every_action_is_callable_on_the_first_turn() -> None:
    room = _FakeRoom(_FakeParticipant("a", [OPEN_DOOR, DIM_LIGHTS]))
    room.local_participant.response = json.dumps("opened")
    open_door = FunctionToolCall(name="open_door", arguments='{"door": "front"}', call_id="o")
    llm = _RecordingLLM(
        [
            _step("open the front door", "", open_door),
            _step("opened", "The front door is open."),
        ]
    )
    toolset = ClientActionSet(room=room)  # type: ignore[arg-type]
    session = AgentSession(llm=llm)
    async with session:
        await session.start(Agent(instructions="x", tools=[toolset]))
        await _wait_described(toolset, "open_door", "dim_lights")
        result = await asyncio.wait_for(session.run(user_input="open the front door"), 5)

    assert llm.offered[0] == {
        "dim_lights": DIM_LIGHTS["parameters"],
        "open_door": OPEN_DOOR["parameters"],
    }
    (call,) = room.local_participant.calls
    assert call["method"] == "action:open_door"
    assert json.loads(call["payload"]) == {"door": "front"}
    result.expect.contains_function_call(name="open_door")
    result.expect.contains_message(role="assistant")
