from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from livekit import rtc

from .. import utils
from ..llm.async_toolset import AsyncToolset
from ..llm.tool_context import (
    RawFunctionTool,
    Tool,
    ToolError,
    Toolset,
    function_tool,
    get_fnc_tool_names,
)
from ..log import logger

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession

ACTIONS_ATTRIBUTE = "lk.actions"
ACTION_METHOD_PREFIX = "action:"
DESCRIBE_METHOD = "lk.actions.describe"
ACTION_DECLINED_CODE = 1710


@dataclass(frozen=True)
class ClientAction:
    name: str
    description: str
    parameters: dict[str, Any]
    consent: Literal["none", "confirm"] = "none"


def _parse_catalog(raw: str, *, identity: str) -> dict[str, str]:
    """Parse an ``lk.actions`` attribute into ``{name: summary}``."""
    try:
        entries = json.loads(raw)
    except ValueError:
        entries = None
    if not isinstance(entries, list):
        logger.warning(
            "ClientActionSet: ignoring malformed lk.actions attribute",
            extra={"participant": identity},
        )
        return {}

    catalog: dict[str, str] = {}
    for entry in entries:
        fields = entry if isinstance(entry, dict) else {}
        name = fields.get("name")
        if not (isinstance(name, str) and name):
            logger.warning(
                "ClientActionSet: skipping malformed catalog entry",
                extra={"participant": identity, "entry": entry},
            )
            continue
        summary = fields.get("summary")
        catalog[name] = summary if isinstance(summary, str) else ""
    return catalog


def _parse_described(raw: str, *, identity: str) -> list[ClientAction]:
    """Parse a ``lk.actions.describe`` response into full actions."""
    try:
        entries = json.loads(raw).get("actions")
    except (ValueError, AttributeError):
        entries = None
    if not isinstance(entries, list):
        logger.warning(
            "ClientActionSet: ignoring malformed describe response",
            extra={"participant": identity},
        )
        return []

    actions: list[ClientAction] = []
    for entry in entries:
        fields = entry if isinstance(entry, dict) else {}
        name = fields.get("name")
        parameters = fields.get("parameters")
        if not (isinstance(name, str) and name and isinstance(parameters, dict)):
            logger.warning(
                "ClientActionSet: skipping malformed action entry",
                extra={"participant": identity, "entry": entry},
            )
            continue
        description = fields.get("description")
        actions.append(
            ClientAction(
                name=name,
                description=description if isinstance(description, str) else "",
                parameters=parameters,
                consent="confirm" if fields.get("consent") == "confirm" else "none",
            )
        )
    return actions


class ClientActionSet(AsyncToolset):
    """Exposes the actions remote participants advertise as tools the LLM can call.

    A participant publishes action names and summaries in the ``lk.actions``
    attribute. Every action becomes a tool right away, first with its summary and an
    open parameter schema; the full entries are then fetched in the background over
    the ``lk.actions.describe`` RPC (one batch per participant) and cached until the
    participant stops advertising the action or changes its summary. Calling a tool
    calls the participant (``action:<name>``). When several participants advertise
    the same name, the tool takes a ``participant`` argument. Names that collide with
    the agent's own tools are skipped. The tool list follows participants joining,
    leaving, and changing their catalog.

    Example::

        session = AgentSession(tools=[ClientActionSet()])  # session-scoped: survives handoff
        agent = Agent(instructions="...", tools=[ClientActionSet()])  # agent-scoped
    """

    def __init__(
        self,
        *,
        id: str = "client_actions",
        room: rtc.Room | None = None,
        response_timeout: float | None = None,
    ) -> None:
        super().__init__(id=id)
        self._room = room
        self._response_timeout = response_timeout
        self._session: AgentSession | None = None
        self._catalogs: dict[str, str] = {}
        self._providers: dict[str, dict[str, str]] = {}  # name -> {identity: summary}
        # (identity, name) -> (summary it was described under, full entry)
        self._cache: dict[tuple[str, str], tuple[str, ClientAction]] = {}
        self._changed = asyncio.Event()
        self._rebind_atask: asyncio.Task[None] | None = None

    def _attach_activity(self, *, activity: AgentActivity | None, session: AgentSession) -> None:
        super()._attach_activity(activity=activity, session=session)
        self._session = session

    async def setup(self) -> Self:
        await super().setup()
        if self._rebind_atask is not None and not self._rebind_atask.done():
            return self

        room = self._room
        if room is None and self._session is not None and self._session._room_io is not None:
            room = self._session._room_io.room
        if room is None:
            logger.warning("ClientActionSet: no room available, client actions disabled")
            return self

        self._room = room
        room.on("participant_attributes_changed", self._on_attributes_changed)
        room.on("participant_connected", self._on_participant_changed)
        room.on("participant_disconnected", self._on_participant_changed)
        self._sync_tools(room)
        self._rebind_atask = asyncio.create_task(self._rebind_loop(room))
        self._changed.set()  # describe the initial catalogs
        return self

    async def aclose(self) -> None:
        if self._room is not None:
            self._room.off("participant_attributes_changed", self._on_attributes_changed)
            self._room.off("participant_connected", self._on_participant_changed)
            self._room.off("participant_disconnected", self._on_participant_changed)
        if self._rebind_atask is not None:
            await utils.aio.cancel_and_wait(self._rebind_atask)
            self._rebind_atask = None
        self._tools = []
        self._catalogs = {}
        self._providers = {}
        self._cache = {}
        await super().aclose()

    def _on_attributes_changed(self, changed: dict[str, str], participant: rtc.Participant) -> None:
        if ACTIONS_ATTRIBUTE in changed and isinstance(participant, rtc.RemoteParticipant):
            self._changed.set()

    def _on_participant_changed(self, participant: rtc.RemoteParticipant) -> None:
        self._changed.set()

    async def _rebind_loop(self, room: rtc.Room) -> None:
        while True:
            await self._changed.wait()
            self._changed.clear()
            try:
                changed = self._sync_tools(room)
                if await self._describe_missing(room):
                    self._rebuild_tools(room)
                    changed = True
                if changed and (agent := self._current_agent()) is not None:
                    await agent.update_tools(agent.tools)
            except Exception:
                logger.exception("ClientActionSet: failed to rebind client actions")

    def _current_agent(self) -> Agent | None:
        return self._session._agent if self._session is not None else None

    def _sync_tools(self, room: rtc.Room) -> bool:
        catalogs = {
            p.identity: p.attributes[ACTIONS_ATTRIBUTE]
            for p in room.remote_participants.values()
            if p.attributes.get(ACTIONS_ATTRIBUTE)
        }
        if catalogs == self._catalogs:
            return False
        self._catalogs = catalogs

        providers: dict[str, dict[str, str]] = {}
        for identity, raw in catalogs.items():
            for name, summary in _parse_catalog(raw, identity=identity).items():
                providers.setdefault(name, {})[identity] = summary

        others: list[Tool | Toolset] = []
        if self._session is not None:
            others.extend(self._session.tools)
        if (agent := self._current_agent()) is not None:
            others.extend(agent.tools)
        # ToolContext rejects duplicate names, so the agent's own tools win
        taken = set(get_fnc_tool_names([t for t in others if t is not self]))

        for name in sorted(providers):
            if name in taken:
                logger.warning(
                    "ClientActionSet: action name collides with an agent tool, skipping",
                    extra={"action": name, "participants": list(providers[name])},
                )
                del providers[name]
        self._providers = providers

        # a changed summary means the participant changed the action, so describe it again
        self._cache = {
            key: cached
            for key, cached in self._cache.items()
            if providers.get(key[1], {}).get(key[0]) == cached[0]
        }
        self._rebuild_tools(room)
        return True

    def _rebuild_tools(self, room: rtc.Room) -> None:
        tools: list[Tool] = []
        for name, by_identity in sorted(self._providers.items()):
            entries = [
                (identity, cached[1] if (cached := self._cache.get((identity, name))) else None)
                for identity in by_identity
            ]
            # the first entry supplies the schema, so prefer a described one
            entries.sort(key=lambda e: e[1] is None)
            tools.append(
                self._bind(
                    room,
                    name,
                    [
                        (identity, action or self._placeholder(name, by_identity[identity]))
                        for identity, action in entries
                    ],
                )
            )
        self._tools = tools

    @staticmethod
    def _placeholder(name: str, summary: str) -> ClientAction:
        return ClientAction(
            name=name,
            description=summary,
            parameters={"type": "object", "additionalProperties": True},
        )

    async def _describe_missing(self, room: rtc.Room) -> bool:
        wanted: dict[str, list[str]] = {}
        for name, by_identity in self._providers.items():
            for identity in by_identity:
                if (identity, name) not in self._cache:
                    wanted.setdefault(identity, []).append(name)
        if not wanted:
            return False

        identities = list(wanted)
        results = await asyncio.gather(
            *(self._describe(room, i, wanted[i]) for i in identities), return_exceptions=True
        )

        loaded = False
        for identity, result in zip(identities, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "ClientActionSet: failed to describe actions",
                    extra={"participant": identity, "error": str(result)},
                )
                continue
            for action in result:
                # the catalog may have changed while the request was in flight
                summary = self._providers.get(action.name, {}).get(identity)
                if summary is not None:
                    self._cache[(identity, action.name)] = (summary, action)
                    loaded = True
        return loaded

    async def _describe(
        self, room: rtc.Room, identity: str, names: list[str]
    ) -> list[ClientAction]:
        resp = await room.local_participant.perform_rpc(
            destination_identity=identity,
            method=DESCRIBE_METHOD,
            payload=json.dumps({"names": names}),
            response_timeout=self._response_timeout,
        )
        return [a for a in _parse_described(resp, identity=identity) if a.name in names]

    def _bind(
        self, room: rtc.Room, name: str, entries: list[tuple[str, ClientAction]]
    ) -> RawFunctionTool:
        identity, action = entries[0]
        description = action.description
        if action.consent == "confirm":
            description += " The participant will be asked to confirm before it runs."

        identities = [i for i, _ in entries]
        multi = len(entries) > 1
        parameters = action.parameters
        if multi:
            parameters = {
                **parameters,
                "properties": {
                    **parameters.get("properties", {}),
                    "participant": {
                        "type": "string",
                        "enum": identities,
                        "description": "Identity of the participant that should perform this action.",
                    },
                },
                "required": [*parameters.get("required", []), "participant"],
            }

        async def _tool_called(raw_arguments: dict[str, Any]) -> Any:
            args = dict(raw_arguments)
            target = identity
            if multi:
                target = args.pop("participant", None)
                if target not in identities:
                    raise ToolError(f"participant must be one of: {', '.join(identities)}")
            return await self._perform(room, target, name, args)

        return function_tool(
            _tool_called,
            raw_schema={"name": name, "description": description, "parameters": parameters},
        )

    async def _perform(self, room: rtc.Room, identity: str, name: str, args: dict[str, Any]) -> Any:
        try:
            resp = await room.local_participant.perform_rpc(
                destination_identity=identity,
                method=f"{ACTION_METHOD_PREFIX}{name}",
                payload=json.dumps(args),
                response_timeout=self._response_timeout,
            )
        except rtc.RpcError as e:
            if e.code == ACTION_DECLINED_CODE:
                reason = f" They said: {e.message}" if e.message else ""
                return f"The participant declined this action.{reason}"
            raise ToolError(f"Action {name} failed: {e.message}") from e

        if not resp:
            return None
        try:
            return json.loads(resp)
        except ValueError:
            return resp
