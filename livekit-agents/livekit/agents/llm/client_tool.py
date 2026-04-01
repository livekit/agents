# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import TYPE_CHECKING, Any, Optional

from livekit import rtc

from ..log import logger
from .tool_context import RawFunctionTool, RawFunctionToolInfo, ToolError, ToolFlag

if TYPE_CHECKING:
    from ..voice.agent import Agent
    from ..voice.events import RunContext
    from ..voice.room_io.room_io import RoomIO

CLIENT_TOOL_ATTRIBUTE_PREFIX = "lk.client_tools."
CLIENT_TOOL_MANIFEST_VERSION = 1
PARTICIPANT_IDENTITY_PARAM = "lk_participant_identity"

def _generate_default_schema(client_name: str) -> dict[str, Any]:
    return {
        "name": client_name,
        "description": (
            f"Client tool '{client_name}' "
            f"(waiting for client to advertise)"
        ),
        "parameters": {"type": "object", "properties": {}},
    }

def _schemas_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two JSON Schema parameter dicts are equivalent."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class ClientTool(RawFunctionTool):
    """A tool whose implementation is provided by a connected client via RPC.

    The tool's description, parameter schema, and implementation all live on the client.
    The client advertises available tools as participant attributes
    (``lk.client_tools.<name>``), and the agent framework routes execution to the client
    via ``perform_rpc``.

    When multiple participants advertise the same tool with matching schemas, a
    ``lk_participant_identity`` parameter is automatically injected so the LLM can
    choose which participant to route the call to.

    Use the :func:`client_tool` factory function to create instances.
    """

    def __init__(
        self,
        func: Any,
        info: RawFunctionToolInfo,
        *,
        client_name: str,
        instance: Any = None,
    ) -> None:
        super().__init__(func, info, instance)
        self._client_name = client_name
        self._advertisers: list[str] = []  # participant identities
        self._base_manifest: Optional[dict[str, Any]] = None

    @property
    def client_name(self) -> str:
        """The tool name as registered by the client."""
        return self._client_name

    @property
    def resolved(self) -> bool:
        """Whether this tool has been resolved from a client manifest."""
        return len(self._advertisers) > 0

    @property
    def advertisers(self) -> list[str]:
        """Participant identities currently advertising this tool."""
        return list(self._advertisers)

    def _update_schema(self) -> None:
        """Rebuild raw_schema from the base manifest and current advertisers."""
        if not self._base_manifest or not self._advertisers:
            self._info.raw_schema = _generate_default_schema(self._client_name)
            return

        description = self._base_manifest.get("description", "")
        parameters = self._base_manifest.get(
            "parameters", {"type": "object", "properties": {}}
        )

        if len(self._advertisers) > 1:
            parameters = self._inject_participant_param(parameters)

        self._info.raw_schema = {
            "name": self._client_name,
            "description": description,
            "parameters": parameters,
        }

    def _inject_participant_param(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a copy of parameters with lk_participant_identity injected."""
        parameters = json.loads(json.dumps(parameters))  # deep copy
        properties = parameters.setdefault("properties", {})
        properties[PARTICIPANT_IDENTITY_PARAM] = {
            "type": "string",
            "enum": sorted(self._advertisers),
            "description": (
                "The identity of the participant to send the request to."
            ),
        }
        required = parameters.setdefault("required", [])
        if PARTICIPANT_IDENTITY_PARAM not in required:
            required.append(PARTICIPANT_IDENTITY_PARAM)
        return parameters

    def _add_advertiser(
        self, identity: str, manifest: dict[str, Any]
    ) -> None:
        """Add a participant as an advertiser of this tool."""
        params = manifest.get(
            "parameters", {"type": "object", "properties": {}}
        )

        if self._base_manifest is not None:
            base_params = self._base_manifest.get(
                "parameters", {"type": "object", "properties": {}}
            )
            if not _schemas_match(base_params, params):
                logger.warning(
                    f"Client tool '{self._client_name}': participant "
                    f"'{identity}' has a different parameter schema than "
                    f"existing advertisers, skipping this participant. "
                    f"All participants advertising the same client tool "
                    f"must use identical parameter schemas."
                )
                return

        if identity not in self._advertisers:
            self._advertisers.append(identity)

        # Use this manifest as the base if we don't have one yet
        if self._base_manifest is None:
            self._base_manifest = manifest

        self._update_schema()
        logger.debug(
            f"Client tool '{self._client_name}': added advertiser "
            f"'{identity}' (total: {len(self._advertisers)})"
        )

    def _remove_advertiser(self, identity: str) -> None:
        """Remove a participant as an advertiser of this tool."""
        if identity not in self._advertisers:
            return

        self._advertisers.remove(identity)

        if not self._advertisers:
            self._base_manifest = None

        self._update_schema()
        logger.debug(
            f"Client tool '{self._client_name}': removed advertiser "
            f"'{identity}' (total: {len(self._advertisers)})"
        )


def client_tool(name: str) -> ClientTool:
    """Declare a client-fulfilled tool by name.

    The tool's description, parameter schema, and implementation are all provided by a
    connected client via participant attributes (``lk.client_tools.<name>``). The agent
    framework reads these attributes from the linked participant to populate the tool's
    metadata, and routes execution to the client via RPC.

    Args:
        name: The tool name. Must match the tool name registered by the client.
            Case-sensitive, typically camelCase.
    """

    async def _execute(
        raw_arguments: dict[str, object],
        # This is a RunContext, but typed as Any because:
        # 1. voice.events imports from llm, so importing RunContext here creates a circular import
        # 2. get_type_hints() (used by prepare_function_arguments in llm/utils.py) evaluates
        #    annotations at runtime and fails if the type isn't in module globals
        # 3. Using from __future__ import annotations doesn't help since get_type_hints still
        #    tries to resolve the string back to the real type
        context: Any,
    ) -> str:
        if not context.session._agent:
            raise ToolError("Agent state missing from context")

        # Find our ClientTool instance
        tool: Optional[ClientTool] = None
        for t in context.session._agent._tools:
            if isinstance(t, ClientTool) and t.client_name == name:
                tool = t
                break

        if tool is None or not tool.resolved:
            raise ToolError(
                f"Client tool '{name}' has not been resolved from a client "
                f"manifest. The client may not have advertised this tool."
            )

        session = context.session
        room_io = session._room_io
        if room_io is None:
            raise ToolError(
                f"Client tool '{name}' cannot execute: session has no room_io"
            )

        room = room_io.room

        # Determine which participant to route to
        args_to_send = dict(raw_arguments)
        if len(tool.advertisers) > 1:
            destination = args_to_send.pop(PARTICIPANT_IDENTITY_PARAM, None)
            if not isinstance(destination, str):
                raise ToolError(
                    f"Client tool '{name}' requires "
                    f"'{PARTICIPANT_IDENTITY_PARAM}' when multiple "
                    f"participants advertise the tool."
                )
            if destination not in tool.advertisers:
                raise ToolError(
                    f"Client tool '{name}': participant '{destination}' is "
                    f"not currently advertising this tool. "
                    f"Available: {tool.advertisers}"
                )
        elif len(tool.advertisers) == 1:
            destination = tool.advertisers[0]
        else:
            raise ToolError(f"Client tool '{name}' has no participants advertising the tool.")

        payload = json.dumps(args_to_send)
        try:
            response = await room.local_participant.perform_rpc(
                destination_identity=destination,
                method=name,
                payload=payload,
            )
        except Exception as e:
            raise ToolError(f"Client tool '{name}' RPC call failed: {e}") from e

        return response

    # Patch the annotation for `context` so the framework's get_type_hints() resolves
    # it as RunContext (needed for automatic RunContext injection in prepare_function_arguments).
    # We can't use RunContext in the annotation directly due to circular imports.
    from ..voice.events import RunContext as _RunContext

    _execute.__annotations__["context"] = _RunContext

    info = RawFunctionToolInfo(
        name=name,
        raw_schema=_generate_default_schema(name),
        flags=ToolFlag.NONE,
    )

    return ClientTool(_execute, info, client_name=name)


class ClientToolWatcher:
    """Watches participant lifecycle and attributes to keep client tools in sync.

    Scans remote participants for ``lk.client_tools.*`` attributes. When multiple
    participants advertise the same tool with matching schemas, a routing parameter
    is injected. Mismatched schemas produce a warning and the conflicting participant
    is skipped.

    Listens to:
    - ``participant_attributes_changed``: client adds/updates/removes a tool attribute
    - ``participant_connected``: a new participant joins with tool attributes already set
    - ``participant_disconnected``: a participant leaves

    Created and started by ``AgentSession`` when client tools are present.
    """

    def __init__(self, agent: "Agent", room_io: "RoomIO") -> None:
        self._agent = agent
        self._room_io = room_io
        self._room = room_io.room
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._room.on(
            "participant_attributes_changed", self._on_attributes_changed
        )
        self._room.on(
            "participant_connected", self._on_participant_connected
        )
        self._room.on(
            "participant_disconnected", self._on_participant_disconnected
        )

        # Initial scan of all current participants
        for participant in self._room.remote_participants.values():
            self._sync_participant(participant)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False

        self._room.off(
            "participant_attributes_changed", self._on_attributes_changed
        )
        self._room.off(
            "participant_connected", self._on_participant_connected
        )
        self._room.off(
            "participant_disconnected", self._on_participant_disconnected
        )

    def _on_attributes_changed(
        self, changed: list[str], participant: rtc.Participant
    ) -> None:
        if not isinstance(participant, rtc.RemoteParticipant):
            return

        has_client_tool_changes = any(
            k.startswith(CLIENT_TOOL_ATTRIBUTE_PREFIX) for k in changed
        )
        if not has_client_tool_changes:
            return

        self._sync_participant(participant)

    def _on_participant_connected(
        self, participant: rtc.RemoteParticipant
    ) -> None:
        self._sync_participant(participant)

    def _on_participant_disconnected(
        self, participant: rtc.RemoteParticipant
    ) -> None:
        for tool in self._get_client_tools():
            tool._remove_advertiser(participant.identity)

    def _get_client_tools(self) -> list[ClientTool]:
        return [t for t in self._agent._tools if isinstance(t, ClientTool)]

    def _parse_manifest(
        self, tool_name: str, attr_value: str
    ) -> Optional[dict[str, Any]]:
        """Parse and validate a client tool manifest from an attribute value."""
        if not attr_value:
            return None

        try:
            manifest = json.loads(attr_value)
        except json.JSONDecodeError:
            logger.warning(
                f"Client tool '{tool_name}': "
                f"malformed manifest attribute, skipping"
            )
            return None

        version = manifest.get("version")
        if version != CLIENT_TOOL_MANIFEST_VERSION:
            logger.warning(
                f"Client tool '{tool_name}': unsupported manifest version "
                f"{version} (expected {CLIENT_TOOL_MANIFEST_VERSION}), skipping"
            )
            return None

        parameters = manifest.get(
            "parameters", {"type": "object", "properties": {}}
        )
        if parameters.get("type") != "object":
            logger.warning(
                f"Client tool '{tool_name}': parameters schema must be type "
                f"'object', got '{parameters.get('type')}', skipping"
            )
            return None

        return manifest

    def _sync_participant(self, participant: rtc.RemoteParticipant) -> None:
        """Sync client tools for a single participant."""
        for tool in self._get_client_tools():
            attr_key = f"{CLIENT_TOOL_ATTRIBUTE_PREFIX}{tool.client_name}"
            attr_value = participant.attributes.get(attr_key, "")

            manifest = self._parse_manifest(tool.client_name, attr_value)
            if manifest is None:
                # Tool not advertised or invalid — remove this participant
                tool._remove_advertiser(participant.identity)
            else:
                tool._add_advertiser(participant.identity, manifest)

