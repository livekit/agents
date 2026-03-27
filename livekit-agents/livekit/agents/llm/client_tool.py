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

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from livekit import rtc

from ..log import logger
from .tool_context import RawFunctionTool, RawFunctionToolInfo, ToolError, ToolFlag

if TYPE_CHECKING:
    from ..voice.agent import Agent
    from ..voice.events import RunContext
    from ..voice.room_io.room_io import RoomIO

CLIENT_TOOL_ATTRIBUTE_PREFIX = "lk.client_tools."
CLIENT_TOOL_MANIFEST_VERSION = 1


class ClientTool(RawFunctionTool):
    """A tool whose implementation is provided by a connected client via RPC.

    The tool's description, parameter schema, and implementation all live on the client.
    The client advertises available tools as participant attributes
    (``lk.client_tools.<name>``), and the agent framework routes execution to the client
    via ``perform_rpc``.

    Use the :func:`client_tool` factory function to create instances.
    """

    def __init__(
        self,
        func: Any,
        info: RawFunctionToolInfo,
        *,
        client_name: str,
        required: bool,
        instance: Any = None,
    ) -> None:
        super().__init__(func, info, instance)
        self._client_name = client_name
        self._required = required
        self._resolved = False

    @property
    def client_name(self) -> str:
        """The tool name as registered by the client."""
        return self._client_name

    @property
    def required(self) -> bool:
        """Whether this client tool must be advertised by the client."""
        return self._required

    @property
    def resolved(self) -> bool:
        """Whether this tool has been resolved from a client manifest."""
        return self._resolved

    def _resolve(self, manifest: dict[str, Any]) -> None:
        """Update the tool's raw_schema from a valid manifest and mark it resolved."""
        description = manifest.get("description", "")
        parameters = manifest.get(
            "parameters", {"type": "object", "properties": {}}
        )
        self._info.raw_schema = {
            "name": self._client_name,
            "description": description,
            "parameters": parameters,
        }
        self._resolved = True
        logger.debug(f"Resolved client tool '{self._client_name}' from manifest")

    def _unresolve(self) -> None:
        """Mark the tool as unresolved, resetting its schema to the placeholder."""
        if not self._resolved:
            return
        self._info.raw_schema = {
            "name": self._client_name,
            "description": (
                f"Client tool '{self._client_name}' "
                f"(waiting for client to advertise)"
            ),
            "parameters": {"type": "object", "properties": {}},
        }
        self._resolved = False
        logger.debug(f"Unresolved client tool '{self._client_name}'")


def client_tool(
    name: str,
    *,
    required: bool = False,
) -> ClientTool:
    """Declare a client-fulfilled tool by name.

    The tool's description, parameter schema, and implementation are all provided by a
    connected client via participant attributes (``lk.client_tools.<name>``). The agent
    framework reads these attributes from the linked participant to populate the tool's
    metadata, and routes execution to the client via RPC.

    Args:
        name: The tool name. Must match the tool name registered by the client.
            Case-sensitive, typically camelCase.
        required: If ``True``, the framework logs a warning when the linked participant is
            connected but hasn't advertised this tool, and returns a ``ToolError`` to the
            LLM if it tries to call the missing tool. If ``False`` (default), the tool is
            silently unavailable until the client advertises it. Use ``required=True`` for
            tools that your agent's workflow depends on. Use ``required=False`` (default)
            for tools that may be registered dynamically later or that only some clients
            implement.
    """

    async def _execute(
        raw_arguments: dict[str, object], context: RunContext
    ) -> str:
        if not context.session._agent:
            raise ToolError("Agent state missing from context")

        # Find our ClientTool instance to check the resolved flag
        tool: ClientTool | None = None
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
        linked = room_io.linked_participant
        if linked is None:
            raise ToolError(
                f"Client tool '{name}' cannot execute: no linked participant"
            )

        payload = json.dumps(raw_arguments)
        try:
            response = await room.local_participant.perform_rpc(
                destination_identity=linked.identity,
                method=name,
                payload=payload,
            )
        except Exception as e:
            raise ToolError(f"Client tool '{name}' RPC call failed: {e}") from e

        return response

    raw_schema: dict[str, Any] = {
        "name": name,
        "description": (
            f"Client tool '{name}' (waiting for client to advertise)"
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }

    info = RawFunctionToolInfo(
        name=name,
        raw_schema=raw_schema,
        flags=ToolFlag.NONE,
    )

    return ClientTool(_execute, info, client_name=name, required=required)


class ClientToolWatcher:
    """Watches linked participant lifecycle and attributes to keep client tools in sync.

    Listens to:
    - ``participant_attributes_changed``: client adds/updates/removes a tool attribute
    - ``participant_connected``: a new participant joins with tool attributes already set
    - ``participant_disconnected``: the linked participant leaves, all tools become unresolved

    Created and started by ``AgentSession`` when client tools are present.
    """

    def __init__(self, agent: Agent, room_io: RoomIO) -> None:
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

        # Initial scan if linked participant is already present
        linked = self._room_io.linked_participant
        if linked is not None:
            self._sync_from_participant(linked)

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
        linked = self._room_io.linked_participant
        if linked is None or participant.identity != linked.identity:
            return

        has_client_tool_changes = any(
            k.startswith(CLIENT_TOOL_ATTRIBUTE_PREFIX) for k in changed
        )
        if not has_client_tool_changes:
            return

        self._sync_from_participant(linked)

    def _on_participant_connected(
        self, participant: rtc.RemoteParticipant
    ) -> None:
        linked = self._room_io.linked_participant
        if linked is None or participant.identity != linked.identity:
            return

        self._sync_from_participant(participant)

    def _on_participant_disconnected(
        self, participant: rtc.RemoteParticipant
    ) -> None:
        linked = self._room_io.linked_participant
        if linked is not None and participant.identity != linked.identity:
            return

        # Linked participant left — unresolve all client tools
        for tool in self._get_client_tools():
            tool._unresolve()

    def _get_client_tools(self) -> list[ClientTool]:
        return [t for t in self._agent._tools if isinstance(t, ClientTool)]

    def _parse_manifest(
        self, tool_name: str, attr_value: str
    ) -> dict[str, Any] | None:
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

    def _sync_from_participant(self, participant: rtc.Participant) -> None:
        """Sync all client tools from the participant's current attributes."""
        client_tools = self._get_client_tools()
        if not client_tools:
            return

        for tool in client_tools:
            attr_key = f"{CLIENT_TOOL_ATTRIBUTE_PREFIX}{tool.client_name}"
            attr_value = participant.attributes.get(attr_key, "")

            manifest = self._parse_manifest(tool.client_name, attr_value)
            if manifest is None:
                # Tool not advertised or invalid — unresolve if it was resolved
                tool._unresolve()
                if attr_value == "" and tool.required:
                    logger.warning(
                        f"Required client tool '{tool.client_name}' is not "
                        f"advertised by the linked participant. "
                        f"Ensure the client registers this tool."
                    )
                continue

            tool._resolve(manifest)
