# Copyright 2023 LiveKit, Inc.
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

from dataclasses import dataclass
from datetime import datetime, field
from enum import Enum
import json
from typing import Callable, List, Union
import uuid

from livekit import api, rtc
from .job_context import JobContext

AgentStatePreset = Enum("AgentStatePreset", "IDLE, LISTENING, THINKING, SPEAKING")
AgentState = Union[AgentStatePreset, str]

_CHAT_TOPIC = "lk-chat-topic"
_CHAT_UPDATE_TOPIC = "lk-chat-update-topic"


class DataTransport:
    """A utility class that helps with sending data to other participants in the Room.

    It uses Participant Metadata and DataChannel messages to send updates to the client.
    Designed to work with Agent Playground and LiveKit Components.
    """

    def __init__(self, ctx: JobContext):
        self._api = ctx.api
        self._lp = ctx.room.local_participant
        self._room_name = ctx.room.name
        self._agent_state: AgentState = AgentStatePreset.IDLE
        self._callback: Callable[["ChatMessage"], None] = None

        self._lp.on("data_received", self._on_data_received)

    @property
    def agent_state(self) -> AgentState:
        return self._agent_state

    async def set_agent_state(self, state: AgentState):
        """Set the agent state, and send to all participants via Participant metadata.

        Args:
            state (AgentState): the new state, either a AgentStatePreset enum or a str
        """
        self._agent_state = state
        await self._update_metadata()

    async def _update_metadata(self):
        s = self._agent_state
        if isinstance(s, AgentStatePreset):
            s = s.name.lower()
        metadata = {"agent_state": s}
        await self._api.room.update_participant(
            api.UpdateParticipantRequest(
                room=self.room_name,
                identity=self._lp.identity,
                metadata=json.dumps(metadata),
            )
        )

    async def send_chat_message(
        self, message: str, asset_urls: List[str] = []
    ) -> "ChatMessage":
        """Send a chat message to the end user using LiveKit Chat Protocol.

        Args:
            message (str): the message to send

        Returns:
            ChatMessage: the message that was sent
        """
        msg = ChatMessage(
            message=message,
            asset_urls=asset_urls,
        )
        await self._lp.publish_data(
            payload=json.dumps(msg.asdict()),
            kind=rtc.DataPacketKind.KIND_RELIABLE,
            topic=_CHAT_TOPIC,
        )
        return msg

    async def update_chat_message(self, message: "ChatMessage"):
        """Update a chat message that was previously sent.

        If message.deleted is set to True, we'll signal to remote participants that the message
        should be deleted.
        """
        await self._lp.publish_data(
            payload=json.dumps(message.asdict()),
            kind=rtc.DataPacketKind.KIND_RELIABLE,
            topic=_CHAT_UPDATE_TOPIC,
        )

    def on_chat_message(self, callback: Callable[["ChatMessage"], None]):
        """Register a callback to be called when a chat message is received from the end user."""
        self._callback = callback

    def _on_data_received(self, dp: rtc.DataPacket):
        # handle both new and updates the same way, as long as the ID is in there
        # the user can decide how to replace the previous message
        if dp.topic == _CHAT_TOPIC or dp.topic == _CHAT_UPDATE_TOPIC:
            msg = ChatMessage(**json.loads(dp.payload))
            if self._callback:
                self._callback(msg)


@dataclass
class ChatMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    asset_urls: List[str]
    deleted: bool = field(default=False)

    def asdict(self):
        d = {
            "id": self.id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.asset_urls:
            d["asset_urls"] = self.asset_urls
        if self.deleted:
            d["deleted"] = True
        return d
