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

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import Any, Callable, Dict, Optional
import uuid

from livekit import rtc
from .job_context import JobContext

_CHAT_TOPIC = "lk-chat-topic"
_CHAT_UPDATE_TOPIC = "lk-chat-update-topic"


class DataHelper:
    """A utility class that helps with sending data to other participants in the Room.

    It uses Participant Metadata and DataChannel messages to send updates to the client.
    Designed to work with Agent Playground and LiveKit Components.
    """

    def __init__(self, ctx: JobContext):
        self._lp = ctx.room.local_participant
        self._callback: Callable[["ChatMessage"], None] = None
        self._metadata: Dict[str, Any] = {}

        ctx.room.on("data_received", self._on_data_received)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    async def set_metadata(self, **kwargs):
        """Set metadata for the local participant.

        Args:
            **kwargs: key-value pairs to set as metadata. It will update only the keys provided,
                and leave the rest as is. To clear a key, set it to None.
        """
        for k, v in kwargs.items():
            if v is None:
                self._metadata.pop(k, None)
            else:
                self._metadata[k] = v
        await self._lp.update_metadata(json.dumps(self._metadata))

    async def send_chat_message(self, message: str) -> "ChatMessage":
        """Send a chat message to the end user using LiveKit Chat Protocol.

        Args:
            message (str): the message to send

        Returns:
            ChatMessage: the message that was sent
        """
        msg = ChatMessage(
            message=message,
            is_local=True,
            participant=self._lp,
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
            try:
                parsed = json.loads(dp.data)
                msg = ChatMessage(**parsed)
                if dp.participant:
                    msg.participant = dp.participant
                if self._callback:
                    self._callback(msg)
            except Exception as e:
                logging.warning("failed to parse chat message: %s", e)


@dataclass
class ChatMessage:
    message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )
    deleted: bool = field(default=False)

    # these fields are not serialized
    # participant field is set on received messages
    participant: Optional[rtc.Participant] = None
    is_local: bool = field(default=False)

    def asdict(self):
        d = {
            "id": self.id,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        if self.deleted:
            d["deleted"] = True
        return d
