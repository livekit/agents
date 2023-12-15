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

import asyncio
import json
from livekit import agents
from enum import Enum

UserState = Enum("UserState", "SPEAKING, SILENT")
AgentState = Enum("AgentState", "LISTENING, THINKING, SPEAKING")


class StateManager:
    def __init__(self, ctx: agents.JobContext):
        self._agent_sending_audio = False
        self._chat_gpt_working = False
        self._user_state = UserState.SILENT
        self._ctx = ctx

    async def _send_datachannel_message(self):
        msg = json.dumps(
            {
                "type": "state",
                "user_state": self.user_state.name.lower(),
                "agent_state": self.agent_state.name.lower(),
            }
        )
        await self._ctx.room.local_participant.publish_data(msg)

    @property
    def agent_sending_audio(self):
        return self._agent_sending_audio

    @agent_sending_audio.setter
    def agent_sending_audio(self, value):
        self._agent_sending_audio = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def chat_gpt_working(self):
        return self._chat_gpt_working

    @chat_gpt_working.setter
    def chat_gpt_working(self, value):
        self._chat_gpt_working = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def user_state(self):
        return self._user_state

    @user_state.setter
    def user_state(self, value):
        self._user_state = value
        asyncio.create_task(self._send_datachannel_message())

    @property
    def agent_state(self):
        if self.agent_sending_audio:
            return AgentState.SPEAKING

        if self.chat_gpt_working:
            return AgentState.THINKING

        return AgentState.LISTENING
