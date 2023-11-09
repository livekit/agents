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

import os
from dataclasses import dataclass
from typing import AsyncIterator
from livekit.plugins import core
import openai
from enum import Enum

ChatGPTMessageRole = Enum(
    'MessageRole', ["system", "user", "assistant", "function"])


@dataclass
class ChatGPTMessage:
    role: ChatGPTMessageRole
    content: str

    def toAPI(self):
        return {
            "role": self.role.name,
            "content": self.content
        }


class ChatGPT:
    def __init__(self, prompt: str, message_capacity: int):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: [ChatGPTMessage] = []

    async def push_message(self, message: ChatGPTMessage) -> AsyncIterator[str]:
        self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)

        return self._generate_text_streamed("gpt-3.5-turbo")

    async def _generate_text_streamed(self, model: str) -> AsyncIterator[str]:
        prompt_message = ChatGPTMessage(
            role=ChatGPTMessageRole.system, content=self._prompt)
        async for chunk in await openai.ChatCompletion.acreate(model=model,
                                                               n=1,
                                                               stream=True,
                                                               messages=[prompt_message.toAPI()] + [m.toAPI() for m in self._messages]):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                yield content


class ChatGPTPlugin(core.Plugin):
    def __init__(self, *, prompt: str, message_capacity: int):
        self._chatgpt = ChatGPT(
            prompt=prompt, message_capacity=message_capacity)

        super().__init__(process=self._chatgpt.push_message)
