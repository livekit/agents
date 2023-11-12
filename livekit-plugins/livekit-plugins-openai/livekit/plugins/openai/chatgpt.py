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
from typing import AsyncIterable
from livekit.plugins import core
from openai import AsyncOpenAI
from enum import Enum

ChatGPTMessageRole = Enum(
    'MessageRole', ["system", "user", "assistant", "function"])


@dataclass
class ChatGPTMessage:
    role: ChatGPTMessageRole
    content: str

    def to_api(self):
        return {
            "role": self.role.name,
            "content": self.content
        }


class ChatGPTPlugin(core.Plugin[ChatGPTMessage, AsyncIterable[str]]):
    def __init__(self, prompt: str, message_capacity: int):
        super().__init__(process=self._process, close=self._close)
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: [ChatGPTMessage] = []
        self._producing_response = False
        self._needs_interrupt = False

    def interrupt(self):
        if self._producing_response:
            self._needs_interrupt = True

    async def _close(self):
        pass

    def _process(self, message_iterator: AsyncIterable[ChatGPTMessage]) -> AsyncIterable[AsyncIterable[str]]:
        async def iterator():
            async for msg in message_iterator:
                self._messages.append(msg)
                if len(self._messages) > self._message_capacity:
                    self._messages.pop(0)

                yield self._generate_text_streamed('gpt-3.5-turbo')

        return iterator()

    async def _generate_text_streamed(self, model: str) -> AsyncIterable[str]:
        prompt_message = ChatGPTMessage(
            role=ChatGPTMessageRole.system, content=self._prompt)
        self._producing_response = True
        async for chunk in await self._client.chat.completions.create(model=model,
                                                                      n=1,
                                                                      stream=True,
                                                                      messages=[prompt_message.to_api()] + [m.to_api() for m in self._messages]):
            content = chunk.choices[0].delta.content

            if self._needs_interrupt:
                self._needs_interrupt = False
                print("interrupted")
                break

            if content is not None:
                yield content

        self._producing_response = False 
