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
import logging
import asyncio
from dataclasses import dataclass
from typing import AsyncIterable
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


class ChatGPTPlugin:
    def __init__(self, prompt: str, message_capacity: int):
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: [ChatGPTMessage] = []
        self._producing_response = False
        self._needs_interrupt = False

    def interrupt(self):
        if self._producing_response:
            self._needs_interrupt = True

    async def close(self):
        pass

    async def add_message(self, message: ChatGPTMessage) -> AsyncIterable[str]:
        self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)
        
        async for text in self._generate_text_streamed('gpt-3.5-turbo'):
            yield text

    async def _generate_text_streamed(self, model: str) -> AsyncIterable[str]:
        prompt_message = ChatGPTMessage(
            role=ChatGPTMessageRole.system, content=self._prompt)
        try:
            chat_stream = await asyncio.wait_for(self._client.chat.completions.create(model=model,
                                                                                      n=1,
                                                                                      stream=True,
                                                                                      messages=[prompt_message.to_api()] + [m.to_api() for m in self._messages]), 10)
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        while True:
            try:
                chunk = await asyncio.wait_for(anext(chat_stream, None), 5)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break

            if chunk is None:
                break
            content = chunk.choices[0].delta.content

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT Interrupted")
                break

            if content is not None:
                yield content

        self._producing_response = False
