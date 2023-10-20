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
import openai
from enum import Enum

MessageRole = Enum('MessageRole', ["system", "user", "assistant", "function"])

@dataclass
class Message:
    role: MessageRole
    content: str

    def toAPI(self):
        return {
            "role": self.role.name,
            "content": self.content
        }


class ChatGPT:
    def __init__(self, prompt: str, message_capacity: int):
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: [Message] = []

    async def push_message(self, message: Message):
        self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)
        
        async for resp in self._generate_text_streamed("gpt-3.5-turbo"):
            yield resp

    async def _generate_text_streamed(self, model: str):
        prompt_message = Message(role=MessageRole.system, content=self._prompt)
        async for chunk in await openai.ChatCompletion.acreate(model=model,
                                                               n=1,
                                                               stream=True,
                                                               messages=[prompt_message.toAPI()] + [m.toAPI() for m in self._messages]):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                yield content