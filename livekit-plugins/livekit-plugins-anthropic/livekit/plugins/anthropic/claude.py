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
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from enum import Enum

ClaudeMessageRole = Enum("MessageRole", ["system", "human", "assistant"])


@dataclass
class ClaudeMessage:
    role: ClaudeMessageRole
    content: str

    def to_api(self):
        if ClaudeMessageRole.system == self.role:
            return f"{self.content}"
        elif ClaudeMessageRole.human == self.role:
            return f"{HUMAN_PROMPT} {self.content}"
        elif ClaudeMessageRole.assistant == self.role:
            return f"{AI_PROMPT} {self.content}"
        else:
            raise ValueError("Invalid message role")


class Claude:
    def __init__(self, model: str = "claude-2", system_message: str = ""):
        self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._system_message = system_message
        self._messages: [ClaudeMessage] = []
        self._producing_response = False
        self._needs_interrupt = False

    def interrupt(self):
        if self._producing_response:
            self._needs_interrupt = True

    async def close(self):
        pass

    async def add_message(self, message: ClaudeMessage) -> AsyncIterable[str]:
        self._messages.append(message)
        async for text in self._generate_text_streamed():
            yield text

    async def _generate_text_streamed(self) -> AsyncIterable[str]:
        system_message = ClaudeMessage(
            role=ClaudeMessageRole.system, content=self._system_message
        )

        try:
            """
            Example Claude2 formatting for prompts:

            Cats are wonderful animals and loved by everyone, no matter how many legs they have.

            Human: I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?

            Assistant: Can I think step-by-step?

            Human: Yes, please do.

            Assistant:
            """
            prompt = "".join(
                [system_message.to_api()]
                + [m.to_api() for m in self._messages]
                + [ClaudeMessage(role=ClaudeMessageRole.assistant, content="").to_api()]
            )
            chat_stream = await asyncio.wait_for(
                self._client.completions.create(
                    model=self._model,
                    max_tokens_to_sample=300,
                    stream=True,
                    prompt=prompt,
                ),
                10,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        full_response = ""

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
            content = chunk.completion

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("Claude interrupted")
                break

            if content is not None:
                full_response += content
                yield content

        self._messages.append(
            ClaudeMessage(role=ClaudeMessageRole.assistant, content=full_response)
        )
        self._producing_response = False
