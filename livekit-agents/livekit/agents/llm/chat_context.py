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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from livekit import rtc
from typing_extensions import Literal

from . import function_context

ChatRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatImage:
    image: str | rtc.VideoFrame
    inference_width: int | None = None
    inference_height: int | None = None
    _cache: dict[Any, Any] = field(default_factory=dict, repr=False, init=False)
    """_cache is used  by LLM implementations to store a processed version of the image
    for later use.
    """


@dataclass
class ChatMessage:
    role: ChatRole
    name: str | None = None
    content: str | list[str | ChatImage] | None = None
    tool_calls: list[function_context.CalledFunction] | None = None
    tool_call_id: str | None = None

    @staticmethod
    def create_tool_from_called_function(
        called_function: function_context.CalledFunction,
    ) -> "ChatMessage":
        if not called_function.task.done():
            raise ValueError("cannot create a tool result from a running ai function")

        content = called_function.task.result()
        if called_function.task.exception() is not None:
            content = f"Error: {called_function.task.exception}"

        return ChatMessage(
            role="tool",
            name=called_function.function_info.name,
            content=content,
            tool_call_id=called_function.tool_call_id,
        )

    @staticmethod
    def create_tool_calls(
        called_functions: list[function_context.CalledFunction],
    ) -> "ChatMessage":
        return ChatMessage(
            role="assistant",
            tool_calls=called_functions,
        )

    @staticmethod
    def create(
        *, text: str = "", images: list[ChatImage] = [], role: ChatRole = "system"
    ) -> "ChatMessage":
        if len(images) == 0:
            return ChatMessage(role=role, content=text)
        else:
            content: list[str | ChatImage] = []
            if text:
                content.append(text)

            if len(images) > 0:
                content.extend(images)

            return ChatMessage(
                role=role,
                content=content,
            )

    def copy(self):
        content = self.content
        if isinstance(content, list):
            content = content.copy()

        tool_calls = self.tool_calls
        if tool_calls is not None:
            tool_calls = tool_calls.copy()

        return ChatMessage(
            role=self.role,
            name=self.name,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=self.tool_call_id,
        )


@dataclass
class ChatContext:
    messages: list[ChatMessage] = field(default_factory=list)

    def append(
        self, *, text: str = "", images: list[ChatImage] = [], role: ChatRole = "system"
    ) -> ChatContext:
        self.messages.append(ChatMessage.create(text=text, images=images, role=role))
        return self

    def copy(self) -> ChatContext:
        return ChatContext(messages=[m.copy() for m in self.messages])
