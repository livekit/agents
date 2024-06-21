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

from typing_extensions import Literal, Required, TypedDict

import enum
from dataclasses import dataclass, field
from typing import Any

from livekit import rtc

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
    def from_called_function(
        called_function: function_context.CalledFunction,
    ) -> "ChatMessage":
        if not called_function.task.done():
            raise ValueError("cannot create a tool result from a running ai function")

        content = called_function.result
        if called_function.exception is not None:
            content = f"Error: {called_function.exception}"

        return ChatMessage(
            role="tool",
            name=called_function.function_info.name,
            content=content,
            tool_call_id=called_function.id,
        )

    @staticmethod
    def create(*, text: str = "", images: list[ChatImage] = [], role: ChatRole = "system") -> "ChatMessage":
        content = []
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

    def append(self, *, text: str="", images: list[ChatImage] = [], role: ChatRole = "system") -> ChatContext:
        message = ChatMessage(role=role, content=text)
        self.messages.append(ChatMessage.create(text=text, images=images, role=role))
        return self

    def copy(self) -> ChatContext:
        return ChatContext(messages=[m.copy() for m in self.messages])
