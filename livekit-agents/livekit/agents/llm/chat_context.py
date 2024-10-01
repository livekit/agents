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
from typing import Any, Literal, Union

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
class ChatAudio:
    frame: rtc.AudioFrame | list[rtc.AudioFrame]


ChatContent = Union[str, ChatImage, ChatAudio]


@dataclass
class ChatMessage:
    role: ChatRole
    id: str | None = None  # used by the OAI realtime API
    name: str | None = None
    content: ChatContent | list[ChatContent] | None = None
    tool_calls: list[function_context.FunctionCallInfo] | None = None
    tool_call_id: str | None = None
    tool_exception: Exception | None = None
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    @staticmethod
    def create_tool_from_called_function(
        called_function: function_context.CalledFunction,
    ) -> "ChatMessage":
        if not called_function.task.done():
            raise ValueError("cannot create a tool result from a running ai function")

        tool_exception: Exception | None = None
        try:
            content = called_function.task.result()
        except BaseException as e:
            if isinstance(e, Exception):
                tool_exception = e
            content = f"Error: {e}"

        return ChatMessage(
            role="tool",
            name=called_function.call_info.function_info.name,
            content=content,
            tool_call_id=called_function.call_info.tool_call_id,
            tool_exception=tool_exception,
        )

    @staticmethod
    def create_tool_calls(
        called_functions: list[function_context.FunctionCallInfo],
    ) -> "ChatMessage":
        return ChatMessage(role="assistant", tool_calls=called_functions)

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

            return ChatMessage(role=role, content=content)

    def copy(self):
        content = self.content
        if isinstance(content, list):
            content = content.copy()

        tool_calls = self.tool_calls
        if tool_calls is not None:
            tool_calls = tool_calls.copy()

        copied_msg = ChatMessage(
            role=self.role,
            name=self.name,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=self.tool_call_id,
        )
        copied_msg._metadata = self._metadata
        return copied_msg


@dataclass
class ChatContext:
    messages: list[ChatMessage] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def append(
        self, *, text: str = "", images: list[ChatImage] = [], role: ChatRole = "system"
    ) -> ChatContext:
        self.messages.append(ChatMessage.create(text=text, images=images, role=role))
        return self

    def copy(self) -> ChatContext:
        copied_chat_ctx = ChatContext(messages=[m.copy() for m in self.messages])
        copied_chat_ctx._metadata = self._metadata
        return copied_chat_ctx
