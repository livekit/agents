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

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from livekit import rtc

if TYPE_CHECKING:
    from livekit.agents.llm import LLM


class ChatRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatImage:
    image: str | rtc.VideoFrame
    inference_width: int | None = None
    inference_height: int | None = None
    _cache: Dict[LLM, Any] = field(default_factory=dict, repr=False, init=False)
    """_cache is used  by LLM implementations to store a processed version of the image
    for later use during inference. It is not intended to be used by the user code.
    """


@dataclass
class ChatMessage:
    role: ChatRole
    text: str
    images: List[ChatImage] = field(default_factory=list)

    def copy(self):
        return ChatMessage(
            role=self.role,
            text=self.text,
            images=self.images.copy(),  # Shallow copy is fine here, no use case right now for images to be mutated
        )


@dataclass
class ChatContext:
    messages: list[ChatMessage] = field(default_factory=list)

    def copy(self):
        return ChatContext(messages=[m.copy() for m in self.messages])
