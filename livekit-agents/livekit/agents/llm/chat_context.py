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

from typing import (
    Literal,
)

from livekit import rtc
from pydantic import BaseModel


class ImageContent(BaseModel):
    """
    ChatImage is used to input images into the ChatContext on supported LLM providers / plugins.

    You may need to consult your LLM provider's documentation on supported URL types.

    ```python
    # Pass a VideoFrame directly, which will be automatically converted to a JPEG data URL internally
    async for event in rtc.VideoStream(video_track):
        chat_image = ChatImage(image=event.frame)
        # this instance is now available for your ChatContext

    # Encode your VideoFrame yourself for more control, and pass the result as a data URL (see EncodeOptions for more details)
    from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions

    image_bytes = encode(
        event.frame,
        EncodeOptions(
            format="PNG",
            resize_options=ResizeOptions(
                width=512, height=512, strategy="scale_aspect_fit"
            ),
        ),
    )
    chat_image = ChatImage(
        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    )

    # With an external URL
    chat_image = ChatImage(image="https://example.com/image.jpg")
    ```
    """

    image: str | rtc.VideoFrame
    """
    Either a string URL or a VideoFrame object
    """
    inference_width: int | None = None
    """
    Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    """
    inference_height: int | None = None
    """
    Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    """
    inference_detail: Literal["auto", "high", "low"] = "auto"
    """
    Detail parameter for LLM provider, if supported.
    
    Currently only supported by OpenAI (see https://platform.openai.com/docs/guides/vision?lang=node#low-or-high-fidelity-image-understanding)
    """


class AudioContent(BaseModel):
    frame: list[rtc.AudioFrame]
    transcript: str | None = None


class FunctionCall(BaseModel):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionCallOutput(BaseModel):
    type: Literal["function_call_output"]
    call_id: str
    output: str
    is_error: bool


class ChatMessage(BaseModel):
    type: Literal["message"]
    role: Literal["developer", "system", "user", "assistant"]
    content: list[str | ImageContent | AudioContent]
    hash: bytes | None = None


class ChatItem(BaseModel):
    id: str
    content: list[ChatMessage | FunctionCall | FunctionCallOutput]


class ChatContext:
    def __init__(self, items: list[ChatItem] | None = None):
        self._items: list[ChatItem] = items or []

    @property
    def items(self) -> list[ChatItem]:
        return self._items

    def get_by_id(self, item_id: str) -> ChatItem | None:
        # ideally, get_by_id should be O(1)
        for item in self.items:
            if item.id == item_id:
                return item

    def copy(self) -> "ChatContext":
        return ChatContext(self.items.copy())

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, _: dict) -> "ChatContext":
        raise NotImplementedError
