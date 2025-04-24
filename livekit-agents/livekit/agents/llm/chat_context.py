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

import time
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from typing_extensions import TypeAlias

from livekit import rtc
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.misc import is_given

from .. import utils
from ..log import logger

if TYPE_CHECKING:
    from ..llm import FunctionTool, RawFunctionTool


class ImageContent(BaseModel):
    """
    ImageContent is used to input images into the ChatContext on supported LLM providers / plugins.

    You may need to consult your LLM provider's documentation on supported URL types.

    ```python
    # Pass a VideoFrame directly, which will be automatically converted to a JPEG data URL internally
    async for event in rtc.VideoStream(video_track):
        chat_image = ImageContent(image=event.frame)
        # this instance is now available for your ChatContext

    # Encode your VideoFrame yourself for more control, and pass the result as a data URL (see EncodeOptions for more details)
    from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions

    image_bytes = encode(
        event.frame,
        EncodeOptions(
            format="PNG",
            resize_options=ResizeOptions(width=512, height=512, strategy="scale_aspect_fit"),
        ),
    )
    chat_image = ImageContent(
        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    )

    # With an external URL
    chat_image = ImageContent(image="https://example.com/image.jpg")
    ```
    """  # noqa: E501

    id: str = Field(default_factory=lambda: utils.shortuuid("img_"))
    """
    Unique identifier for the image
    """

    type: Literal["image_content"] = Field(default="image_content")

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
    mime_type: str | None = None
    """
    MIME type of the image
    """
    _cache: dict[int, Any] = PrivateAttr(default_factory=dict)


class AudioContent(BaseModel):
    type: Literal["audio_content"] = Field(default="audio_content")
    frame: list[rtc.AudioFrame]
    transcript: str | None = None


ChatRole: TypeAlias = Literal["developer", "system", "user", "assistant"]


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["message"] = "message"
    role: ChatRole
    content: list[ChatContent]
    interrupted: bool = False
    hash: bytes | None = None
    created_at: float = Field(default_factory=time.time)

    @property
    def text_content(self) -> str | None:
        """
        Returns a string of all text content in the message.

        Multiple text content items will be joined by a newline.
        """
        text_parts = [c for c in self.content if isinstance(c, str)]
        if not text_parts:
            return None
        return "\n".join(text_parts)


ChatContent: TypeAlias = Union[ImageContent, AudioContent, str]


class FunctionCall(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call"] = "function_call"
    call_id: str
    arguments: str
    name: str


class FunctionCallOutput(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    name: str = Field(default="")
    type: Literal["function_call_output"] = Field(default="function_call_output")
    call_id: str
    output: str
    is_error: bool


ChatItem = Annotated[
    Union[ChatMessage, FunctionCall, FunctionCallOutput], Field(discriminator="type")
]


class ChatContext:
    def __init__(self, items: NotGivenOr[list[ChatItem]] = NOT_GIVEN):
        self._items: list[ChatItem] = items if is_given(items) else []

    @classmethod
    def empty(cls) -> ChatContext:
        return cls([])

    @property
    def items(self) -> list[ChatItem]:
        return self._items

    @items.setter
    def items(self, items: list[ChatItem]):
        self._items = items

    def add_message(
        self,
        *,
        role: ChatRole,
        content: list[ChatContent] | str,
        id: NotGivenOr[str] = NOT_GIVEN,
        interrupted: NotGivenOr[bool] = NOT_GIVEN,
    ) -> ChatMessage:
        kwargs = {}
        if is_given(id):
            kwargs["id"] = id
        if is_given(interrupted):
            kwargs["interrupted"] = interrupted

        if isinstance(content, str):
            message = ChatMessage(role=role, content=[content], **kwargs)
        else:
            message = ChatMessage(role=role, content=content, **kwargs)

        self._items.append(message)
        return message

    def get_by_id(self, item_id: str) -> ChatItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def index_by_id(self, item_id: str) -> int | None:
        return next((i for i, item in enumerate(self.items) if item.id == item_id), None)

    def copy(
        self,
        *,
        exclude_function_call: bool = False,
        exclude_instructions: bool = False,
        tools: NotGivenOr[list[FunctionTool | RawFunctionTool | str | Any]] = NOT_GIVEN,
    ) -> ChatContext:
        items = []

        from .tool_context import (
            get_function_info,
            get_raw_function_info,
            is_function_tool,
            is_raw_function_tool,
        )

        valid_tools = set()
        if is_given(tools):
            for tool in tools:
                if isinstance(tool, str):
                    valid_tools.add(tool)
                elif is_function_tool(tool):
                    valid_tools.add(get_function_info(tool).name)
                elif is_raw_function_tool(tool):
                    valid_tools.add(get_raw_function_info(tool).name)
                # TODO(theomonnom): other tools

        for item in self.items:
            if exclude_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                continue

            if (
                exclude_instructions
                and item.type == "message"
                and item.role in ["system", "developer"]
            ):
                continue

            if (
                is_given(tools)
                and item.type in ["function_call", "function_call_output"]
                and item.name not in valid_tools
            ):
                continue

            items.append(item)

        return ChatContext(items)

    def truncate(self, *, max_items: int) -> ChatContext:
        """Truncate the chat context to the last N items in place.

        Removes leading function calls to avoid partial function outputs.
        Preserves the first system message by adding it back to the beginning.
        """
        instructions = next(
            (item for item in self._items if item.type == "message" and item.role == "system"),
            None,
        )

        new_items = self._items[-max_items:]
        # chat ctx shouldn't start with function_call or function_call_output
        while new_items and new_items[0].type in [
            "function_call",
            "function_call_output",
        ]:
            new_items.pop(0)

        if instructions:
            new_items.insert(0, instructions)

        self._items[:] = new_items
        return self

    def to_dict(
        self,
        *,
        exclude_image: bool = True,
        exclude_audio: bool = True,
        exclude_function_call: bool = False,
    ) -> dict:
        items = []
        for item in self.items:
            if exclude_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                continue

            if item.type == "message":
                item = item.model_copy()
                if exclude_image:
                    item.content = [c for c in item.content if not isinstance(c, ImageContent)]
                if exclude_audio:
                    item.content = [c for c in item.content if not isinstance(c, AudioContent)]

            items.append(item)

        return {
            "items": [
                item.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
                for item in items
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatContext:
        item_adapter = TypeAdapter(list[ChatItem])
        items = item_adapter.validate_python(data["items"])
        return cls(items)

    @property
    def readonly(self) -> bool:
        return False


class _ReadOnlyChatContext(ChatContext):
    """A read-only wrapper for ChatContext that prevents modifications."""

    error_msg = (
        "trying to modify a read-only chat context, "
        "please use .copy() and agent.update_chat_ctx() to modify the chat context"
    )

    class _ImmutableList(list):
        def _raise_error(self, *args, **kwargs):
            logger.error(_ReadOnlyChatContext.error_msg)
            raise RuntimeError(_ReadOnlyChatContext.error_msg)

        # override all mutating methods to raise errors
        append = extend = pop = remove = clear = sort = reverse = _raise_error  # type: ignore
        __setitem__ = __delitem__ = __iadd__ = __imul__ = _raise_error  # type: ignore

        def copy(self):
            return list(self)

    def __init__(self, items: list[ChatItem]):
        self._items = self._ImmutableList(items)

    @property
    def readonly(self) -> bool:
        return True
