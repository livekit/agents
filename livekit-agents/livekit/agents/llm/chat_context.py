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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, overload

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from typing_extensions import TypeAlias

from livekit import rtc

from .. import utils
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from . import _provider_format

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
    _cache: dict[Any, Any] = PrivateAttr(default_factory=dict)


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
    transcript_confidence: float | None = None
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
    created_at: float = Field(default_factory=time.time)


class FunctionCallOutput(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call_output"] = Field(default="function_call_output")
    name: str = Field(default="")
    call_id: str
    output: str
    is_error: bool
    created_at: float = Field(default_factory=time.time)


""""
class AgentHandoff(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_handoff"] = Field(default="agent_handoff")
    old_agent_id: str | None
    new_agent_id: str
    old_agent: Agent | None = Field(exclude=True)
    new_agent: Agent | None = Field(exclude=True)
    created_at: float = Field(default_factory=time.time)
"""

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
    def items(self, items: list[ChatItem]) -> None:
        self._items = items

    def add_message(
        self,
        *,
        role: ChatRole,
        content: list[ChatContent] | str,
        id: NotGivenOr[str] = NOT_GIVEN,
        interrupted: NotGivenOr[bool] = NOT_GIVEN,
        created_at: NotGivenOr[float] = NOT_GIVEN,
    ) -> ChatMessage:
        kwargs: dict[str, Any] = {}
        if is_given(id):
            kwargs["id"] = id
        if is_given(interrupted):
            kwargs["interrupted"] = interrupted
        if is_given(created_at):
            kwargs["created_at"] = created_at

        if isinstance(content, str):
            message = ChatMessage(role=role, content=[content], **kwargs)
        else:
            message = ChatMessage(role=role, content=content, **kwargs)

        if is_given(created_at):
            idx = self.find_insertion_index(created_at=created_at)
            self._items.insert(idx, message)
        else:
            self._items.append(message)
        return message

    def insert(self, item: ChatItem | Sequence[ChatItem]) -> None:
        """Insert an item or list of items into the chat context by creation time."""
        items = list(item) if isinstance(item, Sequence) else [item]

        for _item in items:
            idx = self.find_insertion_index(created_at=_item.created_at)
            self._items.insert(idx, _item)

    def get_by_id(self, item_id: str) -> ChatItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def index_by_id(self, item_id: str) -> int | None:
        return next((i for i, item in enumerate(self.items) if item.id == item_id), None)

    def copy(
        self,
        *,
        exclude_function_call: bool = False,
        exclude_instructions: bool = False,
        exclude_empty_message: bool = False,
        tools: NotGivenOr[Sequence[FunctionTool | RawFunctionTool | str | Any]] = NOT_GIVEN,
    ) -> ChatContext:
        items = []

        from .tool_context import (
            get_function_info,
            get_raw_function_info,
            is_function_tool,
            is_raw_function_tool,
        )

        valid_tools = set[str]()
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

            if exclude_empty_message and item.type == "message" and not item.content:
                continue

            if (
                is_given(tools)
                and (item.type == "function_call" or item.type == "function_call_output")
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

    def merge(
        self,
        other_chat_ctx: ChatContext,
        *,
        exclude_function_call: bool = False,
        exclude_instructions: bool = False,
    ) -> ChatContext:
        """Add messages from `other_chat_ctx` into this one, avoiding duplicates, and keep items sorted by created_at."""
        existing_ids = {item.id for item in self._items}

        for item in other_chat_ctx.items:
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

            if item.id not in existing_ids:
                idx = self.find_insertion_index(created_at=item.created_at)
                self._items.insert(idx, item)
                existing_ids.add(item.id)

        return self

    def to_dict(
        self,
        *,
        exclude_image: bool = True,
        exclude_audio: bool = True,
        exclude_timestamp: bool = True,
        exclude_function_call: bool = False,
    ) -> dict[str, Any]:
        items: list[ChatItem] = []
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

        exclude_fields = set()
        if exclude_timestamp:
            exclude_fields.add("created_at")

        return {
            "items": [
                item.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_defaults=False,
                    exclude=exclude_fields,
                )
                for item in items
            ],
        }

    @overload
    def to_provider_format(
        self, format: Literal["openai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], Literal[None]]: ...

    @overload
    def to_provider_format(
        self, format: Literal["google"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.google.GoogleFormatData]: ...

    @overload
    def to_provider_format(
        self, format: Literal["aws"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.aws.BedrockFormatData]: ...

    @overload
    def to_provider_format(
        self, format: Literal["anthropic"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.anthropic.AnthropicFormatData]: ...

    @overload
    def to_provider_format(
        self, format: Literal["mistralai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], Literal[None]]: ...

    @overload
    def to_provider_format(self, format: str, **kwargs: Any) -> tuple[list[dict], Any]: ...

    def to_provider_format(
        self,
        format: Literal["openai", "google", "aws", "anthropic", "mistralai"] | str,
        *,
        inject_dummy_user_message: bool = True,
        **kwargs: Any,
    ) -> tuple[list[dict], Any]:
        """Convert the chat context to a provider-specific format.

        If ``inject_dummy_user_message`` is ``True``, a dummy user message will be added
        to the beginning or end of the chat context depending on the provider.

        This is necessary because some providers expect a user message to be present for
        generating a response.
        """
        kwargs["inject_dummy_user_message"] = inject_dummy_user_message

        if format == "openai":
            return _provider_format.openai.to_chat_ctx(self, **kwargs)
        elif format == "google":
            return _provider_format.google.to_chat_ctx(self, **kwargs)
        elif format == "aws":
            return _provider_format.aws.to_chat_ctx(self, **kwargs)
        elif format == "anthropic":
            return _provider_format.anthropic.to_chat_ctx(self, **kwargs)
        elif format == "mistralai":
            return _provider_format.mistralai.to_chat_ctx(self, **kwargs)
        else:
            raise ValueError(f"Unsupported provider format: {format}")

    def find_insertion_index(self, *, created_at: float) -> int:
        """
        Returns the index to insert an item by creation time.

        Iterates in reverse, assuming items are sorted by `created_at`.
        Finds the position after the last item with `created_at <=` the given timestamp.
        """
        for i in reversed(range(len(self._items))):
            if self._items[i].created_at <= created_at:
                return i + 1

        return 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatContext:
        item_adapter = TypeAdapter(list[ChatItem])
        items = item_adapter.validate_python(data["items"])
        return cls(items)

    @property
    def readonly(self) -> bool:
        return False

    def is_equivalent(self, other: ChatContext) -> bool:
        """
        Return True if `other` has the same sequence of items with matching
        essential fields (IDs, types, and payload) as this context.

        Comparison rules:
          - Messages: compares the full `content` list, `role` and `interrupted`.
          - Function calls: compares `name`, `call_id`, and `arguments`.
          - Function call outputs: compares `name`, `call_id`, `output`, and `is_error`.

        Does not consider timestamps or other metadata.
        """
        if self is other:
            return True

        if len(self.items) != len(other.items):
            return False

        for a, b in zip(self.items, other.items):
            if a.id != b.id or a.type != b.type:
                return False

            if a.type == "message" and b.type == "message":
                if a.role != b.role or a.interrupted != b.interrupted or a.content != b.content:
                    return False

            elif a.type == "function_call" and b.type == "function_call":
                if a.name != b.name or a.call_id != b.call_id or a.arguments != b.arguments:
                    return False

            elif a.type == "function_call_output" and b.type == "function_call_output":
                if (
                    a.name != b.name
                    or a.call_id != b.call_id
                    or a.output != b.output
                    or a.is_error != b.is_error
                ):
                    return False

        return True


class _ReadOnlyChatContext(ChatContext):
    """A read-only wrapper for ChatContext that prevents modifications."""

    error_msg = (
        "trying to modify a read-only chat context, "
        "please use .copy() and agent.update_chat_ctx() to modify the chat context"
    )

    class _ImmutableList(list[ChatItem]):
        def _raise_error(self, *args: Any, **kwargs: Any) -> None:
            logger.error(_ReadOnlyChatContext.error_msg)
            raise RuntimeError(_ReadOnlyChatContext.error_msg)

        # override all mutating methods to raise errors
        append = extend = pop = remove = clear = sort = reverse = _raise_error  # type: ignore
        __setitem__ = __delitem__ = __iadd__ = __imul__ = _raise_error  # type: ignore

        def copy(self) -> list[ChatItem]:
            return list(self)

    def __init__(self, items: list[ChatItem]):
        self._items = self._ImmutableList(items)

    @property
    def readonly(self) -> bool:
        return True
