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

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Literal, Unpack, overload

from pydantic import TypeAdapter

from .. import _core
from .._generated import public_llm_chat_context as _public
from .._generated.provider_data import (
    AnthropicFormatData,
    BedrockFormatData,
    GoogleFormatData,
    MistralFormatData,
    ProviderFormat,
)
from .._generated.public_llm_chat_context import *  # noqa: F403
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given

if TYPE_CHECKING:
    from anthropic.types import MessageParam as AnthropicMessage
    from google.genai.types import ContentDict as GoogleMessage
    from mistralai.client.models import InputEntriesTypedDict as MistralMessage
    from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
    from openai.types.responses import ResponseInputItemParam as OpenAIResponseItem
    from types_aiobotocore_bedrock_runtime.type_defs import MessageTypeDef as BedrockMessage

    from ..llm import Tool, Toolset

__all__ = [
    "ChatContext",
]
__all__ += _public.__all__


def _essential_fields(item: ChatItem) -> tuple[object, ...]:
    match item:
        case ChatMessage():
            return (item.role, item.interrupted, item.content)
        case FunctionCall():
            return (item.name, item.call_id, item.arguments)
        case FunctionCallOutput():
            return (item.name, item.call_id, item.output, item.is_error)
    return ()


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

    def messages(self) -> list[ChatMessage]:
        """Return only chat messages, ignoring function calls, outputs, and other events."""
        return [item for item in self._items if isinstance(item, ChatMessage)]

    def add_message(self, **message: Unpack[ChatMessageKwargs]) -> ChatMessage:
        """Add a message, placed by its ``created_at`` when given, else at the end."""
        item = ChatMessage.model_validate(message)
        if "created_at" in message:
            self._items.insert(self.find_insertion_index(created_at=item.created_at), item)
        else:
            self._items.append(item)
        return item

    def insert(self, item: ChatItem | Sequence[ChatItem]) -> None:
        """Insert an item or list of items into the chat context by creation time."""
        items = list(item) if isinstance(item, Sequence) else [item]

        for _item in items:
            idx = self.find_insertion_index(created_at=_item.created_at)
            self._items.insert(idx, _item)

    def remove(self, item: ChatItem | str) -> None:
        """Remove the first item from the chat context by ChatItem or item ID.

        Raises ValueError if the item/ID is not found.
        """
        idx = self.index_by_id(item.id if not isinstance(item, str) else item)
        if idx is None:
            raise ValueError(f"Item not found: {item!r}")
        self._items.pop(idx)

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
        exclude_handoff: bool = False,
        exclude_config_update: bool = False,
        tools: NotGivenOr[Sequence[Tool | Toolset | str]] = NOT_GIVEN,
    ) -> ChatContext:
        from .tool_context import FunctionTool, RawFunctionTool, Toolset

        def get_tool_names(
            tools: Sequence[Tool | Toolset | str],
        ) -> Generator[str, None, None]:
            for tool in tools:
                if isinstance(tool, str):
                    yield tool
                elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                    yield tool.info.name
                elif isinstance(tool, Toolset):
                    yield from get_tool_names(tool.tools)

        plan = _core.chat_ctx_copy_plan(
            self.items,
            exclude_function_call=exclude_function_call,
            exclude_instructions=exclude_instructions,
            exclude_empty_message=exclude_empty_message,
            exclude_handoff=exclude_handoff,
            exclude_config_update=exclude_config_update,
            tools=list(get_tool_names(tools)) if is_given(tools) else None,
        )
        return ChatContext([self.items[i] for i in plan])

    def truncate(self, *, max_items: int) -> ChatContext:
        """Truncate the chat context to the last `max_items` items in place.

        Leading function calls are dropped to avoid partial function outputs, and the first
        instruction message (system/developer) is kept at the beginning. A `max_items` of 0
        keeps only that instruction.

        Raises:
            ValueError: If `max_items` is negative.
        """
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        plan = _core.chat_ctx_truncate_plan(self._items, max_items)
        if plan is not None:
            self._items[:] = [self._items[i] for i in plan]
        return self

    def merge(
        self,
        other_chat_ctx: ChatContext,
        *,
        exclude_function_call: bool = False,
        exclude_instructions: bool = False,
        exclude_config_update: bool = False,
    ) -> ChatContext:
        """Add the items of `other_chat_ctx` missing from this one, sorted by `created_at`."""
        plan = _core.chat_ctx_merge_plan(
            self._items,
            other_chat_ctx.items,
            exclude_function_call=exclude_function_call,
            exclude_instructions=exclude_instructions,
            exclude_config_update=exclude_config_update,
        )
        for insertion in plan:
            self._items.insert(insertion.index, other_chat_ctx.items[insertion.item])
        return self

    def to_dict(
        self,
        *,
        exclude_image: bool = True,
        exclude_audio: bool = True,
        exclude_timestamp: bool = True,
        exclude_function_call: bool = False,
        exclude_metrics: bool = False,
        exclude_config_update: bool = False,
        strip_markup: bool = False,
    ) -> dict[str, object]:
        """``{"items": [...]}``, each item's JSON dump. ``strip_markup`` strips the ``<expr/>``
        dialect from assistant messages."""
        return _core.chat_ctx_to_dict(
            self.items,
            exclude_image=exclude_image,
            exclude_audio=exclude_audio,
            exclude_timestamp=exclude_timestamp,
            exclude_function_call=exclude_function_call,
            exclude_metrics=exclude_metrics,
            exclude_config_update=exclude_config_update,
            strip_markup=strip_markup,
        )

    @overload
    async def to_provider_format(
        self, format: Literal["openai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[OpenAIMessage], None]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["openai.responses"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[OpenAIResponseItem], None]: ...

    @overload
    async def to_provider_format(
        self,
        format: Literal["google"],
        *,
        inject_dummy_user_message: bool = True,
        thought_signatures: dict[str, bytes] | None = None,
    ) -> tuple[list[GoogleMessage], GoogleFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["aws"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[BedrockMessage], BedrockFormatData]: ...

    @overload
    async def to_provider_format(
        self,
        format: Literal["anthropic"],
        *,
        inject_dummy_user_message: bool = True,
        inject_trailing_user_message: bool = False,
    ) -> tuple[list[AnthropicMessage], AnthropicFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["mistralai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[MistralMessage], MistralFormatData]: ...

    async def to_provider_format(
        self,
        format: ProviderFormat,
        *,
        inject_dummy_user_message: bool = True,
        inject_trailing_user_message: bool = False,
        thought_signatures: dict[str, bytes] | None = None,
    ) -> tuple[
        list[OpenAIMessage]
        | list[OpenAIResponseItem]
        | list[GoogleMessage]
        | list[BedrockMessage]
        | list[AnthropicMessage]
        | list[MistralMessage],
        GoogleFormatData | BedrockFormatData | AnthropicFormatData | MistralFormatData | None,
    ]:
        """Convert history to the upstream SDK's request types for the selected provider."""
        from .utils import to_chat_ctx

        messages, data = await to_chat_ctx(
            self,
            format,
            inject_dummy_user_message=inject_dummy_user_message,
            inject_trailing_user_message=inject_trailing_user_message,
            thought_signatures=thought_signatures,
        )
        return messages, data

    def find_insertion_index(self, *, created_at: float) -> int:
        """The index after the last item with `created_at <=` the given timestamp.

        Assumes items are sorted by `created_at`.
        """
        for i in reversed(range(len(self._items))):
            if self._items[i].created_at <= created_at:
                return i + 1

        return 0

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChatContext:
        item_adapter: TypeAdapter[list[ChatItem]] = TypeAdapter(list[ChatItem])
        items = item_adapter.validate_python(data["items"])
        return cls(items)

    @property
    def readonly(self) -> bool:
        return False

    def is_equivalent(self, other: ChatContext) -> bool:
        """True if `other` has the same items by id, type and payload.

        Messages compare `content`, `role` and `interrupted`; function calls `name`, `call_id`
        and `arguments`; function call outputs `name`, `call_id`, `output` and `is_error`.
        Timestamps and other metadata are ignored.
        """
        if self is other:
            return True

        if len(self.items) != len(other.items):
            return False

        return all(
            a.id == b.id and a.type == b.type and _essential_fields(a) == _essential_fields(b)
            for a, b in zip(self.items, other.items, strict=True)
        )


class _ReadOnlyChatContext(ChatContext):
    error_msg = (
        "trying to modify a read-only chat context, "
        "edit the turn's context in on_user_turn_completed or hand a new history to "
        "session.update_agent(agent, chat_ctx=)"
    )

    class _ImmutableList(list[ChatItem]):
        def _raise_error(self, *args: object, **kwargs: object) -> None:
            logger.error(_ReadOnlyChatContext.error_msg)
            raise RuntimeError(_ReadOnlyChatContext.error_msg)

        append = extend = insert = pop = remove = clear = sort = reverse = _raise_error  # type: ignore
        __setitem__ = __delitem__ = __iadd__ = __imul__ = _raise_error  # type: ignore

        def copy(self) -> list[ChatItem]:
            return list(self)

    def __init__(self, items: list[ChatItem]):
        self._items = self._ImmutableList(items)

    @property
    def items(self) -> list[ChatItem]:
        return self._items

    @items.setter
    def items(self, items: list[ChatItem]) -> None:
        logger.error(_ReadOnlyChatContext.error_msg)
        raise RuntimeError(_ReadOnlyChatContext.error_msg)

    @property
    def readonly(self) -> bool:
        return True
