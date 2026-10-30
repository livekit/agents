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
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

from pydantic import TypeAdapter

from livekit.protocol.agent_pb import agent_session as agent_pb

from .. import _core
from .._generated.chat_items import (
    AgentConfigUpdate,
    AgentHandoff,
    AudioContent,
    ChatContent,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    MetricsReport,
)
from .._proto import encode_chat_item
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from . import _provider_format

if TYPE_CHECKING:
    from anthropic.types import MessageParam as AnthropicMessage
    from google.genai.types import ContentDict as GoogleMessage
    from mistralai.client.models import InputEntriesTypedDict as MistralMessage
    from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
    from openai.types.responses import ResponseInputItemParam as OpenAIResponseItem
    from types_aiobotocore_bedrock_runtime.type_defs import MessageTypeDef as BedrockMessage

    from ..llm import LLM, Tool, Toolset

__all__ = [
    "AgentConfigUpdate",
    "AgentHandoff",
    "AudioContent",
    "ChatContent",
    "ChatContext",
    "ChatItem",
    "ChatMessage",
    "ChatRole",
    "FunctionCall",
    "FunctionCallOutput",
    "ImageContent",
]

ProviderFormat: TypeAlias = Literal[
    "openai", "openai.responses", "google", "aws", "anthropic", "mistralai"
]


def _essential_fields(item: ChatItem) -> tuple[object, ...]:
    """What identifies an item's content beyond its id and type: what
    :meth:`ChatContext.is_equivalent` compares. Timestamps, metrics and other metadata are left
    out."""
    match item:
        case ChatMessage():
            return (item.role, item.interrupted, item.content)
        case FunctionCall():
            return (item.name, item.call_id, item.arguments)
        case FunctionCallOutput():
            return (item.name, item.call_id, item.output, item.is_error)
    return ()


def _facts(
    items: list[ChatItem],
) -> list[tuple[str, str, str | None, int, str | None, float, str | None]]:
    """What the core's plans read of each item: ``(type, id, role, len(content), name,
    created_at, call_id)``."""
    return [
        (
            item.type,
            item.id,
            item.role if isinstance(item, ChatMessage) else None,
            len(item.content) if item.type == "message" else 0,
            item.name if isinstance(item, (FunctionCall, FunctionCallOutput)) else None,
            item.created_at,
            item.call_id if isinstance(item, (FunctionCall, FunctionCallOutput)) else None,
        )
        for item in items
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

    def messages(self) -> list[ChatMessage]:
        """Return only chat messages, ignoring function calls, outputs, and other events."""
        return [item for item in self._items if isinstance(item, ChatMessage)]

    def add_message(
        self,
        *,
        role: ChatRole,
        content: list[ChatContent] | str,
        id: NotGivenOr[str] = NOT_GIVEN,
        interrupted: NotGivenOr[bool] = NOT_GIVEN,
        created_at: NotGivenOr[float] = NOT_GIVEN,
        metrics: NotGivenOr[MetricsReport] = NOT_GIVEN,
        extra: NotGivenOr[dict[str, object]] = NOT_GIVEN,
    ) -> ChatMessage:
        given = {
            "id": id,
            "interrupted": interrupted,
            "created_at": created_at,
            "metrics": metrics,
            "extra": extra,
        }
        fields: dict[str, object] = {
            "role": role,
            "content": [content] if isinstance(content, str) else content,
            **{key: value for key, value in given.items() if is_given(value)},
        }
        message = ChatMessage.model_validate(fields)
        if is_given(created_at):
            self._items.insert(self.find_insertion_index(created_at=created_at), message)
        else:
            self._items.append(message)
        return message

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
            _facts(self.items),
            exclude_function_call=exclude_function_call,
            exclude_instructions=exclude_instructions,
            exclude_empty_message=exclude_empty_message,
            exclude_handoff=exclude_handoff,
            exclude_config_update=exclude_config_update,
            tools=list(get_tool_names(tools)) if is_given(tools) else None,
        )
        return ChatContext([self.items[i] for i in plan])

    def truncate(self, *, max_items: int) -> ChatContext:
        """Truncate the chat context to the last N items in place.

        Removes leading function calls to avoid partial function outputs.
        Preserves the first instruction message (system/developer) by adding it back
        to the beginning.

        A `max_items` of 0 leaves nothing but that instruction: it asks for no conversational
        items, so none are kept. A negative value is a programming error and raises ValueError.
        """
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        plan = _core.chat_ctx_truncate_plan(_facts(self._items), max_items)
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
        """Add messages from `other_chat_ctx` into this one, avoiding duplicates, and keep items sorted by created_at."""
        plan = _core.chat_ctx_merge_plan(
            _facts(self._items),
            _facts(other_chat_ctx.items),
            exclude_function_call=exclude_function_call,
            exclude_instructions=exclude_instructions,
            exclude_config_update=exclude_config_update,
        )
        for idx, j in plan:
            self._items.insert(idx, other_chat_ctx.items[j])
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
        """``{"items": [...]}``, each item's JSON dump (the core's ``chat_ctx_to_dict``).
        ``strip_markup`` strips the ``<expr/>`` dialect from assistant messages."""
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
    ) -> tuple[list[GoogleMessage], _provider_format.GoogleFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["aws"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[BedrockMessage], _provider_format.BedrockFormatData]: ...

    @overload
    async def to_provider_format(
        self,
        format: Literal["anthropic"],
        *,
        inject_dummy_user_message: bool = True,
        inject_trailing_user_message: bool = False,
    ) -> tuple[list[AnthropicMessage], _provider_format.AnthropicFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["mistralai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[MistralMessage], _provider_format.MistralFormatData]: ...

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
        _provider_format.GoogleFormatData
        | _provider_format.BedrockFormatData
        | _provider_format.AnthropicFormatData
        | _provider_format.MistralFormatData
        | None,
    ]:
        """Convert history to the upstream SDK's request types for the selected provider."""
        # The native dialect writes the SDK wire schema; annotations preserve that contract.
        messages, data = await _provider_format.to_chat_ctx(
            self,
            format,
            inject_dummy_user_message=inject_dummy_user_message,
            inject_trailing_user_message=inject_trailing_user_message,
            thought_signatures=thought_signatures,
        )
        return messages, data

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

    async def _summarize(self, llm_v: LLM, *, keep_last_turns: int = 2) -> ChatContext:
        """Replace all but the last ``keep_last_turns`` turns with an LLM-written summary."""
        plan = _core.chat_ctx_summarize_plan(self.items, keep_last_turns)
        if plan is None:
            return self
        split_idx, preserved, system_prompt, user_content, created_at = plan
        request = ChatContext()
        request.add_message(role="system", content=system_prompt)
        request.add_message(role="user", content=user_content)
        async with llm_v.chat(chat_ctx=request) as stream:
            summary = "".join(
                [c.delta.content async for c in stream if c.delta and c.delta.content]
            ).strip()
        if not summary:
            return self
        tail = self._items[split_idx:]
        self._items = [self._items[i] for i in preserved]
        self.add_message(
            role="assistant",
            content=_core.chat_history_summary_xml(summary),
            created_at=created_at,
            extra={"is_summary": True},
        )
        self._items.extend(tail)
        return self

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChatContext:
        item_adapter: TypeAdapter[list[ChatItem]] = TypeAdapter(list[ChatItem])
        items = item_adapter.validate_python(data["items"])
        return cls(items)

    def to_proto(self) -> agent_pb.ChatContext:
        return agent_pb.ChatContext(items=[encode_chat_item(item) for item in self.items])

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

        return all(
            a.id == b.id and a.type == b.type and _essential_fields(a) == _essential_fields(b)
            for a, b in zip(self.items, other.items, strict=True)
        )


class _ReadOnlyChatContext(ChatContext):
    """A read-only wrapper for ChatContext that prevents modifications."""

    error_msg = (
        "trying to modify a read-only chat context, "
        "edit the turn's context in on_user_turn_completed or hand a new history to "
        "session.update_agent(agent, chat_ctx=)"
    )

    class _ImmutableList(list[ChatItem]):
        def _raise_error(self, *args: object, **kwargs: object) -> None:
            logger.error(_ReadOnlyChatContext.error_msg)
            raise RuntimeError(_ReadOnlyChatContext.error_msg)

        # override all mutating methods to raise errors
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


def to_xml(
    tag_name: str, content: str | None = None, attrs: dict[str, object] | None = None
) -> str:
    return _core.to_xml(tag_name, content, [(k, str(v)) for k, v in (attrs or {}).items()])
