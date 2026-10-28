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
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, overload

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from typing_extensions import TypedDict

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from .. import _core, utils
from .._proto import encode_chat_item
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from . import _provider_format

if TYPE_CHECKING:
    from ..llm import LLM, Tool, Toolset


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

    image_bytes = await encode(
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
    """

    id: str = Field(default_factory=lambda: utils.shortuuid("img_"))
    """
    Unique identifier for the image. Use a new id when replacing inline image data
    or a video frame; input-delta telemetry uses this id instead of hashing media bytes.
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


# The metrics are stored in a dict, since some fields may not be relevant
# in certain context (e.g., text-only mode or when using a speech-to-speech model).
class MetricsMetadata(TypedDict, total=False):
    model_name: str
    model_provider: str


class MetricsReport(TypedDict, total=False):
    started_speaking_at: float
    stopped_speaking_at: float

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech

    User `ChatMessage` only
    """

    end_of_turn_delay: float
    """Amount of time between the end of speech and the decision to end the user's turn

    User `ChatMessage` only
    """

    on_user_turn_completed_delay: float
    """Time taken to invoke the developer's `Agent.on_user_turn_completed` callback.

    User `ChatMessage` only
    """

    llm_node_ttft: float
    """Time taken for the `llm_node` to return the first token

    Assistant `ChatMessage` only
    """

    llm_node_tps: float
    """LLM output tokens per second for this turn, measured at the `llm_node` over the
    streaming window (first to last text chunk). Absent for a reply that arrived in a
    single chunk, which has no measurable rate

    Assistant `ChatMessage` only
    """

    llm_node_ttfs: float
    """Time from LLM generation start until the first sentence reached the TTS provider, as
    segmented by that TTS. Absent when no audio came from a LiveKit TTS this turn: no TTS,
    an interruption before the first frame, or a `tts_node` synthesizing audio on its own

    Assistant `ChatMessage` only
    """

    tts_node_ttfb: float
    """Time taken for the `tts_node` to return the first chunk of audio (after the first text token has been sent)

    Assistant `ChatMessage` only
    """

    playback_latency: float
    """Delay between forwarding the first audio frame and the `AudioOutput` reporting
    playback started. Near-zero for the default room output (self-reported when the frame
    is pushed to the track, so it doesn't account for network delivery to the client);
    meaningful when a remote avatar worker is in the chain and reports playback via
    the `lk.playback_started` RPC.

    Assistant `ChatMessage` only
    """

    e2e_latency: float
    """Time from when the user finished speaking to when the agent began responding

    Assistant `ChatMessage` only
    """

    provider_request_ids: list[str]
    """Provider-known request or response IDs associated with this turn.

    Assistant `ChatMessage` only. These IDs can be used to correlate a turn with
    provider-side logs.
    """

    llm_metadata: MetricsMetadata
    tts_metadata: MetricsMetadata
    stt_metadata: MetricsMetadata


class _ChatItemBase(BaseModel):
    """Shared by every :data:`ChatItem` type; adds no fields."""

    def _essential_fields(self) -> tuple[Any, ...]:
        """What identifies this item's content beyond its id and type: what
        :meth:`ChatContext.is_equivalent` compares. Timestamps, metrics and other metadata are
        left out."""
        return ()


class ChatMessage(_ChatItemBase):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["message"] = "message"
    role: ChatRole
    content: list[ChatContent]
    interrupted: bool = False
    transcript_confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    metrics: MetricsReport = Field(default_factory=lambda: MetricsReport())
    created_at: float = Field(default_factory=time.time)

    @property
    def text_content(self) -> str | None:
        """
        Returns a string of all text content in the message, with LiveKit's
        expressive ``<expr/>`` tags removed from assistant messages.

        Multiple text content items will be joined by a newline.
        Use :attr:`raw_text_content` for the exact model-facing content.
        """
        raw = self.raw_text_content
        if raw is None or self.role != "assistant":
            return raw

        from ..tts import strip_expr_markup

        return strip_expr_markup(raw)

    @property
    def raw_text_content(self) -> str | None:
        """
        Returns a string of all text content in the message, exactly as generated
        (assistant messages may contain expressive ``<expr/>`` tags).

        Multiple text content items will be joined by a newline.
        """
        text_parts = [c for c in self.content if isinstance(c, str)]
        if not text_parts:
            return None
        return "\n".join(text_parts)

    def _essential_fields(self) -> tuple[Any, ...]:
        return (self.role, self.interrupted, self.content)


ChatContent: TypeAlias = ImageContent | AudioContent | str


class FunctionCall(_ChatItemBase):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call"] = "function_call"
    call_id: str
    arguments: str
    name: str
    created_at: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)
    """Extra data for this function call. Can include provider-specific data
    (e.g., extra["google"] for thought signatures)."""
    group_id: str | None = None
    """Optional group ID for parallel function calls. When multiple function calls
    should be grouped together (e.g., parallel tool calls from a single API response),
    set this to a shared value. If not set, falls back to using id for grouping."""

    def _essential_fields(self) -> tuple[Any, ...]:
        return (self.name, self.call_id, self.arguments)


class FunctionCallOutput(_ChatItemBase):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call_output"] = Field(default="function_call_output")
    name: str = Field(default="")
    call_id: str
    output: str
    is_error: bool
    created_at: float = Field(default_factory=time.time)
    reply_required: bool = Field(default=True)
    """Whether the model should answer once it receives this output.

    VoiceSession uses it to decide whether to generate a follow-up reply.
    Realtime models can also use it to schedule their response.
    """

    def _essential_fields(self) -> tuple[Any, ...]:
        return (self.name, self.call_id, self.output, self.is_error)


class AgentHandoff(_ChatItemBase):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_handoff"] = Field(default="agent_handoff")
    old_agent_id: str | None = None
    new_agent_id: str
    created_at: float = Field(default_factory=time.time)


class AgentConfigUpdate(_ChatItemBase):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_config_update"] = Field(default="agent_config_update")

    instructions: str | None = None
    tools_added: list[str] | None = None
    tools_removed: list[str] | None = None

    created_at: float = Field(default_factory=time.time)

    _tools: list[Tool] = PrivateAttr(default_factory=list)
    """Full tool definitions (in-memory only, not serialized)."""


ChatItem = Annotated[
    ChatMessage | FunctionCall | FunctionCallOutput | AgentHandoff | AgentConfigUpdate,
    Field(discriminator="type"),
]


def _facts(items: list[ChatItem]) -> list[tuple[Any, ...]]:
    """What the core's plans read of each item: ``(type, id, role, len(content), name,
    created_at, call_id)``."""
    return [
        (
            item.type,
            item.id,
            getattr(item, "role", None),
            len(item.content) if item.type == "message" else 0,
            getattr(item, "name", None) if item.type.startswith("function_call") else None,
            item.created_at,
            getattr(item, "call_id", None),
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
        extra: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ChatMessage:
        given = {"id": id, "interrupted": interrupted, "metrics": metrics, "extra": extra}
        kwargs: dict[str, Any] = {k: v for k, v in given.items() if is_given(v)}
        if isinstance(content, str):
            content = [content]
        if not is_given(created_at):
            message = ChatMessage(role=role, content=content, **kwargs)
            self._items.append(message)
            return message
        message = ChatMessage(role=role, content=content, created_at=created_at, **kwargs)
        self._items.insert(self.find_insertion_index(created_at=created_at), message)
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
    ) -> dict[str, Any]:
        """``{"items": [...]}``, each item's JSON dump (the core's ``chat_ctx_to_dict``).
        ``strip_markup`` strips the ``<expr/>`` dialect from assistant messages."""
        from .._core_view import to_core_items

        return _core.chat_ctx_to_dict(  # type: ignore[no-any-return]
            to_core_items(self.items),
            exclude_image=exclude_image,
            exclude_audio=exclude_audio,
            exclude_timestamp=exclude_timestamp,
            exclude_function_call=exclude_function_call,
            exclude_metrics=exclude_metrics,
            exclude_config_update=exclude_config_update,
            strip_expr_markup=_core.strip_expr_markup if strip_markup else None,
        )

    @overload
    async def to_provider_format(
        self,
        format: Literal["openai", "openai.responses"],
        *,
        inject_dummy_user_message: bool = True,
    ) -> tuple[list[dict], Literal[None]]: ...

    @overload
    async def to_provider_format(
        self,
        format: Literal["google"],
        *,
        inject_dummy_user_message: bool = True,
        thought_signatures: dict[str, bytes] | None = None,
    ) -> tuple[list[dict], _provider_format.GoogleFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["aws"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.BedrockFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["anthropic"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.AnthropicFormatData]: ...

    @overload
    async def to_provider_format(
        self, format: Literal["mistralai"], *, inject_dummy_user_message: bool = True
    ) -> tuple[list[dict], _provider_format.MistralFormatData]: ...

    @overload
    async def to_provider_format(self, format: str, **kwargs: Any) -> tuple[list[dict], Any]: ...

    async def to_provider_format(
        self,
        format: Literal["openai", "openai.responses", "google", "aws", "anthropic", "mistralai"]
        | str,
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
        return await _provider_format.to_chat_ctx(
            self, format, inject_dummy_user_message=inject_dummy_user_message, **kwargs
        )

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
        from .._core_view import to_core_items

        plan = _core.chat_ctx_summarize_plan(to_core_items(self.items), keep_last_turns)
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
    def from_dict(cls, data: dict[str, Any]) -> ChatContext:
        item_adapter = TypeAdapter(list[ChatItem])
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

        Does not consider timestamps or other metadata. Each item type declares its fields
        in ``_essential_fields``, which item fingerprints hash as well.
        """
        if self is other:
            return True

        if len(self.items) != len(other.items):
            return False

        return all(
            a.id == b.id and a.type == b.type and a._essential_fields() == b._essential_fields()
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
        def _raise_error(self, *args: Any, **kwargs: Any) -> None:
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


def to_xml(tag_name: str, content: str | None = None, attrs: dict[str, Any] | None = None) -> str:
    return _core.to_xml(  # type: ignore[no-any-return]
        tag_name, content, [(k, str(v)) for k, v in (attrs or {}).items()]
    )
