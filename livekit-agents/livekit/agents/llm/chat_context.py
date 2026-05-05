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

import textwrap
import time
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, overload

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
from typing_extensions import TypedDict

from livekit import rtc

from .. import utils
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils.misc import is_given
from . import _provider_format

if TYPE_CHECKING:
    from ..llm import LLM, Tool, Toolset


class Instructions:
    """Instructions with optional modality-specific additions.

    Construction::

        # Simple — same instructions for all modalities
        Instructions("You are a helpful assistant.")

        # With modality-specific additions
        Instructions(
            "You are a helpful assistant.",
            audio="Keep responses short for voice.",
            text="Use markdown formatting.",
        )

    Rendering::

        instr.render()                              # → common text
        instr.render(modality="audio")               # → common + audio addition
        instr.render(modality="text", name="Alex")   # → common + text, with {name} filled
    """

    def __init__(
        self,
        common: str = "",
        *,
        audio: str | None = None,
        text: str | None = None,
    ) -> None:
        self.common = common
        self.audio = audio
        self.text = text

    def render(
        self,
        *,
        modality: Literal["audio", "text"] | None = None,
        data: dict[str, object] | None = None,
    ) -> str:
        """Render instructions to a plain string.

        Args:
            modality: If given, appends the modality-specific addition to the common text.
            data: Template variables to fill. Missing placeholders log a warning
                and are replaced with empty strings.
        """
        parts = [self.common]
        if modality is not None:
            addition = self.audio if modality == "audio" else self.text
            if addition:
                parts.append(addition)

        result = "\n\n".join(p for p in parts if p)

        if data:
            result = utils.misc.safe_render(result, data)

        return result

    @staticmethod
    def resolve_template(template: str, **kwargs: object) -> Instructions:
        """Fill a template string, producing an ``Instructions`` with modality variants.

        If any kwarg value is an ``Instructions`` object, its ``common``/``audio``/``text``
        parts are substituted into the matching variant of the result. This is used by
        workflow tasks to build modality-aware instructions from a single template.
        """
        any_instructions = any(isinstance(v, Instructions) for v in kwargs.values())
        if any_instructions:
            common_kw: dict[str, object] = {
                k: str(v) if isinstance(v, Instructions) else v for k, v in kwargs.items()
            }
            audio_kw: dict[str, object] = {
                k: (v.audio or str(v)) if isinstance(v, Instructions) else v
                for k, v in kwargs.items()
            }
            text_kw: dict[str, object] = {
                k: (v.text or str(v)) if isinstance(v, Instructions) else v
                for k, v in kwargs.items()
            }
            return Instructions(
                common=utils.misc.safe_render(template, common_kw),
                audio=utils.misc.safe_render(template, audio_kw),
                text=utils.misc.safe_render(template, text_kw),
            )
        else:
            rendered = utils.misc.safe_render(template, kwargs)
            return Instructions(common=rendered)

    def __str__(self) -> str:
        return self.common

    def __repr__(self) -> str:
        return f"Instructions({self.common!r})"

    def __hash__(self) -> int:
        return hash((self.common, self.audio, self.text))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Instructions):
            return (
                self.common == other.common
                and self.audio == other.audio
                and self.text == other.text
            )
        if isinstance(other, str):
            return self.common == other
        return NotImplemented


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
    """

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

    llm_metadata: MetricsMetadata
    tts_metadata: MetricsMetadata
    stt_metadata: MetricsMetadata


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["message"] = "message"
    role: ChatRole
    content: list[ChatContent]
    interrupted: bool = False
    transcript_confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    llm_output: BaseModel | None = Field(default=None, exclude=True)
    """Parsed structured output from the LLM when ``llm_output_format`` is set on the Agent."""
    metrics: MetricsReport = Field(default_factory=lambda: MetricsReport())
    created_at: float = Field(default_factory=time.time)
    hash: bytes | None = Field(default=None, deprecated="hash is deprecated")

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


ChatContent: TypeAlias = ImageContent | AudioContent | str


class FunctionCall(BaseModel):
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


class FunctionCallOutput(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call_output"] = Field(default="function_call_output")
    name: str = Field(default="")
    call_id: str
    output: str
    is_error: bool
    created_at: float = Field(default_factory=time.time)


class AgentHandoff(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_handoff"] = Field(default="agent_handoff")
    old_agent_id: str | None = None
    new_agent_id: str
    created_at: float = Field(default_factory=time.time)


class AgentConfigUpdate(BaseModel):
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
        kwargs: dict[str, Any] = {}
        if is_given(id):
            kwargs["id"] = id
        if is_given(interrupted):
            kwargs["interrupted"] = interrupted
        if is_given(created_at):
            kwargs["created_at"] = created_at
        if is_given(metrics):
            kwargs["metrics"] = metrics
        if is_given(extra):
            kwargs["extra"] = extra

        if isinstance(content, Instructions):
            message = ChatMessage(role=role, content=[str(content)], **kwargs)
        elif isinstance(content, str):
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
        exclude_handoff: bool = False,
        exclude_config_update: bool = False,
        tools: NotGivenOr[Sequence[Tool | Toolset | str]] = NOT_GIVEN,
    ) -> ChatContext:
        items = []

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
                else:
                    # TODO(theomonnom): other tools
                    continue

        valid_tools = set(get_tool_names(tools)) if tools else set()
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

            if exclude_handoff and item.type == "agent_handoff":
                continue

            if exclude_config_update and item.type == "agent_config_update":
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
        Preserves the first instruction message (system/developer) by adding it back
        to the beginning.
        """

        if len(self._items) <= max_items:
            return self

        instructions = next(
            (
                item
                for item in self._items
                if item.type == "message" and item.role in ("system", "developer")
            ),
            None,
        )

        new_items = self._items[-max_items:]

        # chat_ctx shouldn't start with function_call or function_call_output
        while new_items and new_items[0].type in [
            "function_call",
            "function_call_output",
        ]:
            new_items.pop(0)

        if instructions and not any(item.id == instructions.id for item in new_items):
            new_items.insert(0, instructions)

        self._items[:] = new_items
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

            if exclude_config_update and item.type == "agent_config_update":
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
        exclude_metrics: bool = False,
        exclude_config_update: bool = False,
    ) -> dict[str, Any]:
        items: list[ChatItem] = []
        for item in self.items:
            if exclude_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                continue

            if exclude_config_update and item.type == "agent_config_update":
                continue

            if item.type == "message":
                item = item.model_copy()
                if exclude_image:
                    item.content = [c for c in item.content if not isinstance(c, ImageContent)]
                if exclude_audio:
                    item.content = [c for c in item.content if not isinstance(c, AudioContent)]

            items.append(item)

        exclude_fields: set[str] = set()
        if exclude_timestamp:
            exclude_fields.add("created_at")
        if exclude_metrics:
            exclude_fields.add("metrics")

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
        self,
        format: Literal["openai", "openai.responses"],
        *,
        inject_dummy_user_message: bool = True,
    ) -> tuple[list[dict], Literal[None]]: ...

    @overload
    def to_provider_format(
        self,
        format: Literal["google"],
        *,
        inject_dummy_user_message: bool = True,
        thought_signatures: dict[str, bytes] | None = None,
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
        self, format: Literal["mistralai"]
    ) -> tuple[list[dict], _provider_format.mistralai.MistralFormatData]: ...

    @overload
    def to_provider_format(self, format: str, **kwargs: Any) -> tuple[list[dict], Any]: ...

    def to_provider_format(
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
        kwargs["inject_dummy_user_message"] = inject_dummy_user_message

        if format == "openai":
            return _provider_format.openai.to_chat_ctx(self, **kwargs)
        elif format == "openai.responses":
            return _provider_format.openai.to_responses_chat_ctx(self, **kwargs)
        elif format == "google":
            return _provider_format.google.to_chat_ctx(self, **kwargs)
        elif format == "aws":
            return _provider_format.aws.to_chat_ctx(self, **kwargs)
        elif format == "anthropic":
            return _provider_format.anthropic.to_chat_ctx(self, **kwargs)
        elif format == "mistralai":
            return _provider_format.mistralai.to_conversations_ctx(self)
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

    def _upsert_item(self, item: ChatItem, *, allow_type_mismatch: bool = False) -> None:
        """Update an item with the same ID if it exists, otherwise append it."""
        idx = self.index_by_id(item.id)
        if idx is not None:
            if not allow_type_mismatch and item.type != self._items[idx].type:
                raise ValueError(f"Item type mismatch: {item.type} != {self._items[idx].type}")
            self._items[idx] = item
        else:
            self._items.append(item)

    async def _summarize(
        self,
        llm_v: LLM,
        *,
        keep_last_turns: int = 2,
    ) -> ChatContext:
        # Split self.items into head/tail. Walk backward, counting only
        # user/assistant ChatMessages toward the keep_last_turns budget (each
        # turn = one user + one assistant message, so budget = keep_last_turns * 2).
        # Everything from the split point onward — including any interleaved
        # FunctionCall/FunctionCallOutput items — is preserved as-is in the tail.
        msg_budget = keep_last_turns * 2
        split_idx = len(self.items)

        if msg_budget > 0:
            msg_count = 0
            for i in range(len(self.items) - 1, -1, -1):
                item = self.items[i]
                if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
                    msg_count += 1
                    if msg_count >= msg_budget:
                        split_idx = i
                        break
            else:
                # Not enough messages to fill the budget — nothing to summarize
                return self

        if split_idx == 0:
            return self

        head_items, tail_items = self.items[:split_idx], self.items[split_idx:]

        # Build summarization input from head_items only.
        to_summarize: list[ChatMessage | FunctionCall | FunctionCallOutput] = []
        for item in head_items:
            if isinstance(item, ChatMessage):
                if item.role not in ("user", "assistant"):
                    continue
                if item.extra.get("is_summary") is True:  # avoid making summary of summaries
                    continue

                text = (item.text_content or "").strip()
                if text:
                    to_summarize.append(item)
            elif isinstance(item, (FunctionCall, FunctionCallOutput)):
                to_summarize.append(item)

        if not to_summarize:
            return self

        # Render items to XML format and collect the contents.
        contents: list[str] = []
        for m in to_summarize:
            if isinstance(m, (FunctionCall, FunctionCallOutput)):
                contents.append(_function_call_item_to_message(m).text_content or "")
            else:
                contents.append(to_xml(m.role, (m.text_content or "").strip()))

        source_text = "\n".join(contents).strip()

        if not source_text:
            return self

        chat_ctx = ChatContext()
        chat_ctx.add_message(
            role="system",
            content=textwrap.dedent("""\
                Compress older conversation history into a short, faithful summary.

                The conversation is formatted as XML. Here is how to read it:
                - <user>…</user>  — something the user said.
                - <assistant>…</assistant>  — something the assistant said.
                - <function_call name="…" call_id="…">…</function_call>  — the assistant invoked an action.
                - <function_call_output name="…" call_id="…">…</function_call_output>  — the result of that \
                action. May contain <error>…</error> if it failed.

                Guidelines:
                - Distill the *information learned* from function call outputs into the summary. \
                Do not mention that a tool/function was called — just preserve the knowledge gained.
                - Focus on: user goals, constraints, decisions, key facts, preferences, entities, \
                and any pending or unresolved tasks.
                - Omit greetings, filler, and chit-chat.
                - Be concise."""),
        )
        chat_ctx.add_message(
            role="user",
            content=f"Conversation to summarize:\n\n{source_text}",
        )

        chunks: list[str] = []
        async with llm_v.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    chunks.append(chunk.delta.content)

        summary = "".join(chunks).strip()
        if not summary:
            return self

        # Rebuild self._items. From head_items, keep only structural
        # items (system messages, agent handoffs, config updates, prior
        # summaries) — everything summarizable is replaced by the summary.
        # Tail items are appended as-is.
        preserved: list[ChatItem] = []
        for it in head_items:
            if isinstance(it, ChatMessage) and it.role in ("user", "assistant"):
                continue
            if isinstance(it, (FunctionCall, FunctionCallOutput)):
                continue
            preserved.append(it)

        self._items = preserved

        created_at_hint = (
            (tail_items[0].created_at - 1e-6) if tail_items else (head_items[-1].created_at + 1e-6)
        )
        self.add_message(
            role="assistant",
            content=to_xml("chat_history_summary", summary),
            created_at=created_at_hint,
            extra={"is_summary": True},
        )

        self._items.extend(tail_items)

        return self

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

        for a, b in zip(self.items, other.items, strict=False):
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


def _to_attrs_str(attrs: dict[str, Any] | None = None) -> str | None:
    if attrs:
        return " ".join([f'{k}="{v}"' for k, v in attrs.items()])
    return None


def to_xml(
    tag_name: str,
    content: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> str:
    attrs_str = _to_attrs_str(attrs)

    if content:
        return "\n".join(
            [
                f"<{tag_name} {attrs_str}>" if attrs_str else f"<{tag_name}>",
                content,
                f"</{tag_name}>",
            ]
        )
    else:
        return f"<{tag_name} {attrs_str} />" if attrs_str else f"<{tag_name} />"


def _function_call_item_to_message(item: FunctionCall | FunctionCallOutput) -> ChatMessage:
    if isinstance(item, FunctionCall):
        return ChatMessage(
            role="user",
            content=[
                to_xml(
                    "function_call",
                    item.arguments,
                    attrs={
                        "name": item.name,
                        "call_id": item.call_id,
                    },
                )
            ],
            created_at=item.created_at,
            extra={"is_function_call": True},
        )
    elif isinstance(item, FunctionCallOutput):
        return ChatMessage(
            role="assistant",
            content=[
                to_xml(
                    "function_call_output",
                    item.output if not item.is_error else to_xml("error", item.output),
                    attrs={
                        "call_id": item.call_id,
                        "name": item.name,
                    },
                )
            ],
            created_at=item.created_at,
            extra={"is_function_call_output": True},
        )
