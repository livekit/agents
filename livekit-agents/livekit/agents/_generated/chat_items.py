# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/chat_context.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import TypedDict

from .. import _bindings, utils

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContent, ChatRole
    from .tool_context import Tool


class ImageContent(BaseModel):
    """ImageContent is used to input images into the ChatContext on supported LLM providers / plugins.

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
    """Unique identifier for the image"""
    type: Literal["image_content"] = "image_content"
    image: str | rtc.VideoFrame
    """Either a string URL or a VideoFrame object"""
    inference_width: int | None = None
    """Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)"""
    inference_height: int | None = None
    """Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)"""
    inference_detail: Literal["auto", "high", "low"] = "auto"
    """Detail parameter for LLM provider, if supported.

    Currently only supported by OpenAI (see https://platform.openai.com/docs/guides/vision?lang=node#low-or-high-fidelity-image-understanding)
    """
    mime_type: str | None = None
    """MIME type of the image"""
    _cache: dict[Any, Any] = PrivateAttr(default_factory=dict)


class AudioContent(BaseModel):
    type: Literal["audio_content"] = "audio_content"
    frame: list[rtc.AudioFrame]
    transcript: str | None = None


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


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["message"] = "message"
    role: ChatRole
    content: list[ChatContent]
    interrupted: bool = False
    transcript_confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    metrics: MetricsReport = Field(default_factory=lambda: MetricsReport())
    created_at: float = Field(default_factory=time.time)
    hash: bytes | None = Field(default=None, deprecated="hash is deprecated")

    @property
    def text_content(self) -> str | None:
        """Returns a string of all text content in the message, with LiveKit's
        expressive ``<expr/>`` tags removed from assistant messages.

        Multiple text content items will be joined by a newline.
        Use :attr:`raw_text_content` for the exact model-facing content.
        """
        return _bindings.chat_message_text_content(self)

    @property
    def raw_text_content(self) -> str | None:
        """Returns a string of all text content in the message, exactly as generated
        (assistant messages may contain expressive ``<expr/>`` tags).

        Multiple text content items will be joined by a newline.
        """
        return _bindings.chat_message_raw_text_content(self)


class FunctionCall(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call"] = "function_call"
    call_id: str
    arguments: str
    name: str
    created_at: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)
    """Extra data for this function call. Can include provider-specific data
    (e.g., extra["google"] for thought signatures).
    """
    group_id: str | None = None
    """Optional group ID for parallel function calls. When multiple function calls
    should be grouped together (e.g., parallel tool calls from a single API response),
    set this to a shared value. If not set, falls back to using id for grouping.
    """


class FunctionCallOutput(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["function_call_output"] = "function_call_output"
    name: str = ""
    call_id: str
    output: str
    is_error: bool
    created_at: float = Field(default_factory=time.time)
    reply_required: bool = True
    """Whether the model should answer once it receives this output (read by realtime models only)."""


class AgentHandoff(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_handoff"] = "agent_handoff"
    old_agent_id: str | None = None
    new_agent_id: str
    created_at: float = Field(default_factory=time.time)


class AgentConfigUpdate(BaseModel):
    id: str = Field(default_factory=lambda: utils.shortuuid("item_"))
    type: Literal["agent_config_update"] = "agent_config_update"
    instructions: str | None = None
    tools_added: list[str] | None = None
    tools_removed: list[str] | None = None
    created_at: float = Field(default_factory=time.time)
    _tools: list[Tool] = PrivateAttr(default_factory=list)
    """Full tool definitions (in-memory only, not serialized)."""


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
ImageContent.__module__ = "livekit.agents.llm.chat_context"
AudioContent.__module__ = "livekit.agents.llm.chat_context"
MetricsMetadata.__module__ = "livekit.agents.llm.chat_context"
MetricsReport.__module__ = "livekit.agents.llm.chat_context"
ChatMessage.__module__ = "livekit.agents.llm.chat_context"
FunctionCall.__module__ = "livekit.agents.llm.chat_context"
FunctionCallOutput.__module__ = "livekit.agents.llm.chat_context"
AgentHandoff.__module__ = "livekit.agents.llm.chat_context"
AgentConfigUpdate.__module__ = "livekit.agents.llm.chat_context"
