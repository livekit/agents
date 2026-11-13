# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Self

from livekit.agents.llm import LLM, AudioGate, Toolset
from livekit.agents.llm.duplex import DuplexModel
from livekit.agents.llm.mcp import MCPToolset
from livekit.agents.llm.realtime import RealtimeModel
from livekit.agents.llm.tool_context import ToolChoice
from livekit.agents.tts import TTS
from livekit.agents.voice.events import RunContext

from ..types import NotGivenOr
from .chat_diff import DiffOps, MergeInsertion
from .chat_items import ChatItem, ChatRole
from .connect import APIConnectOptions
from .duplex import DuplexInputItem
from .llm import ChatChunk, SerializedImage
from .llm_fallback import HostChatContext, HostTools, RealtimeLeg
from .llm_provider_options import ProviderEndpoint, ResponseFormat
from .mcp import MCPToolOptions, ResolvedToolOptions
from .metrics import RealtimeModelMetrics
from .provider_data import TOOL_BEHAVIOR, ProviderFormat, ToolSchema
from .provider_format import ConvertedItem, ProviderMessages
from .realtime import RealtimeCapabilities
from .realtime_options import (
    AudioTranscription,
    InferenceGateway,
    NoiseReduction,
    RealtimeProvider,
    Reasoning,
    Tracing,
    Truncation,
    TurnDetection,
    Voice,
)
from .realtime_wire import Modalities
from .rtc_frames import AudioFrame
from .tool_context import RawFunctionDescription
from .tool_report import ToolReport, ToolReturn
from .tts import TTSCapabilities

def _core_payload_to_item(payload: dict[str, object]) -> ChatItem:
    """Decode a persisted protobuf ChatItem with the shared history codec."""

def chat_ctx_copy_plan(
    items: Sequence[ChatItem],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_empty_message: bool = False,
    exclude_handoff: bool = False,
    exclude_config_update: bool = False,
    tools: Sequence[str] | None = None,
) -> list[int]:
    """`ChatContext.copy` as an index plan over the host's own items."""

def chat_ctx_truncate_plan(items: Sequence[ChatItem], max_items: int) -> list[int] | None:
    """`ChatContext.truncate` as an index plan; `None` when under the limit."""

def chat_ctx_merge_plan(
    mine: Sequence[ChatItem],
    theirs: Sequence[ChatItem],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_config_update: bool = False,
) -> list[MergeInsertion]:
    """`ChatContext.merge` as insertions, in order."""

def remove_instructions(item_ids: Sequence[str]) -> list[int]:
    """`remove_instructions(chat_ctx)` as a plan: the positions of the rendered prompt's items,
    ascending, given the items' ids.
    """

def compute_chat_ctx_diff(old: Sequence[ChatItem], new: Sequence[ChatItem]) -> DiffOps:
    """The operations turning ``old`` into ``new``."""

class FixedGate:
    """Opens on output that stands out from a silence the plugin already knows."""
    def __new__(
        cls,
        silence: float,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
    ) -> Self: ...
    def update(self, frame: AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""

class AdaptiveNoiseGate:
    """Opens on output that stands out from the model's own silence: the floor is the quietest
    `min_silence_duration` stretch it produced while not speaking, within `window`.
    """
    def __new__(
        cls,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
        window: float = 10.0,
    ) -> Self: ...
    def update(self, frame: AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""

class DuplexInput:
    """`_core.DuplexInput`: a duplex session's input. The engine pushes the caller's audio, the plugin
    pulls it; a plugin more than 250 ms behind skips to the newest audio after a gap.
    """
    def __new__(cls) -> Self: ...
    def push(self, frame: AudioFrame) -> None: ...
    def mute(self, muted: bool) -> None:
        """Queued after the audio pushed so far; raises when the plugin stopped pulling."""
    def flush(self) -> None:
        """Drops the queued audio: what is pulled next is the newest."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> DuplexInputItem: ...

class DuplexRealtimeModel:
    """`DuplexRealtimeAdapter`'s engine, the session's realtime model when handed to it."""
    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        gate: Callable[[], AudioGate] | None = None,
        audio_timeout: float = 0.8,
    ) -> None:
        """Each session's gate is `gate`'s, else the model's own, else the adaptive gate.
        `audio_timeout`: how long to wait for the next frame before the output counts as finished.
        """
    @property
    def capabilities(self) -> RealtimeCapabilities:
        """The realtime capabilities the model runs under."""

class FallbackLlm:
    """The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks."""
    def __init__(
        self,
        llm: Sequence[LLM[Any]],
        *,
        attempt_timeout: float = 5.0,
        max_retry_per_llm: int = 0,
        retry_interval: float = 0.5,
        sticky: bool = False,
    ) -> None:
        """`llm` in order; `observer` hears the chain's availability and failures."""
    @property
    def _core_active_index(self) -> int:
        """Which entry most recently served, as a position in the host's list."""
    async def aclose(self) -> None: ...
    async def _core_native_chat(
        self,
        *,
        chat_ctx: HostChatContext,
        tools: HostTools | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackLlmChunks:
        """A completion of the host `chat_ctx`, offering `tools`; the chain picks an entry on the
        first pull.
        """

class FallbackLlmChunks:
    async def recv(self) -> ChatChunk | None: ...
    async def aclose(self) -> None: ...

class FallbackTts:
    """`tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python."""
    def __init__(
        self,
        tts: Sequence[TTS[Any]],
        *,
        max_retry_per_tts: int = 2,
        attempt_timeout: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """`tts` in order; `sample_rate` defaults to the highest of theirs, the others resampled.
        `observer` hears the chain's availability and failures.
        """
    @property
    def _core_active_index(self) -> int:
        """Which entry most recently served, as a position in the host's list."""
    @property
    def capabilities(self) -> TTSCapabilities: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    async def aclose(self) -> None: ...

class RealtimeFallbackModel:
    """`llm.RealtimeModelFallbackAdapter`'s engine: the generated facade holds native resources; host
    provider objects use the shared bridge.
    """
    def __init__(
        self,
        models: Sequence[RealtimeLeg],
        *,
        cooldown: float = 10.0,
        attempt_timeout: float | None = None,
        regenerate_on_swap: bool = True,
    ) -> None:
        """`models` in order; `observer` hears the chain's availability and failures."""
    @property
    def _core_active_index(self) -> int:
        """Which entry most recently served, as a position in the host's list."""
    @property
    def capabilities(self) -> RealtimeCapabilities:
        """The members' capabilities, merged: the pipeline-shaping ones agree, the others are ANDed."""
    async def _core_restart(self, *, switch_model: bool = False) -> None: ...

class InferenceLlm:
    """The gateway LLM behind `inference.LLM`. One instance serves every job it is handed to; each
    request carries the identity of the job it runs for.
    """
    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        endpoint: ProviderEndpoint | None = None,
    ) -> None:
        """A client of `model`. Starts nothing. On the LiveKit Inference gateway by default, its
        URL and key pair resolved from the given values, the environment and the server's
        settings; with `endpoint`, an OpenAI-compatible provider called with its own key and no
        LiveKit header.
        """
    @property
    def model(self) -> str: ...
    @property
    def _core_extra_kwargs(self) -> dict[str, object]:
        """The persistent request options, a copy."""
    @property
    def _core_inference_class(self) -> str | None:
        """The scheduling class of its requests."""
    def _core_update_options(
        self,
        *,
        model: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        endpoint: ProviderEndpoint | None = None,
    ) -> None:
        """Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs, and
        ``endpoint`` the provider endpoint.
        """
    def _core_native_chat(
        self,
        *,
        chat_ctx: HostChatContext,
        tools: HostTools | None = None,
        conn_options: APIConnectOptions | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        inference_class: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> InferenceLlmChunks:
        """Opens a completion of `chat_ctx`; no I/O until the first pull. Its requests carry the
        identity of the job `cx` runs for and `inference_class`, else the client's class.
        """

class InferenceLlmChunks:
    async def recv(self) -> ChatChunk | None: ...
    async def aclose(self) -> None: ...

class LlmChoiceParser:
    """Accumulates the streamed choices of one OpenAI-compatible completion into chunks:
    `inference.LLMStream` over a plugin's own client. The model's own syntax (reasoning, template
    tokens) is taken out above every provider, not here.
    """
    def __new__(cls) -> Self: ...
    def parse(self, id: str, choice: object) -> ChatChunk | None:
        """The chunk `choice` (one streamed choice of completion `id`) completes, if any."""

def llm_usage_chunk(chunk: object) -> ChatChunk | None:
    """The usage chunk of a streamed completion chunk, when it reports usage."""

class McpSession:
    """An SDK session whose owner cancels and joins independently of request borrowers."""
    def __new__(
        cls,
        init_timeout: float,
        request_timeout: float | None,
        read_timeout: float | None = None,
    ) -> Self:
        """`init_timeout` bounds connecting, the handshake and a listing; `request_timeout` one
        request without a reply or progress, `None` for no limit; `read_timeout` an idle HTTP
        reply stream.
        """
    @property
    def connected(self) -> bool: ...
    @property
    def retry_in(self) -> float:
        """Seconds until a failed session may reconnect; zero when it may now."""
    async def connect_http(
        self,
        url: str,
        headers: Mapping[str, str],
        auth: str | None,
    ) -> None: ...
    async def connect_stdio(
        self,
        command: str,
        args: Sequence[str],
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None: ...
    async def list_tools(self) -> object: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        timeout: float | None,
        progress: RunContext | None,
        host_context: object | None,
    ) -> object: ...
    async def close(self) -> None: ...

class _core_McpToolsetLifecycle:
    """An `MCPToolset`'s lifecycle (docs/tools.md "MCP servers"): one listing at a time; an
    unreachable server leaves the toolset without tools and is listed again in the background at
    the session's backoff until it answers; the connection closes [`MCP_IDLE_CLOSE_SECONDS`] after the
    last run released the toolset unless a run sets it up again first. `host` is the toolset,
    held only while a recovery or an idle close is pending.
    """
    def __new__(cls, server: str, session: McpSession) -> Self:
        """`server` names the server in logs; `session` owns the reconnect backoff."""
    async def setup(self, host: MCPToolset) -> None:
        """Disarm a pending idle close, then list the tools unless they are listed or a recovery
        is under way. A failed listing is logged and recovered in the background; a cancelled
        setup cancels its listing.
        """
    async def release(self, host: MCPToolset) -> None:
        """The last run released the toolset: stop a recovery (its listing cancelled and settled)
        and arm the idle close.
        """

def resolve_mcp_tool_options(config: MCPToolOptions | None) -> ResolvedToolOptions: ...

def mcp_raw_schema(
    name: str,
    description: str | None,
    input_schema: dict[str, object],
    meta: dict[str, object] | None,
) -> object: ...

def mcp_render_result(tool_name: str, result: object) -> str: ...

async def to_provider_format(
    format: ProviderFormat,
    items: Sequence[ChatItem],
    *,
    images: Mapping[str, SerializedImage] | None = None,
    inject_dummy_user_message: bool = True,
    inject_trailing_user_message: bool = False,
    thought_signatures: Mapping[str, bytes | bytearray | memoryview | list[int]] | None = None,
) -> ProviderMessages: ...

def convert_mid_conversation_instructions(
    items: Sequence[ChatItem],
    *,
    role: ChatRole,
    template: str,
) -> list[ConvertedItem]: ...

def to_fnc_ctx(
    format: ProviderFormat,
    tools: Sequence[ToolSchema],
    *,
    tool_behavior: TOOL_BEHAVIOR | None = None,
    use_parameters_json_schema: bool = False,
) -> list[object]: ...

def to_strict_json_schema(
    schema: object,
    *,
    null_sentinel_for_defaults: bool = False,
) -> dict[str, object]: ...

def inject_schema_defaults(value: object, schema: object, root: object | None = None) -> object: ...

def build_legacy_openai_schema(
    name: str,
    description: str | None,
    parameters: object,
    *,
    internally_tagged: bool = False,
) -> dict[str, object]: ...

def build_strict_openai_schema(
    name: str,
    description: str | None,
    parameters: object,
) -> dict[str, object]: ...

def to_openai_response_format(name: str, schema: object) -> dict[str, object]: ...

def parse_function_arguments(raw: str) -> dict[str, object]: ...

def inject_confirm_duplicate(parameters: object) -> dict[str, object]: ...

def _core_realtime_connection_metrics(
    model: str,
    provider: str,
    acquire_time: float,
) -> RealtimeModelMetrics:
    """`_report_connection_acquired`: the connection's timing as a `RealtimeModelMetrics` with zero
    usage, stamped now.
    """

class HostRealtimeModel:
    """A plugin's realtime model as the core's own: its session opens when the driver first asks and
    is the same when asked again.
    """
    def __new__(cls, model: RealtimeModel) -> Self: ...

class OpenAIRealtimeModel:
    def __init__(
        self,
        *,
        model: str | None = None,
        voice: Voice | None = None,
        modalities: Sequence[Modalities] | None = None,
        tool_choice: NotGivenOr[ToolChoice | None] = ...,
        base_url: str | None = None,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = ...,
        input_audio_noise_reduction: NotGivenOr[NoiseReduction | None] = ...,
        turn_detection: NotGivenOr[TurnDetection | None] = ...,
        speed: NotGivenOr[float] = ...,
        tracing: NotGivenOr[Tracing | None] = ...,
        truncation: NotGivenOr[Truncation | None] = ...,
        reasoning: NotGivenOr[Reasoning | None] = ...,
        max_response_output_tokens: NotGivenOr[int | None] = ...,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        azure_endpoint: str | None = None,
        entra_token: str | None = None,
        max_session_duration: NotGivenOr[timedelta | None] = ...,
        conn_options: APIConnectOptions | None = None,
        dialect: RealtimeProvider | None = None,
        label: str | None = None,
        gateway: InferenceGateway | None = None,
    ) -> None:
        """Not given, `model`, `voice` and the session options are the dialect's defaults. The key
        and base URL fall back to the dialect's environment: for Azure (`azure_deployment` or
        `entra_token`), `AZURE_OPENAI_API_KEY` and `{azure_endpoint}/openai`, `azure_endpoint`
        else `AZURE_OPENAI_ENDPOINT`; for OpenAI, `OPENAI_API_KEY` and `OPENAI_BASE_URL`, else
        OpenAI's. `dialect`, `label` and `gateway` are the provider class's own. One instance
        serves every job it is handed to; a LiveKit Inference connection carries the identity of
        the job it runs for.
        """
    @property
    def provider(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def label(self) -> str: ...
    @property
    def capabilities(self) -> RealtimeCapabilities: ...
    def _core_update_options(
        self,
        *,
        voice: NotGivenOr[Voice] = ...,
        turn_detection: NotGivenOr[TurnDetection | None] = ...,
        tool_choice: NotGivenOr[ToolChoice | None] = ...,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = ...,
        input_audio_noise_reduction: NotGivenOr[NoiseReduction | None] = ...,
        max_response_output_tokens: NotGivenOr[int | None] = ...,
        speed: NotGivenOr[float] = ...,
        tracing: NotGivenOr[Tracing | None] = ...,
        truncation: NotGivenOr[Truncation | None] = ...,
        reasoning: NotGivenOr[Reasoning | None] = ...,
    ) -> None:
        """The given options; every live session sends what changed."""

def chat_ctx_to_dict(
    items: Sequence[ChatItem],
    *,
    exclude_image: bool = True,
    exclude_audio: bool = True,
    exclude_timestamp: bool = True,
    exclude_function_call: bool = False,
    exclude_metrics: bool = False,
    exclude_config_update: bool = False,
    strip_markup: bool = False,
) -> dict[str, object]:
    """``{"items": [...]}``, each item's JSON dump. Media content needs its exclusion; with
    ``strip_markup`` the ``<expr/>`` dialect is stripped from assistant messages.
    """

def tool_report(value: ToolReturn) -> ToolReport:
    """What a tool's body returned or raised, classified for the core, which decides what the
    model reads and whether it replies.
    """

class ChatRequest:
    """The request as a host `LLM._core_chat` reads it: a `_core.ChatRequest` handle over a clone. The
    host objects cross as themselves, and the context as its `ChatContext.to_dict()` form, read only
    when the request carries no host context of its own.
    """
    @property
    def chat_ctx_host(self) -> object | None:
        """The host's canonical `llm.ChatContext`, when the request came from one."""
    @property
    def chat_ctx(self) -> dict[str, object]:
        """`ChatContext.to_dict()` of the request's context, timestamps kept (images and audio are not
        rendered): the context of a request the core minted itself.
        """
    @property
    def tool_names(self) -> list[str]:
        """The names of the tools this turn offers, in order: the host pool is narrowed to these."""
    @property
    def function_schemas(self) -> list[RawFunctionDescription]:
        """Canonical declarations of every function offered by this request, including tools
        executed by the core that have no callable object in the host pool.
        """
    @property
    def tools_host(self) -> object | None:
        """The host's live `list[Tool]` pool."""
    @property
    def tools_host_extra(self) -> list[object]:
        """The pools the session added on top of the agent's."""
    @property
    def parallel_tool_calls(self) -> bool | None: ...
    @property
    def tool_choice(self) -> ToolChoice | None: ...
    @property
    def extra(self) -> dict[str, object] | None: ...
    @property
    def inference_class(self) -> str | None: ...

def check_tool_names(names: Sequence[str]) -> None:
    """`names`, one per tool (the same tool listed twice is one tool): refuses the first name
    two tools share.
    """

class _core_ToolsetLifetime:
    def __new__(cls) -> Self: ...

class _core_ToolsetGroup:
    def __new__(cls, toolsets: Sequence[Toolset]) -> Self: ...
    async def start(self) -> None: ...
    async def aclose(self) -> None: ...

class _core_ToolsetScopes:
    def __new__(cls) -> Self: ...
    async def acquire(self, scope: int, toolsets: Sequence[Toolset]) -> int:
        """Zero is the whole-session scope; other IDs are stable host agent identities."""
    async def release(self, scope: int | None) -> None:
        """End the oldest outstanding activation of that agent, or every scope on shutdown."""
