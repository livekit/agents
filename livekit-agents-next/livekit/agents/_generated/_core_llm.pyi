# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from livekit.agents._core_view import AvailabilityObserver, Duplex
from livekit.agents.llm import LLM
from livekit.agents.llm.mcp import _Progress, _Streams
from livekit.agents.llm.realtime import RealtimeModel
from livekit.agents.llm.utils import SerializedImage
from livekit.agents.tts import TTS
from livekit.agents.types import APIConnectOptions
from livekit.rtc import AudioFrame

from .._core import ChatItem, FallbackTtsStream
from .mcp import MCPToolOptions
from .mcp_session import McpCleanup
from .realtime_options import NativeRealtimeConfig, RealtimeOptions
from .tool_context import DuplicateMode, DuplicateScope

def chat_ctx_copy_plan(
    facts: Sequence[tuple[str, str, str | None, int, str | None, float, str | None]],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_empty_message: bool = False,
    exclude_handoff: bool = False,
    exclude_config_update: bool = False,
    tools: Sequence[str] | None = None,
) -> list[int]:
    """`ChatContext.copy` as an index plan over the host's own items."""

def chat_ctx_truncate_plan(
    facts: Sequence[tuple[str, str, str | None, int, str | None, float, str | None]],
    max_items: int,
) -> list[int] | None:
    """`ChatContext.truncate` as an index plan; `None` when under the limit."""

def chat_ctx_merge_plan(
    mine: Sequence[tuple[str, str, str | None, int, str | None, float, str | None]],
    theirs: Sequence[tuple[str, str, str | None, int, str | None, float, str | None]],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_config_update: bool = False,
) -> list[tuple[int, int]]:
    """`ChatContext.merge` as `(index, j)` insertions."""

def remove_instructions(item_ids: Sequence[str]) -> list[int]:
    """`remove_instructions(chat_ctx)` as a plan: the positions of the rendered prompt's items,
    ascending, given the items' ids.
    """

def compute_chat_ctx_diff(
    old: Sequence[ChatItem],
    new: Sequence[ChatItem],
) -> tuple[list[str], list[tuple[str | None, str]], list[tuple[str | None, str]]]:
    """The operations turning ``old`` into ``new``, as ``(to_remove, to_create, to_update)``; an
    update patches the item in place, keeping its server-side position.
    """

class FixedGate:
    """Opens on output that stands out from a silence the plugin already knows."""
    def __init__(
        self,
        silence: float,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
    ) -> None: ...
    def update(self, frame: AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""

class AdaptiveNoiseGate:
    """Opens on output that stands out from the model's own silence: the floor is the quietest
    `min_silence_duration` stretch it produced while not speaking, within `window`.
    """
    def __init__(
        self,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
        window: float = 10.0,
    ) -> None: ...
    def update(self, frame: AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""

class DuplexRealtimeModel:
    """`DuplexRealtimeAdapter`'s engine, the session's realtime model when handed to it."""
    def __init__(self, host: Duplex, *, audio_timeout: float = 0.8) -> None:
        """`audio_timeout`: how long to wait for the next frame before the output counts as finished."""
    def capabilities(self) -> dict[str, object]:
        """The realtime capabilities the model runs under."""

class FallbackLlm:
    """The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks."""
    def __init__(
        self,
        llms: Sequence[LLM[Any]],
        *,
        attempt_timeout: float = 5.0,
        max_retry_per_llm: int = 0,
        retry_interval: float = 0.5,
        sticky: bool = False,
        on_availability: AvailabilityObserver | None = None,
    ) -> None: ...
    @property
    def active_index(self) -> int: ...
    async def aclose(self) -> None: ...
    async def chat(
        self,
        chat_ctx_host: object,
        *,
        tools: Sequence[dict[str, object]],
        tools_host: object | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackLlmChunks: ...

class FallbackLlmChunks:
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

class FallbackTts:
    """`tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python."""
    def __init__(
        self,
        ttss: Sequence[TTS[Any]],
        *,
        max_retry_per_tts: int = 2,
        sample_rate: int | None = None,
        on_availability: AvailabilityObserver | None = None,
    ) -> None: ...
    @property
    def active_index(self) -> int: ...
    @property
    def capabilities(self) -> dict[str, object]: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    async def aclose(self) -> None: ...
    async def synthesize(
        self,
        text: str,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackTtsStream: ...
    async def stream(self, conn_options: APIConnectOptions | None = None) -> FallbackTtsStream: ...

class TtsTokenLog:
    """The pushed-token log as the host's `SynthesizeStream` keeps it (`_core.TtsTokenLog`): the text
    a recovering stream replays.

    `self._pushed_tokens` plus the live child sink. The text leg may not drop (§1.2d), so nothing
    here is lossy or bounded.
    """
    def __init__(self) -> None: ...
    def push(self, text: str) -> None: ...
    def flush(self) -> None: ...
    def end(self) -> None: ...
    def snapshot(self) -> tuple[list[str | None], bool]: ...

class InferenceLlm:
    """The gateway LLM behind `inference.LLM`. Its options are shared by every view
    (`identified`); each view's requests carry its own job's identification.
    """
    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str,
        api_key: str,
        api_secret: str,
        inference_class: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> None:
        """A client of `model` on the gateway at `base_url`. Starts nothing."""
    def identified(
        self,
        headers: Sequence[tuple[str, str]],
        reporter: LLM[Any] | None = None,
    ) -> InferenceLlm:
        """This client as a session of the job `headers` (`get_inference_headers()`) identifies
        holds it: the same options, its requests carrying that job. `reporter` is the host
        `llm.LLM` whose `error` events report the attempts this view retries.
        """
    @property
    def model(self) -> str: ...
    @property
    def extra_kwargs(self) -> dict[str, object]:
        """The persistent request options, a copy."""
    def update_options(
        self,
        *,
        model: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Read by the next `chat()`. `extra_kwargs` replaces the persistent options."""
    def chat(
        self,
        items: Sequence[ChatItem],
        *,
        tools: Sequence[dict[str, object]],
        conn_options: APIConnectOptions,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict[str, object] | None = None,
        response_format: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        headers: Sequence[tuple[str, str]] | None = None,
    ) -> InferenceLlmChunks:
        """Opens a completion of `items`; no I/O until the first pull. `headers` identify the
        caller's job, read in its context.
        """

class InferenceLlmChunks:
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

class LlmChoiceParser:
    """Accumulates the streamed choices of one OpenAI-compatible completion into chunks, filtering
    the model's thinking tags: `inference.LLMStream` over a plugin's own client.
    """
    def __init__(self, model: str) -> None: ...
    def parse(self, id: str, choice: object) -> dict[str, object] | None:
        """The chunk `choice` (one streamed choice of completion `id`) completes, if any."""

def llm_usage_chunk(chunk: object) -> dict[str, object] | None:
    """The usage chunk of a streamed completion chunk, when it reports usage."""

class McpSession:
    """An SDK session whose owner cancels and joins independently of request borrowers."""
    def __init__(self, client_session_timeout_seconds: float) -> None: ...
    @property
    def connected(self) -> bool: ...
    def set_allowed_tools(self, allowed: Sequence[str]) -> None: ...
    async def connect_http(
        self,
        url: str,
        headers: Sequence[tuple[str, str]],
        transport: str,
        timeout: float,
        sse_read_timeout: float,
    ) -> None: ...
    async def connect_stdio(
        self,
        command: str,
        args: Sequence[str],
        env: Sequence[tuple[str, str]],
        cwd: str | None,
    ) -> None: ...
    async def connect_host(self, transport: _Streams, host_context: object | None) -> None: ...
    async def list_tools(self) -> object: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        progress: _Progress | None,
        host_context: object | None,
    ) -> object: ...
    async def close(self) -> None: ...
    @property
    def cleanup(self) -> McpCleanup: ...

def resolve_mcp_tool_options(config: MCPToolOptions | None) -> dict[str, object]: ...

def mcp_raw_schema(
    name: str,
    description: str | None,
    input_schema: dict[str, object],
    meta: dict[str, object] | None,
) -> object: ...

def mcp_plan_setup(toolset_initialized: bool, server_connected: bool) -> str: ...

def mcp_http_transport(url: str, transport_type: str | None) -> str: ...

def mcp_render_result(tool_name: str, result: object) -> str: ...

def _core_mcp_timeout_check(field: str, seconds: float) -> None:
    """Validate public MCP constructor arguments before a transport can be opened."""

async def to_provider_format(
    format: str,
    items: Sequence[ChatItem],
    *,
    images: Mapping[str, SerializedImage] | None = None,
    inject_dummy_user_message: bool = True,
    inject_trailing_user_message: bool = False,
    thought_signatures: Mapping[str, bytes | bytearray | memoryview | list[int]] | None = None,
) -> tuple[list[object], object]: ...

def convert_mid_conversation_instructions(
    items: Sequence[ChatItem],
    *,
    role: str,
    template: str,
) -> list[tuple[int, ChatItem | None]]: ...

def to_fnc_ctx(
    format: str,
    tools: Sequence[object],
    *,
    tool_behavior: str | None = None,
    use_parameters_json_schema: bool = False,
) -> list[object]: ...

def to_strict_json_schema(
    schema: object,
    *,
    null_sentinel_for_defaults: bool = False,
) -> dict[str, object]: ...

def resolve_ref(*, root: dict[str, object], ref: str) -> dict[str, object]: ...

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

def duplicate_key(
    name: str,
    args: object | None,
    scope: DuplicateScope,
) -> tuple[str, str | None]: ...

def decide_duplicate(
    name: str,
    mode: DuplicateMode,
    confirm_duplicate: bool | None,
    in_flight: Sequence[tuple[str, bool]],
) -> tuple[str, object | None]: ...

def inject_confirm_duplicate(parameters: object) -> dict[str, object]: ...

class HostRealtimeModel:
    """A plugin's realtime model as the core's own: its session opens when the driver first asks and
    is the same when asked again.
    """
    def __init__(self, model: RealtimeModel[Any]) -> None: ...

class NativeRealtimeModel:
    def __init__(self, config: NativeRealtimeConfig) -> None: ...
    @property
    def provider(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def label(self) -> str: ...
    def capabilities(self) -> dict[str, object]: ...
    def update_options(self, options: RealtimeOptions) -> None: ...
    def bind_identification(self, headers: Sequence[tuple[str, str]]) -> None: ...

class RealtimeFallbackModel:
    """The generated facade holds native resources; host provider objects use the shared bridge."""
    def __init__(
        self,
        models: Sequence[object],
        *,
        cooldown: float = 10.0,
        regenerate_on_swap: bool = True,
        on_availability: object | None = None,
    ) -> None: ...
    @property
    def active_index(self) -> int: ...
    def capabilities(self) -> dict[str, object]: ...
    async def restart(self, *, switch_model: bool = False) -> None: ...

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
