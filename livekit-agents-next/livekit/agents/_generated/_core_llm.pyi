# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from typing import Any

from livekit.agents._core_view import AvailabilityObserver
from livekit.agents.llm import LLM
from livekit.agents.llm.mcp import _Progress, _Streams
from livekit.agents.types import APIConnectOptions

from .._core import ChatItem
from .mcp_session import McpCleanup
from .realtime_options import NativeRealtimeConfig, RealtimeOptions

class FallbackLlm:
    """The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks."""
    def __init__(
        self,
        llms: list[LLM[Any]],
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
        tools: list[dict[str, object]],
        tools_host: object | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackLlmChunks: ...

class FallbackLlmChunks:
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

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
        headers: list[tuple[str, str]],
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
        items: list[ChatItem],
        *,
        tools: list[dict[str, object]],
        conn_options: APIConnectOptions,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict[str, object] | None = None,
        response_format: object | None = None,
        extra_kwargs: dict[str, object] | None = None,
        headers: list[tuple[str, str]] | None = None,
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
    def set_allowed_tools(self, allowed: list[str]) -> None: ...
    def invalidate_cache(self) -> None: ...
    async def connect_http(
        self,
        url: str,
        headers: list[tuple[str, str]],
        transport: str,
        timeout: float,
        sse_read_timeout: float,
    ) -> None: ...
    async def connect_stdio(
        self,
        command: str,
        args: list[str],
        env: list[tuple[str, str]],
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

def resolve_mcp_tool_options(config: dict[str, object] | None) -> dict[str, object]: ...

def mcp_raw_schema(
    name: str,
    description: str | None,
    input_schema: dict[str, object],
    meta: dict[str, object] | None,
) -> object: ...

def mcp_plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> str: ...

def mcp_http_transport(url: str, transport_type: str | None) -> str: ...

def mcp_render_result(tool_name: str, result: object) -> str: ...

def mcp_check_result(result: object) -> None: ...

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
    def bind_identification(self, headers: list[tuple[str, str]]) -> None: ...

class RealtimeFallbackModel:
    """The generated facade holds native resources; host provider objects use the shared bridge."""
    def __init__(
        self,
        models: list[object],
        *,
        cooldown: float = 10.0,
        regenerate_on_swap: bool = True,
        on_availability: object | None = None,
    ) -> None: ...
    @property
    def active_index(self) -> int: ...
    def capabilities(self) -> dict[str, object]: ...
    async def restart(self, *, switch_model: bool = False) -> None: ...
