from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar, Union

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..debug import trace_types, tracer
from ..log import logger
from ..metrics import LLMMetrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio
from ..utils.images import EncodeOptions, encode
from .chat_context import (
    AudioContent,
    ChatContext,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from .tool_context import (
    FunctionTool,
    RawFunctionTool,
    ToolChoice,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)


class CompletionUsage(BaseModel):
    completion_tokens: int
    """The number of tokens in the completion."""
    prompt_tokens: int
    """The number of input tokens used (includes cached tokens)."""
    prompt_cached_tokens: int = 0
    """The number of cached input tokens used."""
    cache_creation_tokens: int = 0
    """The number of tokens used to create the cache."""
    cache_read_tokens: int = 0
    """The number of tokens read from the cache."""
    total_tokens: int
    """The total number of tokens used (completion + prompt tokens)."""


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str


class ChoiceDelta(BaseModel):
    role: ChatRole | None = None
    content: str | None = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)


class ChatChunk(BaseModel):
    id: str
    delta: ChoiceDelta | None = None
    usage: CompletionUsage | None = None


class LLMError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["llm_error"] = "llm_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class LLM(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]],
    Generic[TEvent],
):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        """Get the model name/identifier for this LLM instance.

        Returns:
            The model name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their model information.
        """
        return "unknown"

    @property
    def provider(self) -> str:
        """Get the provider for this LLM instance.

        Returns:
            The provider following OpenTelemetry GenAI conventions if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their provider information.
            Use lowercase standardized values like: "openai", "anthropic", "cohere",
            "aws.bedrock", "azure.ai.openai", "gcp.gemini", etc.
        """
        return "unknown"

    @property
    def has_opentelemetry_instrumentation(self) -> bool:
        """Check if this plugin provides OpenTelemetry instrumentation.

        Returns:
            True if the plugin provides native OpenTelemetry instrumentation, False otherwise.

        Note:
            Plugins should override this property to indicate if they provide OpenTelemetry
            instrumentation. This allows the hybrid approach to make more informed decisions
            about span creation.
        """
        return False

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream: ...

    def prewarm(self) -> None:
        """Pre-warm connection to the LLM service"""
        pass

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class LLMStream(ABC):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options

        self._event_ch = aio.Chan[ChatChunk]()
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._current_attempt_has_error = False
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="LLM._metrics_task"
        )

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._llm_request_span: trace.Span | None = None

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            # Only create our own span if the provider doesn't have auto-instrumentation
            if not self._llm.has_opentelemetry_instrumentation:
                # No auto-instrumentation - create our own span as the generation span
                self._llm_request_span = tracer.start_span("llm_request")
                self._add_genai_attributes()
            else:
                # Provider has auto-instrumentation - let it handle span creation
                self._llm_request_span = None

            try:
                return await self._run()
            except APIError as e:
                # Record the exception on the span
                if self._llm_request_span:
                    self._llm_request_span.record_exception(e)
                    self._llm_request_span.set_status(Status(StatusCode.ERROR, description=str(e)))

                retry_interval = self._conn_options._interval_for_retry(i)

                if self._conn_options.max_retry == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif i == self._conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to generate LLM completion after {self._conn_options.max_retry + 1} attempts",  # noqa: E501
                    ) from e

                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to generate LLM completion, retrying in {retry_interval}s",  # noqa: E501
                        exc_info=e,
                        extra={
                            "llm": self._llm._label,
                            "attempt": i + 1,
                        },
                    )

                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)

                # reset the flag when retrying
                self._current_attempt_has_error = False

            except Exception as e:
                # Record the exception on the span
                if self._llm_request_span:
                    self._llm_request_span.record_exception(e)
                    self._llm_request_span.set_status(Status(StatusCode.ERROR, description=str(e)))
                self._emit_error(e, recoverable=False)
                raise

    def _add_genai_attributes(self) -> None:
        """Add OpenTelemetry GenAI semantic conventions to the current span."""
        if not self._llm_request_span:
            return

        # Add model and provider information using trace_types constants
        self._llm_request_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, self._llm.model)

        self._llm_request_span.set_attribute(trace_types.ATTR_GEN_AI_SYSTEM, self._llm.provider)

        # Add operation name
        self._llm_request_span.set_attribute(trace_types.ATTR_GEN_AI_OPERATION_NAME, "chat")

        # Add streaming indicator (this is always true for LLMStream)
        self._llm_request_span.set_attribute(trace_types.ATTR_LLM_IS_STREAMING, True)

        # Emit OpenTelemetry GenAI events for conversation context
        # Events are the standard way to capture conversation content in OpenTelemetry
        self._emit_conversation_events()

        # Add function information for available tools (OpenLLMetry standard)
        if self._tools:
            for i, tool in enumerate(self._tools):
                if is_function_tool(tool):
                    info = get_function_info(tool)
                    self._llm_request_span.set_attribute(
                        trace_types.ATTR_LLM_REQUEST_FUNCTIONS_NAME.format(i), info.name
                    )
                    if info.description:
                        self._llm_request_span.set_attribute(
                            trace_types.ATTR_LLM_REQUEST_FUNCTIONS_DESCRIPTION.format(i),
                            info.description,
                        )
                elif is_raw_function_tool(tool):
                    raw_info = get_raw_function_info(tool)
                    self._llm_request_span.set_attribute(
                        trace_types.ATTR_LLM_REQUEST_FUNCTIONS_NAME.format(i), raw_info.name
                    )
                    # For raw function tools, description and parameters are in the raw_schema
                    if "description" in raw_info.raw_schema:
                        self._llm_request_span.set_attribute(
                            trace_types.ATTR_LLM_REQUEST_FUNCTIONS_DESCRIPTION.format(i),
                            raw_info.raw_schema["description"],
                        )
                    if "parameters" in raw_info.raw_schema:
                        self._llm_request_span.set_attribute(
                            trace_types.ATTR_LLM_REQUEST_FUNCTIONS_PARAMETERS.format(i),
                            json.dumps(raw_info.raw_schema["parameters"]),
                        )

    def _add_completion_attributes(
        self,
        usage: CompletionUsage | None,
        request_id: str,
        metrics: Any,
        response_content: str,
        tool_calls: list[FunctionToolCall],
        completion_start_time: float | None,
    ) -> None:
        """Add OpenTelemetry completion attributes after LLM completion."""
        if not self._llm_request_span:
            return

        # Add response ID
        if request_id:
            self._llm_request_span.set_attribute(trace_types.ATTR_GEN_AI_RESPONSE_ID, request_id)

        # Add usage/token information
        if usage:
            if usage.prompt_tokens is not None:
                self._llm_request_span.set_attribute(
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
                )

            if usage.completion_tokens is not None:
                self._llm_request_span.set_attribute(
                    trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
                )

            # Add total tokens (OpenLLMetry standard)
            if usage.total_tokens is not None:
                self._llm_request_span.set_attribute(
                    trace_types.ATTR_LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
                )

        # Add model name to response
        self._llm_request_span.set_attribute(
            trace_types.ATTR_GEN_AI_RESPONSE_MODEL, self._llm.model
        )

        # Add LiveKit-specific metrics
        self._llm_request_span.set_attribute(
            trace_types.ATTR_LLM_METRICS, metrics.model_dump_json()
        )

        # Add completion start time for Langfuse
        if completion_start_time is not None:
            self._llm_request_span.set_attribute(
                trace_types.ATTR_LANGFUSE_COMPLETION_START_TIME,
                f'"{completion_start_time}"',
            )

        # Emit response event if we have content or tool calls
        if response_content or tool_calls:
            try:
                response_event_body = {}

                # Add content if present
                if response_content:
                    response_event_body["content"] = response_content

                # Add tool calls if present
                if tool_calls:
                    response_event_body["tool_calls"] = json.dumps(
                        [
                            {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            }
                            for tool_call in tool_calls
                        ]
                    )

                self._llm_request_span.add_event(
                    name=trace_types.EVENT_GEN_AI_CHOICE, attributes=response_event_body
                )
            except Exception as e:
                # Don't let event emission errors break the main flow
                logger.debug("Failed to emit response event", exc_info=e)

        # End the span now that all attributes are set
        self._llm_request_span.end()
        self._llm_request_span = None

    def _emit_conversation_events(self) -> None:
        """Emit OpenTelemetry GenAI events for conversation context."""
        if not self._chat_ctx.items:
            return

        # Emit events for each conversation item
        for item in self._chat_ctx.items:
            try:
                if isinstance(item, ChatMessage):
                    event_name = self._get_message_event_name(item.role)

                    # Extract content following OpenLLMetry patterns
                    # OpenLLMetry serializes complex content as JSON arrays for multimodal support
                    content_for_event = []
                    for content_item in item.content:
                        if isinstance(content_item, str):
                            content_for_event.append({"type": "text", "text": content_item})
                        elif isinstance(content_item, ImageContent):
                            # Follow OpenLLMetry image content format
                            if isinstance(content_item.image, str):
                                content_for_event.append(
                                    {"type": "image_url", "image_url": {"url": content_item.image}}
                                )
                            else:
                                # rtc.VideoFrame - encode as base64 data URL
                                try:
                                    # Encode frame to JPEG bytes
                                    image_bytes = encode(
                                        content_item.image, EncodeOptions(format="JPEG", quality=75)
                                    )

                                    # Create data URL
                                    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

                                    content_for_event.append(
                                        {"type": "image_url", "image_url": {"url": data_url}}
                                    )
                                except Exception as e:
                                    # Fallback to placeholder if encoding fails
                                    logger.debug("Failed to encode VideoFrame", exc_info=e)
                                    content_for_event.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"[ImageFrame: {content_item.id}]"
                                            },
                                        }
                                    )
                        elif isinstance(content_item, AudioContent):
                            # Audio not supported in OpenLLMetry - use placeholder
                            content_for_event.append(
                                {"type": "text", "text": f"[Audio: {content_item.id}]"}
                            )
                        else:
                            content_for_event.append({"type": "text", "text": str(content_item)})

                    # For event content, use the structured format if complex, otherwise simple text
                    if len(content_for_event) == 1 and content_for_event[0].get("type") == "text":
                        content_text = content_for_event[0]["text"]
                    else:
                        content_text = json.dumps(content_for_event)

                    # Create event body
                    event_body = {
                        "content": content_text,
                    }

                    # Add role if it's not redundant with event name
                    if item.role not in event_name:
                        event_body["role"] = item.role

                    # Emit the event to the LLM request span
                    self._llm_request_span.add_event(name=event_name, attributes=event_body)

                elif isinstance(item, (FunctionCall, FunctionCallOutput)):
                    # Emit tool message event for function calls
                    if isinstance(item, FunctionCall):
                        content = item.arguments
                    else:  # FunctionCallOutput
                        content = item.output

                    event_body = {"content": content, "name": item.name}

                    # Emit the event to the LLM request span
                    self._llm_request_span.add_event(
                        name=trace_types.EVENT_GEN_AI_TOOL_MESSAGE, attributes=event_body
                    )

            except Exception as e:
                # Don't let event emission errors break the main flow
                logger.debug("Failed to emit GenAI event", exc_info=e)

    def _get_message_event_name(self, role: str) -> str:
        """Get the appropriate GenAI event name for a message role."""
        role_to_event = {
            "system": trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE,
            "user": trace_types.EVENT_GEN_AI_USER_MESSAGE,
            "assistant": trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
        }
        return role_to_event.get(role.lower(), trace_types.EVENT_GEN_AI_USER_MESSAGE)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._current_attempt_has_error = True
        self._llm.emit(
            "error",
            LLMError(
                timestamp=time.time(),
                label=self._llm._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    @utils.log_exceptions(logger=logger)
    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[ChatChunk]) -> None:
        start_time = time.perf_counter()
        ttft = -1.0
        request_id = ""
        usage: CompletionUsage | None = None
        response_content = ""
        tool_calls: list[FunctionToolCall] = []
        completion_start_time = None

        async for ev in event_aiter:
            request_id = ev.id
            if ttft == -1.0:
                ttft = time.perf_counter() - start_time

                # Capture completion start time when first token arrives
                completion_start_time = datetime.now(timezone.utc).isoformat()

            # Collect response content and tool calls from deltas
            if ev.delta:
                if ev.delta.content:
                    response_content += ev.delta.content
                if ev.delta.tool_calls:
                    tool_calls.extend(ev.delta.tool_calls)

            if ev.usage is not None:
                usage = ev.usage

        duration = time.perf_counter() - start_time

        if self._current_attempt_has_error:
            return

        metrics = LLMMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttft=ttft,
            duration=duration,
            cancelled=self._task.cancelled(),
            label=self._llm._label,
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            prompt_cached_tokens=usage.prompt_cached_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            tokens_per_second=usage.completion_tokens / duration if usage else 0.0,
        )
        self._add_completion_attributes(
            usage, request_id, metrics, response_content, tool_calls, completion_start_time
        )
        self._llm.emit("metrics_collected", metrics)

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[FunctionTool | RawFunctionTool]:
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        await self._metrics_task
        if self._llm_request_span:
            self._llm_request_span.end()
            self._llm_request_span = None

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __aenter__(self) -> LLMStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def to_str_iterable(self) -> AsyncIterable[str]:
        """
        Convert the LLMStream to an async iterable of strings.
        This assumes the stream will not call any tools.
        """

        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()
