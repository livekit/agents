from __future__ import annotations

import contextvars
import json
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from . import trace_types

if TYPE_CHECKING:
    from ..llm import (
        ChatContext,
        ChatItem,
        CompletionUsage,
        FunctionCall,
        FunctionToolCall,
        Tool,
    )

    # both carry the call_id/name/arguments the convention's tool_call part needs
    AnyFunctionCall: TypeAlias = FunctionCall | FunctionToolCall
    from ..metrics import LLMMetrics


_FALSY = ("0", "false", "no", "off")

# the env var name the GenAI conventions standardise for this opt-in
_capture_content: bool = (
    os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").strip().lower()
    not in _FALSY
)


def set_capture_content(enabled: bool) -> None:
    """When off, spans keep every non-content GenAI attribute and omit the message
    payloads, tool definitions and tool call arguments/results."""
    global _capture_content
    _capture_content = enabled


def capture_content_enabled() -> bool:
    return _capture_content


# A custom `llm_node` may do the inference itself — returning a plain str, streaming its
# own chunks, or calling a third-party engine — and never construct an LLMStream. Those
# paths have no nested `llm_request` span to carry the convention's attributes, so the node
# span records them instead. LLMStream marks the context when it does create one, which is
# what tells the two cases apart.
_inference_recorded: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "lk_inference_recorded", default=None
)


def track_inference_span() -> list[bool]:
    """Start tracking, returning a marker that fills in if an ``llm_request`` span is created.

    No reset: the caller runs as its own asyncio task, so the context copy — and this
    variable with it — is discarded when that task finishes.
    """
    recorded: list[bool] = []
    _inference_recorded.set(recorded)
    return recorded


def mark_inference_span_recorded() -> None:
    """Called where an ``llm_request`` span is created, so the enclosing node stands down."""
    if (recorded := _inference_recorded.get()) is not None:
        recorded.append(True)


def _text_part(content: str) -> dict[str, Any]:
    return {"type": "text", "content": content}


def _message_parts(item: ChatItem) -> list[dict[str, Any]]:
    from ..llm import AudioContent, ImageContent

    parts: list[dict[str, Any]] = []
    if item.type == "message":
        for content in item.content:
            if isinstance(content, str):
                parts.append(_text_part(content))
            elif isinstance(content, ImageContent):
                # a data: URL is inline bytes, which the convention models as a blob;
                # recording the base64 payload on a span is never worth its size, so
                # both forms are reported as a uri part without the payload
                image = content.image
                parts.append(
                    {
                        "type": "uri",
                        "modality": "image",
                        "mime_type": content.mime_type,
                        "uri": image
                        if isinstance(image, str) and not image.startswith("data:")
                        else "",
                    }
                )
            elif isinstance(content, AudioContent):
                part: dict[str, Any] = {"type": "blob", "modality": "audio", "content": ""}
                if content.transcript:
                    # the transcript is the only audio content worth carrying; the frames
                    # are recorded separately by session recording, never on a span
                    part["transcript"] = content.transcript
                parts.append(part)
    elif item.type == "function_call":
        parts.append(
            {
                "type": "tool_call",
                "id": item.call_id,
                "name": item.name,
                "arguments": _maybe_json(item.arguments),
            }
        )
    elif item.type == "function_call_output":
        parts.append(
            {
                "type": "tool_call_response",
                "id": item.call_id,
                "response": _maybe_json(item.output),
            }
        )
    return parts


def _maybe_json(raw: str) -> Any:
    """Best-effort deserialization, as the convention asks of instrumentations."""
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def to_system_instructions(chat_ctx: ChatContext) -> list[dict[str, Any]]:
    """LiveKit carries an agent's instructions as ``system``/``developer`` messages in
    the chat context, but they originate from ``Agent(instructions=...)`` rather than
    from the conversation, so they are reported as instructions rather than history."""
    parts: list[dict[str, Any]] = []
    for item in chat_ctx.items:
        if item.type == "message" and item.role in ("system", "developer"):
            if (text := item.raw_text_content) is not None:
                parts.append(_text_part(text))
    return parts


def to_input_messages(chat_ctx: ChatContext) -> list[dict[str, Any]]:
    """History in the order it was sent. ``system``/``developer`` messages go to
    ``gen_ai.system_instructions`` instead, and non-conversational items (agent
    handoffs, config updates) are skipped."""
    messages: list[dict[str, Any]] = []
    for item in chat_ctx.items:
        role: str
        if item.type == "message":
            if item.role in ("system", "developer"):
                continue
            role = item.role
        elif item.type == "function_call":
            role = "assistant"
        elif item.type == "function_call_output":
            role = "tool"
        else:
            continue

        parts = _message_parts(item)
        if not parts:
            continue

        # consecutive tool calls from one assistant turn belong to a single message
        if (
            messages
            and messages[-1]["role"] == role == "assistant"
            and item.type == "function_call"
        ):
            messages[-1]["parts"].extend(parts)
            continue

        messages.append({"role": role, "parts": parts})
    return messages


def to_output_messages(
    *,
    text: str | None = None,
    function_calls: Sequence[AnyFunctionCall] = (),
    finish_reason: str | None = None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if text:
        parts.append(_text_part(text))
    for call in function_calls:
        parts.append(
            {
                "type": "tool_call",
                "id": call.call_id,
                "name": call.name,
                "arguments": _maybe_json(call.arguments),
            }
        )
    if not parts:
        return []

    message: dict[str, Any] = {"role": "assistant", "parts": parts}
    if finish_reason:
        message["finish_reason"] = finish_reason
    return [message]


def to_tool_definitions(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    """``parameters`` is deliberately omitted: the convention marks it NOT RECOMMENDED by
    default because a schema is large, and building one per request would be pure
    overhead for telemetry."""
    from ..llm.tool_context import (
        ProviderTool,
        get_function_info,
        get_raw_function_info,
        is_function_tool,
        is_raw_function_tool,
    )

    definitions: list[dict[str, Any]] = []
    for tool in tools:
        definition: dict[str, Any]
        if is_function_tool(tool):
            info = get_function_info(tool)
            definition = {"type": "function", "name": info.name}
            if info.description:
                definition["description"] = info.description
        elif is_raw_function_tool(tool):
            raw = get_raw_function_info(tool)
            definition = {"type": "function", "name": raw.name}
            if isinstance(description := raw.raw_schema.get("description"), str) and description:
                definition["description"] = description
        elif isinstance(tool, ProviderTool):
            definition = {"type": tool.id, "name": tool.id}
        else:
            continue
        definitions.append(definition)
    return definitions


def finish_reason_for(
    *, function_calls: Sequence[AnyFunctionCall] = (), interrupted: bool = False
) -> str:
    if interrupted:
        # checked first: a generation that emitted a tool call and then failed ended
        # abnormally, and the convention has no `cancelled` value for that
        return trace_types.GenAIFinishReason.ERROR
    if function_calls:
        return trace_types.GenAIFinishReason.TOOL_CALL
    return trace_types.GenAIFinishReason.STOP


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _conversation_id() -> str | None:
    """The convention forbids fabricating one (no UUIDs, trace ids or content hashes),
    so this is the room sid LiveKit already stamps on every record."""
    from ..job import get_job_context

    ctx = get_job_context(required=False)
    if ctx is None:
        return None
    return ctx.job.room.sid or None


def set_content_attributes(
    span: trace.Span,
    *,
    system_instructions: list[dict[str, Any]] | None = None,
    input_messages: list[dict[str, Any]] | None = None,
    output_messages: list[dict[str, Any]] | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> None:
    """Values are JSON strings: OpenTelemetry attributes cannot hold structured values
    yet, which the convention explicitly allows for spans."""
    if not _capture_content or not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {}
    if system_instructions:
        attrs[trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS] = _json(system_instructions)
    if input_messages:
        attrs[trace_types.ATTR_GEN_AI_INPUT_MESSAGES] = _json(input_messages)
    if output_messages:
        attrs[trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES] = _json(output_messages)
    if tool_definitions:
        attrs[trace_types.ATTR_GEN_AI_TOOL_DEFINITIONS] = _json(tool_definitions)
    if attrs:
        span.set_attributes(attrs)


def set_request_attributes(
    span: trace.Span,
    *,
    operation: str,
    provider: str | None = None,
    model: str | None = None,
    stream: bool | None = None,
    output_type: str | None = None,
) -> None:
    """The attributes the convention asks for at span creation time."""
    if not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {trace_types.ATTR_GEN_AI_OPERATION_NAME: operation}
    if (normalized := trace_types.gen_ai_provider_name(provider)) is not None:
        attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] = normalized
    if model:
        attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] = model
    if stream:
        # "if and only if the request is streaming; if unset, assumed non-streaming"
        attrs[trace_types.ATTR_GEN_AI_REQUEST_STREAM] = True
    if (conv := _conversation_id()) is not None:
        attrs[trace_types.ATTR_GEN_AI_CONVERSATION_ID] = conv
    if output_type:
        attrs[trace_types.ATTR_GEN_AI_OUTPUT_TYPE] = output_type
    span.set_attributes(attrs)


def set_response_attributes(
    span: trace.Span,
    *,
    response_id: str | None = None,
    model: str | None = None,
    finish_reasons: Sequence[str] | None = None,
    time_to_first_chunk: float | None = None,
) -> None:
    if not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {}
    if response_id:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_ID] = response_id
    if model:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] = model
    if finish_reasons:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_FINISH_REASONS] = list(finish_reasons)
    if time_to_first_chunk is not None and time_to_first_chunk >= 0:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = time_to_first_chunk
    if attrs:
        span.set_attributes(attrs)


def set_usage_attributes(span: trace.Span, usage: CompletionUsage | LLMMetrics) -> None:
    """Per the convention the detailed counts are subsets of the totals, so cached and
    reasoning tokens are reported alongside — not added to — input and output tokens."""
    if not span.is_recording():
        return

    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    cache_read = getattr(usage, "cache_read_tokens", 0) or getattr(usage, "prompt_cached_tokens", 0)
    cache_write = getattr(usage, "cache_creation_tokens", 0)
    reasoning = getattr(usage, "reasoning_tokens", 0)

    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
    }
    if cache_read:
        attrs[trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
        # the unofficial spelling is what Datadog's mapping table keys on, and the realtime
        # path has always emitted it. #6852 deliberately left it off the pipeline path,
        # which meant cached tokens — usually the largest cost lever in a multi-turn agent —
        # were attributed for realtime sessions and silently absent for pipeline ones.
        attrs[trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS] = cache_read
    if cache_write:
        attrs[trace_types.ATTR_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] = cache_write
    if reasoning:
        attrs[trace_types.ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = reasoning
        # unofficial spelling recognised by Langfuse, kept alongside the standard one
        attrs[trace_types.ATTR_GEN_AI_USAGE_REASONING_TOKENS] = reasoning
    span.set_attributes(attrs)


def set_tool_attributes(
    span: trace.Span,
    *,
    name: str,
    call_id: str | None = None,
    tool_type: str = "function",
    description: str | None = None,
    arguments: str | None = None,
    agent_name: str | None = None,
) -> None:
    if not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_GEN_AI_OPERATION_NAME: trace_types.GenAIOperationName.EXECUTE_TOOL,
        trace_types.ATTR_GEN_AI_TOOL_NAME: name,
        trace_types.ATTR_GEN_AI_TOOL_TYPE: tool_type,
    }
    if call_id:
        attrs[trace_types.ATTR_GEN_AI_TOOL_CALL_ID] = call_id
    if agent_name:
        # "the human-readable name of the agent executing the tool", conditionally required
        attrs[trace_types.ATTR_GEN_AI_AGENT_NAME] = agent_name
    # not in the convention's execute_tool table, but Datadog groups a session by this
    # attribute rather than by trace membership, so a tool span without it drops out of
    # the session view
    if (conv := _conversation_id()) is not None:
        attrs[trace_types.ATTR_GEN_AI_CONVERSATION_ID] = conv
    if _capture_content:
        if description:
            attrs[trace_types.ATTR_GEN_AI_TOOL_DESCRIPTION] = description
        if arguments is not None:
            attrs[trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] = _json(_maybe_json(arguments))
    span.set_attributes(attrs)


def set_tool_result(span: trace.Span, *, result: str | None, is_error: bool) -> None:
    if not span.is_recording():
        return
    if is_error:
        # the convention records the result only on success
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, "tool_error")
        return
    if _capture_content and result is not None:
        span.set_attribute(trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT, _json(_maybe_json(result)))


def set_agent_attributes(
    span: trace.Span,
    *,
    operation: str,
    agent_name: str,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    if not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_GEN_AI_OPERATION_NAME: operation,
        trace_types.ATTR_GEN_AI_AGENT_NAME: agent_name,
    }
    # required on create_agent; the model is conditionally required and an agent is
    # configured with exactly one, which is when the convention asks for it
    if (normalized := trace_types.gen_ai_provider_name(provider)) is not None:
        attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] = normalized
    if model:
        attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] = model
    if (conv := _conversation_id()) is not None:
        attrs[trace_types.ATTR_GEN_AI_CONVERSATION_ID] = conv
    span.set_attributes(attrs)


def set_workflow_attributes(span: trace.Span, *, name: str) -> None:
    """LiveKit's session is the convention's workflow."""
    if not span.is_recording():
        return

    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_GEN_AI_OPERATION_NAME: trace_types.GenAIOperationName.INVOKE_WORKFLOW,
        trace_types.ATTR_GEN_AI_WORKFLOW_NAME: name,
    }
    if (conv := _conversation_id()) is not None:
        attrs[trace_types.ATTR_GEN_AI_CONVERSATION_ID] = conv
    span.set_attributes(attrs)


def set_error_type(span: trace.Span, error: BaseException | str) -> None:
    """A low-cardinality identifier, never the error message: that is free-form and may
    carry user data."""
    if not span.is_recording():
        return
    if isinstance(error, str):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, error)
        return

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, str(status_code))
        return
    span.set_attribute(trace_types.ATTR_ERROR_TYPE, type(error).__qualname__)
