from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry import metrics as metrics_api

from ..metrics.base import (
    AgentMetrics,
    InterruptionMetrics,
    LLMMetrics,
    Metadata,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
)
from . import trace_types

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContext, MetricsMetadata, MetricsReport

_meter = metrics_api.get_meter("livekit-agents")

# -- Per-turn latency histograms --
_turn_e2e_latency = _meter.create_histogram(
    "lk.agents.turn.e2e_latency",
    unit="s",
    description="End-to-end turn latency",
)
_turn_llm_ttft = _meter.create_histogram(
    "lk.agents.turn.llm_ttft",
    unit="s",
    description="Pipeline-level LLM time to first token",
)
_turn_tts_ttfb = _meter.create_histogram(
    "lk.agents.turn.tts_ttfb",
    unit="s",
    description="Pipeline-level TTS time to first byte",
)
_turn_transcription_delay = _meter.create_histogram(
    "lk.agents.turn.transcription_delay",
    unit="s",
    description="Time from end of speech to transcript available",
)
_turn_end_of_turn_delay = _meter.create_histogram(
    "lk.agents.turn.end_of_turn_delay",
    unit="s",
    description="Time from end of speech to turn decision",
)
_turn_on_user_turn_completed_delay = _meter.create_histogram(
    "lk.agents.turn.on_user_turn_completed_delay",
    unit="s",
    description="Time to invoke the on_user_turn_completed callback",
)

# -- Usage counters --
_llm_input_tokens = _meter.create_counter("lk.agents.usage.llm_input_tokens")
_llm_input_cached_tokens = _meter.create_counter("lk.agents.usage.llm_input_cached_tokens")
_llm_output_tokens = _meter.create_counter("lk.agents.usage.llm_output_tokens")
_llm_input_audio_tokens = _meter.create_counter("lk.agents.usage.llm_input_audio_tokens")
_llm_input_text_tokens = _meter.create_counter("lk.agents.usage.llm_input_text_tokens")
_llm_output_audio_tokens = _meter.create_counter("lk.agents.usage.llm_output_audio_tokens")
_llm_output_text_tokens = _meter.create_counter("lk.agents.usage.llm_output_text_tokens")
_llm_session_duration = _meter.create_counter(
    "lk.agents.usage.llm_session_duration",
    unit="s",
)
_tts_characters = _meter.create_counter("lk.agents.usage.tts_characters")
_tts_audio_duration = _meter.create_counter(
    "lk.agents.usage.tts_audio_duration",
    unit="s",
)
_stt_audio_duration = _meter.create_counter(
    "lk.agents.usage.stt_audio_duration",
    unit="s",
)
_interruption_num_requests = _meter.create_counter("lk.agents.usage.interruption_num_requests")

# -- Connection metrics --
_connection_acquire_time = _meter.create_histogram(
    "lk.agents.connection.acquire_time",
    unit="s",
    description="Time to acquire a connection (WebSocket only)",
)


# https://github.com/open-telemetry/semantic-conventions-genai (docs/gen-ai/gen-ai-metrics.md).
# Emitted alongside the `lk.agents.*` instruments above so a GenAI-aware backend
# (Datadog Agent Observability, Langfuse) gets the metrics under the names it expects.
_genai_token_usage = _meter.create_histogram(
    trace_types.METRIC_GEN_AI_CLIENT_TOKEN_USAGE,
    unit="{token}",
    description="Number of input and output tokens used by the GenAI operation",
)
_genai_operation_duration = _meter.create_histogram(
    trace_types.METRIC_GEN_AI_CLIENT_OPERATION_DURATION,
    unit="s",
    description="GenAI operation duration",
)
_genai_time_to_first_chunk = _meter.create_histogram(
    trace_types.METRIC_GEN_AI_CLIENT_TIME_TO_FIRST_CHUNK,
    unit="s",
    description="Time to first chunk of a streaming GenAI response",
)
_genai_execute_tool_duration = _meter.create_histogram(
    trace_types.METRIC_GEN_AI_EXECUTE_TOOL_DURATION,
    unit="s",
    description="Tool execution duration",
)
_genai_invoke_agent_duration = _meter.create_histogram(
    trace_types.METRIC_GEN_AI_INVOKE_AGENT_DURATION,
    unit="s",
    description="Agent invocation duration",
)


def _genai_attrs(metadata: Metadata | None, *, operation: str) -> dict[str, Any]:
    """The convention's required metric attributes, on top of the job attribution."""
    attrs = _job_attrs()
    attrs[trace_types.ATTR_GEN_AI_OPERATION_NAME] = operation
    if metadata:
        provider = trace_types.gen_ai_provider_name(metadata.model_provider)
        if provider:
            attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] = provider
        if metadata.model_name:
            attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] = metadata.model_name
            attrs[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] = metadata.model_name
    return attrs


def _record_genai_tokens(attrs: dict[str, Any], *, input_tokens: int, output_tokens: int) -> None:
    """``gen_ai.client.token.usage``, split by the convention's `gen_ai.token.type`."""
    if input_tokens:
        _genai_token_usage.record(
            input_tokens, attributes={**attrs, trace_types.ATTR_GEN_AI_TOKEN_TYPE: "input"}
        )
    if output_tokens:
        _genai_token_usage.record(
            output_tokens, attributes={**attrs, trace_types.ATTR_GEN_AI_TOKEN_TYPE: "output"}
        )


def record_execute_tool_duration(duration: float, *, tool_name: str, error: bool = False) -> None:
    """``gen_ai.execute_tool.duration`` for one function-tool execution."""
    attrs = _job_attrs()
    attrs[trace_types.ATTR_GEN_AI_OPERATION_NAME] = trace_types.GenAIOperationName.EXECUTE_TOOL
    attrs[trace_types.ATTR_GEN_AI_TOOL_NAME] = tool_name
    if error:
        attrs[trace_types.ATTR_ERROR_TYPE] = "tool_error"
    _genai_execute_tool_duration.record(duration, attributes=attrs)


def record_invoke_agent_duration(duration: float, *, agent_name: str) -> None:
    """``gen_ai.invoke_agent.duration`` for one agent turn."""
    attrs = _job_attrs()
    attrs[trace_types.ATTR_GEN_AI_OPERATION_NAME] = trace_types.GenAIOperationName.INVOKE_AGENT
    attrs[trace_types.ATTR_GEN_AI_AGENT_NAME] = agent_name
    _genai_invoke_agent_duration.record(duration, attributes=attrs)


def _job_attrs() -> dict[str, Any]:
    """Per-measurement job attribution.

    The meter provider has process lifetime (the OTel metrics global is
    set-once) and worker processes are reused across jobs, so per-job fields
    cannot live on the provider's resource. Instead, each measurement carries
    the same per-job attributes that are stamped on spans and logs (identity,
    simulation ids, redaction flag, ...), which is also correct for concurrent
    jobs in THREAD mode. Returns a fresh dict — callers may add to it.
    """
    from ..job import get_job_context  # local import: job.py imports this module

    ctx = get_job_context(required=False)
    if ctx is None:
        return {}
    if (state := ctx._telemetry_state) is not None:
        return dict(state.attributes)
    # recording was not initialized (disabled, or the crash path); keep identity
    return {"room_id": ctx.job.room.sid, "job_id": ctx.job.id}


def _model_attrs(metadata: Metadata | None) -> dict[str, Any]:
    attrs = _job_attrs()
    if metadata:
        if metadata.model_provider:
            attrs["model_provider"] = metadata.model_provider
        if metadata.model_name:
            attrs["model_name"] = metadata.model_name
    return attrs


def flush_turn_metrics(chat_ctx: ChatContext) -> None:
    """Emit per-turn latency histograms from the chat history. Called at session end."""
    for msg in chat_ctx.messages():
        _record_turn_metrics(msg.metrics)


def _metadata_to_attrs(metadata: MetricsMetadata | None) -> dict[str, Any]:
    attrs = _job_attrs()
    if metadata:
        if "model_name" in metadata:
            attrs["model_name"] = metadata["model_name"]
        if "model_provider" in metadata:
            attrs["model_provider"] = metadata["model_provider"]
    return attrs


def _record_turn_metrics(report: MetricsReport) -> None:
    llm_attrs = _metadata_to_attrs(report.get("llm_metadata"))
    tts_attrs = _metadata_to_attrs(report.get("tts_metadata"))
    stt_attrs = _metadata_to_attrs(report.get("stt_metadata"))

    if "e2e_latency" in report:
        _turn_e2e_latency.record(report["e2e_latency"], attributes=llm_attrs)
    if "llm_node_ttft" in report:
        _turn_llm_ttft.record(report["llm_node_ttft"], attributes=llm_attrs)
    if "tts_node_ttfb" in report:
        _turn_tts_ttfb.record(report["tts_node_ttfb"], attributes=tts_attrs)
    if "transcription_delay" in report:
        _turn_transcription_delay.record(report["transcription_delay"], attributes=stt_attrs)
    if "end_of_turn_delay" in report:
        _turn_end_of_turn_delay.record(report["end_of_turn_delay"], attributes=stt_attrs)
    if "on_user_turn_completed_delay" in report:
        _turn_on_user_turn_completed_delay.record(
            report["on_user_turn_completed_delay"], attributes=stt_attrs
        )


def collect_usage(ev: AgentMetrics) -> None:
    """Record usage counters directly from each metrics event."""
    if isinstance(ev, LLMMetrics):
        attrs = _model_attrs(ev.metadata)
        if ev.prompt_tokens:
            _llm_input_tokens.add(ev.prompt_tokens, attributes=attrs)
        if ev.prompt_cached_tokens:
            _llm_input_cached_tokens.add(ev.prompt_cached_tokens, attributes=attrs)
        if ev.completion_tokens:
            _llm_output_tokens.add(ev.completion_tokens, attributes=attrs)

        genai_attrs = _genai_attrs(ev.metadata, operation=trace_types.GenAIOperationName.CHAT)
        _record_genai_tokens(
            genai_attrs, input_tokens=ev.prompt_tokens, output_tokens=ev.completion_tokens
        )
        if ev.duration > 0:
            _genai_operation_duration.record(ev.duration, attributes=genai_attrs)
        if ev.ttft >= 0:
            _genai_time_to_first_chunk.record(ev.ttft, attributes=genai_attrs)

    elif isinstance(ev, RealtimeModelMetrics):
        attrs = _model_attrs(ev.metadata)
        if ev.input_tokens:
            _llm_input_tokens.add(ev.input_tokens, attributes=attrs)
        if ev.input_token_details.cached_tokens:
            _llm_input_cached_tokens.add(ev.input_token_details.cached_tokens, attributes=attrs)
        if ev.output_tokens:
            _llm_output_tokens.add(ev.output_tokens, attributes=attrs)
        if ev.input_token_details.audio_tokens:
            _llm_input_audio_tokens.add(ev.input_token_details.audio_tokens, attributes=attrs)
        if ev.input_token_details.text_tokens:
            _llm_input_text_tokens.add(ev.input_token_details.text_tokens, attributes=attrs)
        if ev.output_token_details.audio_tokens:
            _llm_output_audio_tokens.add(ev.output_token_details.audio_tokens, attributes=attrs)
        if ev.output_token_details.text_tokens:
            _llm_output_text_tokens.add(ev.output_token_details.text_tokens, attributes=attrs)
        if ev.session_duration:
            _llm_session_duration.add(ev.session_duration, attributes=attrs)

        genai_attrs = _genai_attrs(
            ev.metadata, operation=trace_types.GenAIOperationName.GENERATE_CONTENT
        )
        _record_genai_tokens(
            genai_attrs, input_tokens=ev.input_tokens, output_tokens=ev.output_tokens
        )
        if ev.duration > 0:
            _genai_operation_duration.record(ev.duration, attributes=genai_attrs)
        if ev.ttft >= 0:
            _genai_time_to_first_chunk.record(ev.ttft, attributes=genai_attrs)

    elif isinstance(ev, TTSMetrics):
        attrs = _model_attrs(ev.metadata)
        if ev.characters_count:
            _tts_characters.add(ev.characters_count, attributes=attrs)
        if ev.audio_duration:
            _tts_audio_duration.add(ev.audio_duration, attributes=attrs)

    elif isinstance(ev, STTMetrics):
        attrs = _model_attrs(ev.metadata)
        if ev.audio_duration:
            _stt_audio_duration.add(ev.audio_duration, attributes=attrs)

    elif isinstance(ev, InterruptionMetrics):
        attrs = _model_attrs(ev.metadata)
        if ev.num_requests:
            _interruption_num_requests.add(ev.num_requests, attributes=attrs)

    # Connection timing
    if isinstance(ev, (STTMetrics, TTSMetrics, RealtimeModelMetrics)):
        if ev.acquire_time > 0:
            conn_attrs = _model_attrs(ev.metadata)
            conn_attrs["connection_reused"] = str(ev.connection_reused).lower()
            _connection_acquire_time.record(ev.acquire_time, attributes=conn_attrs)
