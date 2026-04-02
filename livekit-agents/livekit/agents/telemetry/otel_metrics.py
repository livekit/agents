from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import metrics as metrics_api

from ..metrics.base import (
    AgentMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
)
from ..metrics.usage import (
    InterruptionModelUsage,
    LLMModelUsage,
    ModelUsageCollector,
    STTModelUsage,
    TTSModelUsage,
)

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContext, MetricsReport

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
_interruption_requests = _meter.create_counter("lk.agents.usage.interruption_requests")

# -- Connection metrics --
_connection_acquire_time = _meter.create_histogram(
    "lk.agents.connection.acquire_time",
    unit="s",
    description="Time to acquire a connection (WebSocket only)",
)

# Per-model usage collectors
_usage_collector = ModelUsageCollector()


def flush_turn_metrics(chat_ctx: ChatContext) -> None:
    """Emit per-turn latency histograms from the chat history. Called at session end."""
    for msg in chat_ctx.messages():
        _record_turn_metrics(msg.metrics)


def _record_turn_metrics(report: MetricsReport) -> None:
    if "e2e_latency" in report:
        _turn_e2e_latency.record(report["e2e_latency"])
    if "llm_node_ttft" in report:
        _turn_llm_ttft.record(report["llm_node_ttft"])
    if "tts_node_ttfb" in report:
        _turn_tts_ttfb.record(report["tts_node_ttfb"])
    if "transcription_delay" in report:
        _turn_transcription_delay.record(report["transcription_delay"])
    if "end_of_turn_delay" in report:
        _turn_end_of_turn_delay.record(report["end_of_turn_delay"])
    if "on_user_turn_completed_delay" in report:
        _turn_on_user_turn_completed_delay.record(report["on_user_turn_completed_delay"])


def collect_usage(ev: AgentMetrics) -> None:
    """Buffer usage per model and record per-event metrics. Called on each metrics event."""
    _usage_collector.collect(ev)

    if isinstance(ev, (LLMMetrics, STTMetrics, TTSMetrics, RealtimeModelMetrics)):
        if ev.acquire_time > 0:
            attrs: dict[str, str] = {"connection_reused": str(ev.connection_reused)}
            if ev.metadata:
                if ev.metadata.model_provider:
                    attrs["model_provider"] = ev.metadata.model_provider
                if ev.metadata.model_name:
                    attrs["model_name"] = ev.metadata.model_name
            _connection_acquire_time.record(ev.acquire_time, attributes=attrs)


def flush_usage() -> None:
    """Emit all buffered usage as OTEL counters, keyed by model/provider. Called at session end."""
    for usage in _usage_collector.flatten():
        attrs: dict[str, str] = {}
        if usage.provider:
            attrs["model_provider"] = usage.provider
        if usage.model:
            attrs["model_name"] = usage.model

        if isinstance(usage, LLMModelUsage):
            _emit_llm_usage(usage, attrs)
        elif isinstance(usage, TTSModelUsage):
            _emit_tts_usage(usage, attrs)
        elif isinstance(usage, STTModelUsage):
            _emit_stt_usage(usage, attrs)
        elif isinstance(usage, InterruptionModelUsage):
            _emit_interruption_usage(usage, attrs)


def _emit_llm_usage(usage: LLMModelUsage, attrs: dict[str, str]) -> None:
    if usage.input_tokens:
        _llm_input_tokens.add(usage.input_tokens, attributes=attrs)
    if usage.input_cached_tokens:
        _llm_input_cached_tokens.add(usage.input_cached_tokens, attributes=attrs)
    if usage.output_tokens:
        _llm_output_tokens.add(usage.output_tokens, attributes=attrs)
    if usage.input_audio_tokens:
        _llm_input_audio_tokens.add(usage.input_audio_tokens, attributes=attrs)
    if usage.input_text_tokens:
        _llm_input_text_tokens.add(usage.input_text_tokens, attributes=attrs)
    if usage.output_audio_tokens:
        _llm_output_audio_tokens.add(usage.output_audio_tokens, attributes=attrs)
    if usage.output_text_tokens:
        _llm_output_text_tokens.add(usage.output_text_tokens, attributes=attrs)
    if usage.session_duration:
        _llm_session_duration.add(usage.session_duration, attributes=attrs)


def _emit_tts_usage(usage: TTSModelUsage, attrs: dict[str, str]) -> None:
    if usage.characters_count:
        _tts_characters.add(usage.characters_count, attributes=attrs)
    if usage.audio_duration:
        _tts_audio_duration.add(usage.audio_duration, attributes=attrs)


def _emit_stt_usage(usage: STTModelUsage, attrs: dict[str, str]) -> None:
    if usage.audio_duration:
        _stt_audio_duration.add(usage.audio_duration, attributes=attrs)


def _emit_interruption_usage(usage: InterruptionModelUsage, attrs: dict[str, str]) -> None:
    if usage.total_requests:
        _interruption_requests.add(usage.total_requests, attributes=attrs)
