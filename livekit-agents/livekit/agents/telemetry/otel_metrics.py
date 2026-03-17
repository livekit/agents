from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import metrics as metrics_api

from ..metrics.base import AgentMetrics, Metadata
from ..metrics.usage_collector import UsageCollector, UsageSummary

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

# -- Usage counters (emitted at session end) --
_llm_prompt_tokens = _meter.create_counter("lk.agents.usage.llm_prompt_tokens")
_llm_prompt_cached_tokens = _meter.create_counter("lk.agents.usage.llm_prompt_cached_tokens")
_llm_completion_tokens = _meter.create_counter("lk.agents.usage.llm_completion_tokens")
_llm_input_audio_tokens = _meter.create_counter("lk.agents.usage.llm_input_audio_tokens")
_llm_input_text_tokens = _meter.create_counter("lk.agents.usage.llm_input_text_tokens")
_llm_output_audio_tokens = _meter.create_counter("lk.agents.usage.llm_output_audio_tokens")
_llm_output_text_tokens = _meter.create_counter("lk.agents.usage.llm_output_text_tokens")
_tts_characters = _meter.create_counter("lk.agents.usage.tts_characters")
_tts_audio_duration = _meter.create_counter(
    "lk.agents.usage.tts_audio_duration",
    unit="s",
)
_stt_audio_duration = _meter.create_counter(
    "lk.agents.usage.stt_audio_duration",
    unit="s",
)

# Per-model usage collectors (reuses UsageCollector directly)
_usage_by_model: dict[tuple[str | None, str | None], UsageCollector] = {}


def flush_turn_metrics(chat_ctx: ChatContext) -> None:
    """Emit per-turn latency histograms by reading the chat history. Called at session end."""
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


def collect_usage(ev: AgentMetrics) -> None:
    """Buffer usage per model using UsageCollector. Called on each metrics event."""
    metadata: Metadata | None = getattr(ev, "metadata", None)
    key = (
        metadata.model_name if metadata else None,
        metadata.model_provider if metadata else None,
    )
    if key not in _usage_by_model:
        _usage_by_model[key] = UsageCollector()
    _usage_by_model[key].collect(ev)


def flush_usage() -> None:
    """Emit all buffered usage as OTEL counters. Called once at session end."""
    for (model_name, model_provider), collector in _usage_by_model.items():
        summary = collector.get_summary()
        attrs: dict[str, str] = {}
        if model_name:
            attrs["model_name"] = model_name
        if model_provider:
            attrs["model_provider"] = model_provider

        _emit_usage_summary(summary, attrs)
    _usage_by_model.clear()


def _emit_usage_summary(summary: UsageSummary, attrs: dict[str, str]) -> None:
    if summary.llm_prompt_tokens:
        _llm_prompt_tokens.add(summary.llm_prompt_tokens, attributes=attrs)
    if summary.llm_prompt_cached_tokens:
        _llm_prompt_cached_tokens.add(summary.llm_prompt_cached_tokens, attributes=attrs)
    if summary.llm_completion_tokens:
        _llm_completion_tokens.add(summary.llm_completion_tokens, attributes=attrs)
    if summary.llm_input_audio_tokens:
        _llm_input_audio_tokens.add(summary.llm_input_audio_tokens, attributes=attrs)
    if summary.llm_input_text_tokens:
        _llm_input_text_tokens.add(summary.llm_input_text_tokens, attributes=attrs)
    if summary.llm_output_audio_tokens:
        _llm_output_audio_tokens.add(summary.llm_output_audio_tokens, attributes=attrs)
    if summary.llm_output_text_tokens:
        _llm_output_text_tokens.add(summary.llm_output_text_tokens, attributes=attrs)
    if summary.tts_characters_count:
        _tts_characters.add(summary.tts_characters_count, attributes=attrs)
    if summary.tts_audio_duration:
        _tts_audio_duration.add(summary.tts_audio_duration, attributes=attrs)
    if summary.stt_audio_duration:
        _stt_audio_duration.add(summary.stt_audio_duration, attributes=attrs)
