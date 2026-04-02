from __future__ import annotations

from typing import TYPE_CHECKING

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


def _model_attrs(metadata: Metadata | None) -> dict[str, str]:
    attrs: dict[str, str] = {}
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
    """Record usage counters directly from each metrics event."""
    attrs = _model_attrs(getattr(ev, "metadata", None))

    if isinstance(ev, LLMMetrics):
        if ev.prompt_tokens:
            _llm_input_tokens.add(ev.prompt_tokens, attributes=attrs)
        if ev.prompt_cached_tokens:
            _llm_input_cached_tokens.add(ev.prompt_cached_tokens, attributes=attrs)
        if ev.completion_tokens:
            _llm_output_tokens.add(ev.completion_tokens, attributes=attrs)

    elif isinstance(ev, RealtimeModelMetrics):
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

    elif isinstance(ev, TTSMetrics):
        if ev.characters_count:
            _tts_characters.add(ev.characters_count, attributes=attrs)
        if ev.audio_duration:
            _tts_audio_duration.add(ev.audio_duration, attributes=attrs)

    elif isinstance(ev, STTMetrics):
        if ev.audio_duration:
            _stt_audio_duration.add(ev.audio_duration, attributes=attrs)

    elif isinstance(ev, InterruptionMetrics):
        if ev.num_requests:
            _interruption_requests.add(ev.num_requests, attributes=attrs)

    # Connection timing
    if isinstance(ev, (STTMetrics, TTSMetrics, RealtimeModelMetrics)):
        if ev.acquire_time > 0:
            conn_attrs = _model_attrs(ev.metadata)
            conn_attrs["connection_reused"] = str(ev.connection_reused).lower()
            _connection_acquire_time.record(ev.acquire_time, attributes=conn_attrs)
