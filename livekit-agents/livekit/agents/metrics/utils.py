from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import AgentMetrics, EOUMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = default_logger

    if isinstance(metrics, LLMMetrics):
        logger.info(
            f"LLM metrics: ttft={metrics.ttft:.2f}, input_tokens={metrics.prompt_tokens},  cached_input_tokens={metrics.prompt_cached_tokens}, output_tokens={metrics.completion_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, RealtimeModelMetrics):
        logger.info(
            f"RealtimeModel metrics: ttft={metrics.ttft:.2f}, input_tokens={metrics.input_tokens}, cached_input_tokens={metrics.input_token_details.cached_tokens}, output_tokens={metrics.output_tokens}, total_tokens={metrics.total_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            f"TTS metrics: ttfb={metrics.ttfb}, audio_duration={metrics.audio_duration:.2f}"
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            f"EOU metrics: end_of_utterance_delay={metrics.end_of_utterance_delay:.2f}, transcription_delay={metrics.transcription_delay:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(f"STT metrics: audio_duration={metrics.audio_duration:.2f}")
