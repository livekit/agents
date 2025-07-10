from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import AgentMetrics, EOUMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = default_logger

    if isinstance(metrics, LLMMetrics):
        logger.info(
            "LLM metrics",
            extra={
                "ttft": round(metrics.ttft, 2),
                "prompt_tokens": metrics.prompt_tokens,
                "prompt_cached_tokens": metrics.prompt_cached_tokens,
                "completion_tokens": metrics.completion_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, RealtimeModelMetrics):
        logger.info(
            "RealtimeModel metrics",
            extra={
                "ttft": round(metrics.ttft, 2),
                "input_tokens": metrics.input_tokens,
                "cached_input_tokens": metrics.input_token_details.cached_tokens,
                "output_tokens": metrics.output_tokens,
                "total_tokens": metrics.total_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            "TTS metrics",
            extra={
                "ttfb": metrics.ttfb,
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            "EOU metrics",
            extra={
                "end_of_utterance_delay": round(metrics.end_of_utterance_delay, 2),
                "transcription_delay": round(metrics.transcription_delay, 2),
            },
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(
            "STT metrics",
            extra={
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
