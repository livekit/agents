from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import AgentMetrics, EOUMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = default_logger

    metadata: dict[str, str | float] = {}
    if metrics.metadata:
        metadata |= {
            "model_name": metrics.metadata.model_name or "unknown",
            "model_provider": metrics.metadata.model_provider or "unknown",
        }

    if isinstance(metrics, LLMMetrics):
        logger.info(
            "LLM metrics",
            extra=metadata
            | {
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
            extra=metadata
            | {
                "ttft": round(metrics.ttft, 2),
                "input_tokens": metrics.input_tokens,
                "cached_input_tokens": metrics.input_token_details.cached_tokens,
                "input_text_tokens": metrics.input_token_details.text_tokens,
                "input_cached_text_tokens": metrics.input_token_details.cached_tokens_details.text_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "input_image_tokens": metrics.input_token_details.image_tokens,
                "input_cached_image_tokens": metrics.input_token_details.cached_tokens_details.image_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "input_audio_tokens": metrics.input_token_details.audio_tokens,
                "input_cached_audio_tokens": metrics.input_token_details.cached_tokens_details.audio_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "output_tokens": metrics.output_tokens,
                "output_text_tokens": metrics.output_token_details.text_tokens,
                "output_audio_tokens": metrics.output_token_details.audio_tokens,
                "output_image_tokens": metrics.output_token_details.image_tokens,
                "total_tokens": metrics.total_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            "TTS metrics",
            extra=metadata
            | {
                "ttfb": metrics.ttfb,
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            "EOU metrics",
            extra=metadata
            | {
                "end_of_utterance_delay": round(metrics.end_of_utterance_delay, 2),
                "transcription_delay": round(metrics.transcription_delay, 2),
            },
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(
            "STT metrics",
            extra=metadata
            | {
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
