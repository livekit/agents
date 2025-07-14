from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import (
    AgentMetrics,
    LLMMetrics,
    PipelineEOUMetrics,
    PipelineLLMMetrics,
    PipelineSTTMetrics,
    PipelineTTSMetrics,
    STTMetrics,
    TTSMetrics,
)


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None):
    if logger is None:
        logger = default_logger

    if isinstance(metrics, PipelineLLMMetrics):
        logger.info(
            f"Pipeline LLM metrics: sequence_id={metrics.sequence_id}, ttft={metrics.ttft:.2f}, input_tokens={metrics.prompt_tokens}, output_tokens={metrics.completion_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"
        )
    elif isinstance(metrics, LLMMetrics):
        logger.info(
            f"LLM metrics: ttft={metrics.ttft:.2f}, input_tokens={metrics.prompt_tokens}, output_tokens={metrics.completion_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"
        )
    elif isinstance(metrics, PipelineTTSMetrics):
        logger.info(
            f"Pipeline TTS metrics: sequence_id={metrics.sequence_id}, ttfb={metrics.ttfb}, audio_duration={metrics.audio_duration:.2f}"
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            f"TTS metrics: ttfb={metrics.ttfb}, audio_duration={metrics.audio_duration:.2f}"
        )
    elif isinstance(metrics, PipelineEOUMetrics):
        logger.info(
            f"Pipeline EOU metrics: sequence_id={metrics.sequence_id}, end_of_utterance_delay={metrics.end_of_utterance_delay:.2f}, transcription_delay={metrics.transcription_delay:.2f}"
        )
    elif isinstance(metrics, PipelineSTTMetrics):
        logger.info(
            f"Pipeline STT metrics: duration={metrics.duration:.2f}, audio_duration={metrics.audio_duration:.2f}"
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(f"STT metrics: audio_duration={metrics.audio_duration:.2f}")
