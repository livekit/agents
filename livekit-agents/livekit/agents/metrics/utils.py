import logging
from dataclasses import dataclass
from typing import Callable

from livekit.agents.metrics import PipelineMetrics

from ..log import logger as default_logger


@dataclass
class PipelineUsageSummary:
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    tts_characters_count: int = 0
    stt_audio_duration: float = 0


def create_metrics_logger(
    logger: logging.Logger | None = None,
) -> tuple[Callable[[PipelineMetrics], PipelineUsageSummary], None]:
    if logger is None:
        logger = default_logger

    summary = PipelineUsageSummary()

    def log_metrics(metrics: PipelineMetrics):
        nonlocal summary
        metrics_type = metrics["type"]
        if metrics_type == "vad_metrics":
            return  # don't log VAD metrics because it is noisy
        elif metrics_type == "llm_metrics":
            summary.llm_prompt_tokens += metrics["prompt_tokens"]
            summary.llm_completion_tokens += metrics["completion_tokens"]

            sequence_id = metrics["sequence_id"]
            ttft = metrics["ttft"]
            tokens_per_second = metrics["tokens_per_second"]

            logger.info(
                f"LLM metrics: sequence_id={sequence_id}, ttft={ttft:.2f}, input_tokens={metrics['prompt_tokens']}, output_tokens={metrics['completion_tokens']}, tokens_per_second={tokens_per_second:.2f}"
            )
        elif metrics_type == "tts_metrics":
            summary.tts_characters_count += metrics["characters_count"]

            sequence_id = metrics["sequence_id"]
            ttfb = metrics["ttfb"]
            audio_duration = metrics["audio_duration"]

            logger.info(
                f"TTS metrics: sequence_id={sequence_id}, ttfb={ttfb}, audio_duration={audio_duration:.2f}"
            )
        elif metrics_type == "eou_metrics":
            sequence_id = metrics["sequence_id"]
            end_of_utterance_delay = metrics["end_of_utterance_delay"]
            transcription_delay = metrics["transcription_delay"]

            logger.info(
                f"EOU metrics: sequence_id={sequence_id}, end_of_utterance_delay={end_of_utterance_delay:.2f}, transcription_delay={transcription_delay:.2f}"
            )
        elif metrics_type == "stt_metrics":
            summary.stt_audio_duration += metrics["audio_duration"]

            logger.info(f"STT metrics: audio_duration={metrics['audio_duration']:.2f}")

    return log_metrics, summary
