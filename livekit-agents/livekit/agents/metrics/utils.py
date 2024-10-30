import logging
from typing import Callable, Optional, TypedDict

from livekit.agents.metrics import AgentMetrics

from ..log import logger as default_logger


class UsageSummary(TypedDict):
    llm_prompt_tokens: int
    llm_completion_tokens: int
    tts_characters_count: int
    stt_audio_duration: float


def create_metrics_logger(
    logger: Optional[logging.Logger] = None,
) -> Callable[[AgentMetrics], None]:
    if logger is None:
        logger = default_logger

    def log_metrics(metrics: AgentMetrics):
        metrics_type = metrics["type"]
        if metrics_type == "vad_metrics":
            return  # don't log VAD metrics because it is noisy
        elif metrics_type == "llm_metrics":
            sequence_id = metrics["sequence_id"]
            ttft = metrics["ttft"]
            tokens_per_second = metrics["tokens_per_second"]

            logger.info(
                f"LLM metrics: sequence_id={sequence_id}, ttft={ttft:.2f}, input_tokens={metrics['prompt_tokens']}, output_tokens={metrics['completion_tokens']}, tokens_per_second={tokens_per_second:.2f}"
            )
        elif metrics_type == "tts_metrics":
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
            logger.info(f"STT metrics: audio_duration={metrics['audio_duration']:.2f}")

    return log_metrics


def create_summary_collector() -> tuple[Callable[[AgentMetrics], None], UsageSummary]:
    summary = UsageSummary()

    def collect_metrics(metrics: AgentMetrics):
        nonlocal summary
        metrics_type = metrics["type"]
        if metrics_type == "vad_metrics":
            return  # don't log VAD metrics because it is noisy
        elif metrics_type == "llm_metrics":
            if "llm_prompt_tokens" not in summary:
                summary["llm_prompt_tokens"] = 0
            if "llm_completion_tokens" not in summary:
                summary["llm_completion_tokens"] = 0
            summary["llm_prompt_tokens"] += metrics["prompt_tokens"]
            summary["llm_completion_tokens"] += metrics["completion_tokens"]
        elif metrics_type == "tts_metrics":
            if "tts_characters_count" not in summary:
                summary["tts_characters_count"] = 0
            summary["tts_characters_count"] += metrics["characters_count"]
        elif metrics_type == "stt_metrics":
            if "stt_audio_duration" not in summary:
                summary["stt_audio_duration"] = 0
            summary["stt_audio_duration"] += metrics["audio_duration"]

    return collect_metrics, summary
