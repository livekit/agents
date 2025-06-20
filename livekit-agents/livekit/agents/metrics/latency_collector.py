from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

from .base import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics, AgentMetrics
from ..log import logger as default_logger

@dataclass
class _TurnData:
    start_time: float = 0.0
    stt_duration: float = 0.0
    llm_duration: float = 0.0
    tts_duration: float = 0.0
    audio_duration: float = 0.0

class LatencyCollector:
    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or default_logger
        self._turns: Dict[str, _TurnData] = {}
        self._last_stt: STTMetrics | None = None

    def on_metrics(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, STTMetrics):
            self._last_stt = metrics
        elif isinstance(metrics, EOUMetrics):
            start = (
                metrics.timestamp
                - metrics.end_of_utterance_delay
                - metrics.transcription_delay
                - metrics.on_user_turn_completed_delay
            )
            self._turns[metrics.speech_id or ""] = _TurnData(
                start_time=start,
                stt_duration=self._last_stt.duration if self._last_stt else 0.0,
            )
        elif isinstance(metrics, LLMMetrics):
            data = self._turns.setdefault(metrics.speech_id or "", _TurnData())
            data.llm_duration = metrics.duration
        elif isinstance(metrics, TTSMetrics):
            data = self._turns.setdefault(metrics.speech_id or "", _TurnData())
            data.tts_duration += metrics.duration
            data.audio_duration += metrics.audio_duration

    def on_playback_finished(
        self,
        speech_id: str,
        first_frame_ts: float,
        end_ts: float,
        playback_position: float,
    ) -> None:
        data = self._turns.pop(speech_id, None)
        if data is None:
            return
        telephony_latency = (end_ts - first_frame_ts) - playback_position
        total = end_ts - data.start_time
        network_latency = total - (
            data.stt_duration + data.llm_duration + data.tts_duration + telephony_latency
        )
        self._logger.info(
            "Latency stats",
            extra={
                "speech_id": speech_id,
                "total": round(total, 3),
                "stt": round(data.stt_duration, 3),
                "llm": round(data.llm_duration, 3),
                "tts": round(data.tts_duration, 3),
                "telephony": round(telephony_latency, 3),
                "network": round(network_latency, 3),
            },
        )