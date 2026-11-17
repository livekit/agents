from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import soxr  # type: ignore[import-untyped]

from livekit import rtc
from livekit.agents.language import LanguageCode
from livekit.agents.llm import ChatContext
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.turn import TurnDetectionEvent
from marimba import Marimba, Question
from marimba.audio import SAMPLE_RATE

from .log import logger

_TURN_END_QUESTION = Question.score("Has the speaker finished their turn?", task="turn_end")


class TurnDetector:
    """Audio-only streaming turn detector for LiveKit Agents."""

    def __init__(
        self,
        *,
        model: Marimba | None = None,
        unlikely_threshold: float = 0.5,
        prediction_timeout: float = 2.0,
    ) -> None:
        """Create an audio-only detector.

        Args:
            model: Preloaded Marimba model. If omitted, load the default model in
                a background thread when the first stream opens.
            unlikely_threshold: Below this probability, wait for the maximum
                endpointing delay. Must be between 0 and 1.
            prediction_timeout: Seconds LiveKit waits for a prediction before
                committing without one. Must be greater than zero.
        """
        if not 0.0 <= unlikely_threshold <= 1.0:
            raise ValueError("unlikely_threshold must be between 0 and 1")
        if not prediction_timeout > 0:
            raise ValueError("prediction_timeout must be greater than 0")
        self._model = model
        self._threshold = unlikely_threshold
        self._prediction_timeout = prediction_timeout
        self._lock = threading.Lock()
        self._warmup_start_lock = threading.Lock()
        self._warmup_started = False

    @property
    def model(self) -> str:
        """Hugging Face model identifier."""
        return "charcot-ai/marimba"

    @property
    def provider(self) -> str:
        """Provider identifier used in telemetry."""
        return "marimba"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TurnDetectorStream:
        """Open a per-session stream and start background model loading.

        The model runs locally, so connection options are unused. Opening a
        stream does not require a running event loop.
        """
        with self._warmup_start_lock:
            if self._model is None and not self._warmup_started:
                self._warmup_started = True
                threading.Thread(target=self._warmup, daemon=True).start()
        return TurnDetectorStream(self)

    def _warmup(self) -> None:
        try:
            with self._lock:
                if self._model is None:
                    self._model = Marimba.from_pretrained()
        except Exception:
            logger.exception("Marimba model warmup failed")

    async def unlikely_threshold(self, language: LanguageCode | None) -> float:
        """Return the configured language-independent turn-end threshold."""
        return self._threshold

    async def supports_language(self, language: LanguageCode | None) -> bool:
        """Accept any language code; the model uses audio, not transcripts."""
        return True

    def _predict(self, audio: np.ndarray[Any, Any], sample_rate: int) -> float:
        if sample_rate != SAMPLE_RATE:
            audio = soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="HQ")
        audio = audio[-8 * SAMPLE_RATE :]
        with self._lock:
            if self._model is None:
                self._model = Marimba.from_pretrained()
            decisions = self._model(audio, [_TURN_END_QUESTION], sampling_rate=SAMPLE_RATE)
        if not decisions or decisions[0].probability is None:
            raise ValueError("Marimba did not return a turn-end probability")
        return float(decisions[0].probability)


class TurnDetectorStream:
    """Per-session audio buffer and asynchronous turn-end predictions."""

    def __init__(self, detector: TurnDetector) -> None:
        """Create a stream sharing the detector's model and inference lock."""
        self._detector = detector
        self._audio = np.empty(0, dtype=np.float32)
        self._sample_rate: int | None = None
        self._closed = False
        self._future: asyncio.Future[TurnDetectionEvent] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending: (
            tuple[asyncio.Future[TurnDetectionEvent], np.ndarray[Any, Any], int] | None
        ) = None

    @property
    def model(self) -> str:
        """Model identifier used by this stream."""
        return self._detector.model

    @property
    def provider(self) -> str:
        """Provider identifier used by this stream."""
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        """Whether the stream has switched to a fallback provider."""
        return False

    @property
    def prediction_timeout(self) -> float:
        """Maximum time LiveKit waits for a prediction, in seconds."""
        return self._detector._prediction_timeout

    async def unlikely_threshold(self, language: LanguageCode | None) -> float:
        """Return the detector's turn-end threshold."""
        return await self._detector.unlikely_threshold(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> None:
        """Return None because this detector does not score backchannels."""
        return None

    async def supports_language(self, language: LanguageCode | None) -> bool:
        """Accept any language code for audio-only inference."""
        return await self._detector.supports_language(language)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Retain the latest eight seconds of PCM audio, mixing channels to mono.

        Sample rate must remain constant until flush. Frames received after
        end_input are ignored.
        """
        if self._closed:
            return
        if self._sample_rate is not None and frame.sample_rate != self._sample_rate:
            raise ValueError("Audio sample rate must remain constant until flush")
        self._sample_rate = frame.sample_rate
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(-1, frame.num_channels).mean(axis=1)
        self._audio = np.concatenate((self._audio, audio))[-8 * frame.sample_rate :]

    @staticmethod
    def _event(probability: float, duration: float | None = None) -> TurnDetectionEvent:
        return TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=probability,
            last_speaking_time=time.time(),
            inference_duration=duration,
        )

    def predict(self, chat_ctx: ChatContext | None = None) -> asyncio.Future[TurnDetectionEvent]:
        """Score a snapshot off the event loop, superseding any pending request.

        Empty or closed streams resolve immediately with probability 1.0.
        Model failures also resolve with 1.0, allowing the turn to commit.
        Chat context is unused by this audio-only detector.
        """
        self.cancel_inference()
        future: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
        if self._closed or not self._audio.size:
            future.set_result(self._event(1.0))
            return future
        self._future = future
        self._pending = future, self._audio.copy(), self._sample_rate or SAMPLE_RATE
        if not any(not task.done() for task in self._tasks):
            task = asyncio.create_task(self._run_predictions())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return future

    async def _run_predictions(self) -> None:
        while self._pending is not None:
            pending, self._pending = self._pending, None
            await self._run_prediction(*pending)

    async def _run_prediction(
        self,
        future: asyncio.Future[TurnDetectionEvent],
        audio: np.ndarray[Any, Any],
        sample_rate: int,
    ) -> None:
        start = time.monotonic()
        try:
            probability = await asyncio.to_thread(self._detector._predict, audio, sample_rate)
        except Exception:
            logger.exception("Marimba turn prediction failed")
            probability = 1.0
        if not future.done():
            future.set_result(self._event(probability, time.monotonic() - start))

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        """Resolve the pending request with 0.0 and ignore its eventual result.

        Running model calls cannot be interrupted. The timed_out flag does not
        change local inference behavior.
        """
        if self._future is not None and not self._future.done():
            self._future.set_result(self._event(0.0))
        self._future = None
        self._pending = None

    def flush(self, reason: str | None = None) -> None:
        """Clear retained audio and cancel the pending prediction."""
        self.cancel_inference()
        self._audio = np.empty(0, dtype=np.float32)
        self._sample_rate = None

    def end_input(self) -> None:
        """Clear the buffer and stop accepting audio."""
        self.flush()
        self._closed = True

    def on_start_of_turn(self, started_at: float) -> None:
        """Accept a turn-start notification; audio inference needs no text timing."""

    async def aclose(self) -> None:
        """End input and wait for outstanding worker-thread predictions."""
        self.end_input()
        # A running model call cannot be interrupted safely in its worker thread.
        if self._tasks:
            await asyncio.gather(*self._tasks)
            self._tasks.clear()
