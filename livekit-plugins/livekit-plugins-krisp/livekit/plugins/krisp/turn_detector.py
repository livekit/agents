# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Krisp VIVA Turn Taking (TT) as a LiveKit streaming turn detector.

:class:`TurnDetector` satisfies the ``_StreamingTurnDetector`` protocol, so it
is passed straight to ``turn_handling={"turn_detection": ...}``. The session
pushes every user audio frame to the stream and calls ``predict()`` once the
user pauses; the Krisp model's turn-shift probability is returned as the
end-of-turn probability.

Krisp TT runs in-process on the public ``krisp_audio`` wheel (license mode)
and needs a Krisp VAD model to supply its per-frame voice-activity input.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import aio
from livekit.agents.voice.turn import TurnDetectionEvent

from ._sdk import (
    _FrameChunker,
    _SessionPair,
    frame_samples,
    open_in_thread,
    resolve_license_key,
    resolve_model_path,
)
from .log import logger

DEFAULT_PREDICTION_TIMEOUT = 1.0
DEFAULT_MAX_PREDICTION_DELAY = 0.8


@dataclass
class _TurnDetectorOptions:
    model_path: str
    vad_model_path: str
    license_key: str
    unlikely_threshold: float
    vad_threshold: float
    frame_duration_ms: int
    prediction_timeout: float
    max_prediction_delay: float


@dataclass
class _PredictRequest:
    fut: asyncio.Future[TurnDetectionEvent]
    requested_at: float = field(default_factory=time.time)
    armed_index: int | None = None  # frame index when the worker reached the request


class _ResetSentinel:
    pass


class TurnDetector:
    """End-of-turn detector backed by Krisp VIVA Turn Taking.

    Example:
        ```python
        from livekit.agents import AgentSession, inference
        from livekit.plugins import krisp

        session = AgentSession(
            vad=inference.VAD(),
            stt=...,
            llm=...,
            tts=...,
            turn_handling={"turn_detection": krisp.TurnDetector()},
        )
        ```
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        vad_model_path: str | None = None,
        license_key: str | None = None,
        unlikely_threshold: float = 0.5,
        vad_threshold: float = 0.5,
        frame_duration_ms: int = 20,
        prediction_timeout: float = DEFAULT_PREDICTION_TIMEOUT,
        max_prediction_delay: float = DEFAULT_MAX_PREDICTION_DELAY,
    ) -> None:
        """
        Args:
            model_path: Krisp TT model (``.kef``). Defaults to ``KRISP_VIVA_TT_MODEL_PATH``.
            vad_model_path: Krisp VAD model (``.kef``) feeding TT's voice-activity input.
                Defaults to ``KRISP_VIVA_VAD_MODEL_PATH``.
            license_key: Krisp license key. Defaults to ``KRISP_VIVA_SDK_LICENSE_KEY``.
            unlikely_threshold: Below this end-of-turn probability the session waits the
                maximum endpointing delay before committing the user's turn.
            vad_threshold: Krisp VAD probability at or above which a frame counts as voice.
            frame_duration_ms: Frame size fed to the Krisp sessions (10, 15, 20, 30 or 32).
            prediction_timeout: Seconds the session waits for a prediction before
                committing without one.
            max_prediction_delay: Krisp TT's probability keeps rising while the user stays
                silent. A prediction is answered as soon as it reaches
                ``unlikely_threshold``, or with its latest value after this many seconds of
                audio. Must be below ``prediction_timeout``.
        """
        if max_prediction_delay >= prediction_timeout:
            raise ValueError("max_prediction_delay must be below prediction_timeout")
        self._opts = _TurnDetectorOptions(
            model_path=resolve_model_path(model_path, "KRISP_VIVA_TT_MODEL_PATH", "TT"),
            vad_model_path=resolve_model_path(vad_model_path, "KRISP_VIVA_VAD_MODEL_PATH", "VAD"),
            license_key=resolve_license_key(license_key),
            unlikely_threshold=unlikely_threshold,
            vad_threshold=vad_threshold,
            frame_duration_ms=frame_duration_ms,
            prediction_timeout=prediction_timeout,
            max_prediction_delay=max_prediction_delay,
        )

    @property
    def model(self) -> str:
        return "krisp-viva-tt"

    @property
    def provider(self) -> str:
        return "krisp"

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.unlikely_threshold

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return None

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return True

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TurnDetectorStream:
        return TurnDetectorStream(detector=self, opts=self._opts)


class TurnDetectorStream:
    """Per-session TT stream: feeds every user frame to Krisp and answers ``predict()``.

    Krisp TT reports 0 while the user speaks. About 200 ms into the following silence it
    produces a turn-shift probability that keeps rising while the silence lasts. A
    ``predict()`` request therefore resolves as soon as that probability reaches the
    unlikely threshold, or with its latest value after ``max_prediction_delay`` of audio.
    """

    def __init__(self, *, detector: TurnDetector, opts: _TurnDetectorOptions) -> None:
        self._detector = detector
        self._opts = opts
        self._pair: _SessionPair | None = None  # loaded off the event loop by _run
        self._chunker = _FrameChunker(frame_samples(opts.frame_duration_ms))
        self._ch = aio.Chan[list[np.ndarray] | _PredictRequest | _ResetSentinel]()

        self._request: _PredictRequest | None = None  # latest predict(), until resolved
        self._armed: _PredictRequest | None = None  # the request the worker has reached
        self._reset_next = False
        self._result_ready = False  # a post-speech probability exists since the last voice
        self._last_probability = 0.0

        self._budget_frames = round(opts.max_prediction_delay * 1000 / opts.frame_duration_ms)
        self._frame_index = 0

        self._task = asyncio.create_task(self._run(), name="krisp.TurnDetectorStream")

    # region: streaming turn detector protocol

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return self._opts.prediction_timeout

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.unlikely_threshold(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.backchannel_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Request an end-of-turn prediction for the current pause."""
        fut: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
        if self._ch.closed:
            fut.set_result(_event(1.0))
            return fut

        self.cancel_inference()  # supersede any previous request
        self._request = _PredictRequest(fut=fut)
        self._ch.send_nowait(self._request)
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        request = self._request
        self._request = None
        self._armed = None
        if request is not None and not request.fut.done():
            request.fut.set_result(_event(0.0))

    def flush(self, reason: str | None = None) -> None:
        """Turn boundary: reset Krisp TT's state before the next user turn."""
        self.cancel_inference()
        if not self._ch.closed:
            self._ch.send_nowait(_ResetSentinel())

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._ch.closed:
            return
        frames = self._chunker.push(frame)
        if frames:
            self._ch.send_nowait(frames)

    def end_input(self) -> None:
        self.flush()
        self._ch.close()

    async def aclose(self) -> None:
        self.end_input()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            await aio.cancel_and_wait(self._task)
        self.cancel_inference()
        if self._pair is not None:
            self._pair.close()

    # endregion

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        try:
            self._pair = await open_in_thread(
                lambda: _SessionPair(
                    kind="tt",
                    license_key=self._opts.license_key,
                    model_path=self._opts.model_path,
                    vad_model_path=self._opts.vad_model_path,
                    vad_threshold=self._opts.vad_threshold,
                    frame_duration_ms=self._opts.frame_duration_ms,
                )
            )
        except Exception:
            logger.exception("failed to load the Krisp TT models; turn detection is disabled")
            self._disable()
            return

        try:
            await self._process_input()
        except Exception:
            logger.exception("Krisp TT processing failed; turn detection is disabled")
            self._disable()

    def _disable(self) -> None:
        # predict() on a closed stream answers 1.0 immediately, so turns commit on the min delay
        self._ch.close()
        self._armed = None
        request, self._request = self._request, None
        if request is not None and not request.fut.done():
            request.fut.set_result(_event(1.0))

    async def _process_input(self) -> None:
        async for item in self._ch:
            if isinstance(item, _ResetSentinel):
                self._reset_next = True
                self._result_ready = False
            elif isinstance(item, _PredictRequest):
                if item is self._request and not item.fut.done():
                    self._armed = item
                    item.armed_index = self._frame_index
                    if self._result_ready and self._last_probability >= self._threshold:
                        self._resolve(self._last_probability)
            else:
                reset, self._reset_next = self._reset_next, False
                results = await asyncio.to_thread(self._process, item, reset)
                for voice, probability in results:
                    self._on_frame(voice, probability)

    def _on_frame(self, voice: bool, probability: float) -> None:
        index = self._frame_index
        self._frame_index += 1

        if voice:
            self._result_ready = False
        else:
            if not self._result_ready and probability > 0.0:
                self._result_ready = True
            self._last_probability = probability

        request = self._armed
        if request is None or request.armed_index is None:
            return
        if self._result_ready and probability >= self._threshold:
            self._resolve(probability)
        elif index - request.armed_index + 1 >= self._budget_frames:
            # no confident answer within the budget: report the latest post-speech value
            self._resolve(self._last_probability if self._result_ready else 0.0)

    @property
    def _threshold(self) -> float:
        return self._opts.unlikely_threshold

    def _process(self, frames: list[np.ndarray], reset: bool) -> list[tuple[bool, float]]:
        assert self._pair is not None
        # botSpeaking=True on a single frame resets TT's preprocessor between turns
        return [
            self._pair.process(frame, bot_speaking=reset and i == 0)
            for i, frame in enumerate(frames)
        ]

    def _resolve(self, probability: float) -> None:
        request = self._armed
        self._armed = None
        if request is None or request is not self._request:
            return
        self._request = None
        if not request.fut.done():
            request.fut.set_result(
                _event(probability, detection_delay=time.time() - request.requested_at)
            )


def _event(probability: float, *, detection_delay: float | None = None) -> TurnDetectionEvent:
    return TurnDetectionEvent(
        type="eot_prediction",
        last_speaking_time=time.time(),
        end_of_turn_probability=probability,
        detection_delay=detection_delay,
    )
