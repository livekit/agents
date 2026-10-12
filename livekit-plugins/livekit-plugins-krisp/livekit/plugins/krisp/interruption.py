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

"""Krisp VIVA Interruption Prediction (IP) as a LiveKit interruption detector.

:class:`InterruptionDetector` is passed as ``turn_handling={"interruption":
{"mode": detector}}``. While the agent speaks, every user frame runs through
Krisp IP with ``botSpeaking=True``; when the user's overlapping speech reaches
the probability threshold it is reported as a genuine interruption, otherwise
the overlap is reported as a backchannel when it ends.

Krisp IP runs in-process on the public ``krisp_audio`` wheel (license mode)
and needs a Krisp VAD model to supply its per-frame voice-activity input.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from livekit import rtc
from livekit.agents import inference, utils
from livekit.agents.inference.interruption import (
    InterruptionDataFrameType,
    InterruptionDetectionError,
    OverlappingSpeechEvent,
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.metrics.base import InterruptionMetrics, Metadata
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import aio

from ._sdk import (
    _FrameChunker,
    _SessionPair,
    frame_samples,
    open_in_thread,
    resolve_license_key,
    resolve_model_path,
)
from .log import logger


@dataclass
class _InterruptionOptions:
    model_path: str
    vad_model_path: str
    license_key: str
    threshold: float
    vad_threshold: float
    frame_duration_ms: int


class InterruptionDetector(inference.InterruptionDetector):
    """Interruption detector backed by Krisp VIVA Interruption Prediction.

    Example:
        ```python
        from livekit.agents import AgentSession, inference
        from livekit.plugins import krisp

        session = AgentSession(
            vad=inference.VAD(),
            stt=...,
            llm=...,
            tts=...,
            turn_handling={"interruption": {"mode": krisp.InterruptionDetector()}},
        )
        ```
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        vad_model_path: str | None = None,
        license_key: str | None = None,
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
        frame_duration_ms: int = 20,
    ) -> None:
        """
        Args:
            model_path: Krisp IP model (``.kef``). Defaults to ``KRISP_VIVA_IP_MODEL_PATH``.
            vad_model_path: Krisp VAD model (``.kef``) feeding IP's voice-activity input.
                Defaults to ``KRISP_VIVA_VAD_MODEL_PATH``.
            license_key: Krisp license key. Defaults to ``KRISP_VIVA_SDK_LICENSE_KEY``.
            threshold: Interruption probability at or above which overlapping speech
                is a genuine interruption. Krisp IP reports 1.0 for words on its
                interruption list and 0.0 for words on its backchannel list; for other
                speech the probability grows with its duration and phoneme count (about
                0.33 per second with the stock model). The threshold therefore sets how long
                unrecognized speech may overlap the agent before interrupting it: 0.5 is
                roughly 1.5 s, 0.3 roughly 0.9 s.
            vad_threshold: Krisp VAD probability at or above which a frame counts as voice.
            frame_duration_ms: Frame size fed to the Krisp sessions (10, 15, 20, 30 or 32).
        """
        super().__init__()
        self._opts = _InterruptionOptions(
            model_path=resolve_model_path(model_path, "KRISP_VIVA_IP_MODEL_PATH", "IP"),
            vad_model_path=resolve_model_path(vad_model_path, "KRISP_VIVA_VAD_MODEL_PATH", "VAD"),
            license_key=resolve_license_key(license_key),
            threshold=threshold,
            vad_threshold=vad_threshold,
            frame_duration_ms=frame_duration_ms,
        )

    @property
    def model(self) -> str:
        return "krisp-viva-ip"

    @property
    def provider(self) -> str:
        return "krisp"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> InterruptionDetectorStream:
        return InterruptionDetectorStream(detector=self, opts=self._opts)


class InterruptionDetectorStream:
    """Per-session IP stream emitting one :class:`OverlappingSpeechEvent` per overlap."""

    def __init__(self, *, detector: InterruptionDetector, opts: _InterruptionOptions) -> None:
        self._detector = detector
        self._opts = opts
        self._pair: _SessionPair | None = None  # loaded off the event loop by _run
        self._chunker = _FrameChunker(frame_samples(opts.frame_duration_ms))
        self._input_ch = aio.Chan[InterruptionDataFrameType]()
        self._event_ch = aio.Chan[OverlappingSpeechEvent]()

        self._bot_speaking = False
        self._overlap_started_at: float | None = None  # set while an overlap awaits a verdict
        self._last_probability = 0.0
        self._prediction_duration = 0.0

        self._task = asyncio.create_task(self._run(), name="krisp.InterruptionDetectorStream")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    def push_frame(self, frame: InterruptionDataFrameType) -> None:
        if self._input_ch.closed:
            raise RuntimeError("krisp.InterruptionDetectorStream input ended")
        self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        self._input_ch.close()

    async def aclose(self) -> None:
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
        if self._pair is not None:
            self._pair.close()

    def __aiter__(self) -> AsyncIterator[OverlappingSpeechEvent]:
        return self

    async def __anext__(self) -> OverlappingSpeechEvent:
        try:
            return await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None
            raise

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        try:
            self._pair = await open_in_thread(
                lambda: _SessionPair(
                    kind="ip",
                    license_key=self._opts.license_key,
                    model_path=self._opts.model_path,
                    vad_model_path=self._opts.vad_model_path,
                    vad_threshold=self._opts.vad_threshold,
                    frame_duration_ms=self._opts.frame_duration_ms,
                )
            )
        except Exception as e:
            logger.exception("failed to load the Krisp IP models")
            await self._fail(e)
            return

        try:
            await self._process_input()
        except Exception as e:
            logger.exception("Krisp IP processing failed")
            await self._fail(e)

    async def _process_input(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, _AgentSpeechStartedSentinel | _AgentSpeechEndedSentinel):
                # on the end transition the SDK also resets IP's own state
                self._bot_speaking = isinstance(item, _AgentSpeechStartedSentinel)
                self._overlap_started_at = None
                self._last_probability = 0.0
            elif isinstance(item, _OverlapSpeechStartedSentinel):
                if self._bot_speaking:
                    # judged on the next scored frame: _last_probability may predate this overlap
                    self._overlap_started_at = item._started_at
            elif isinstance(item, _OverlapSpeechEndedSentinel):
                if self._overlap_started_at is not None:
                    self._emit(
                        is_interruption=False,
                        detected_at=item._ended_at,
                        agent_ended=item._agent_ended,
                    )
            elif isinstance(item, rtc.AudioFrame):
                frames = self._chunker.push(item)
                if not frames:
                    continue
                t0 = time.perf_counter()
                probabilities = await asyncio.to_thread(self._process, frames, self._bot_speaking)
                self._prediction_duration = time.perf_counter() - t0
                for probability in probabilities:
                    self._last_probability = probability
                    if self._overlap_started_at is not None and probability >= self._opts.threshold:
                        self._emit(is_interruption=True)

    async def _fail(self, error: Exception) -> None:
        # an unrecoverable error makes the session fall back to VAD interruptions
        self._detector.emit(
            "error",
            InterruptionDetectionError(label=self._detector.label, error=error, recoverable=False),
        )
        async for _ in self._input_ch:
            pass  # keep accepting frames until the session detaches the stream

    def _process(self, frames: list[np.ndarray], bot_speaking: bool) -> list[float]:
        assert self._pair is not None
        return [self._pair.process(frame, bot_speaking)[1] for frame in frames]

    def _emit(
        self,
        *,
        is_interruption: bool,
        detected_at: float | None = None,
        agent_ended: bool = False,
    ) -> None:
        started_at = self._overlap_started_at
        self._overlap_started_at = None  # one verdict per overlap
        detected_at = detected_at or time.time()
        ev = OverlappingSpeechEvent(
            detected_at=detected_at,
            is_interruption=is_interruption,
            agent_ended=agent_ended,
            overlap_started_at=started_at,
            probability=self._last_probability,
            prediction_duration=self._prediction_duration,
            total_duration=self._prediction_duration,
            detection_delay=detected_at - started_at if started_at is not None else 0.0,
            num_requests=1,
        )
        if is_interruption:
            logger.debug(
                "krisp interruption detected",
                extra={"probability": ev.probability, "detection_delay": ev.detection_delay},
            )
        self._event_ch.send_nowait(ev)
        self._detector.emit("overlapping_speech", ev)
        self._detector.emit(
            "metrics_collected",
            InterruptionMetrics(
                timestamp=time.time(),
                total_duration=ev.total_duration,
                prediction_duration=ev.prediction_duration,
                detection_delay=ev.detection_delay,
                num_interruptions=1 if is_interruption else 0,
                num_backchannels=1 if not is_interruption and not agent_ended else 0,
                num_requests=ev.num_requests,
                metadata=Metadata(
                    model_name=self._detector.model, model_provider=self._detector.provider
                ),
            ),
        )
