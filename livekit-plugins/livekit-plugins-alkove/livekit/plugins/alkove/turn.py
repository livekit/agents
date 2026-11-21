# Copyright 2026 Eddy Bordi
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

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

import alkove
from livekit import rtc
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.turn import TurnDetectionEvent

from ._models import models as shared_models
from .log import logger


class TurnDetector:
    """Alkove Cue's end of turn head as the session's turn detector, from the user's audio.

    Focus then Cue run on the agent's CPU and read the head every 0.1 s of each pause. A
    prediction resolves on the first read over `threshold` once the pause reaches `min_pause`,
    or at `max_pause` with the head's last word; the session then waits its endpointing delay,
    the short one over `threshold`, the long one under it.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        min_pause: float = 0.3,
        max_pause: float = 1.5,
        models: alkove.Models | None = None,
    ) -> None:
        """Create a turn detector from models already loaded.

        Args:
            threshold: The end of turn probability over which the turn is over.
            min_pause: Seconds of pause before any read can end the turn.
            max_pause: Seconds of pause after which the head's last read is the answer.
            models: The models to use, with Focus and Cue. Defaults to those `download-files`
                put in the cache, shared by every session of the process.
        """
        self.params = alkove.TurnParams(threshold, min_pause, max_pause)
        self._models = models or shared_models(download=False)

    @property
    def model(self) -> str:
        return str(alkove.MANIFEST["models"]["cue"]["name"])

    @property
    def provider(self) -> str:
        return "alkove"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TurnDetectorStream:
        """Create a stream for one audio track."""
        return TurnDetectorStream(self)


class TurnDetectorStream:
    """One track's audio through Focus and Cue, and the predictions the session asks for."""

    def __init__(self, detector: TurnDetector) -> None:
        self._detector = detector
        self._live: alkove.Live | None = None
        self._rate = 0
        self._fut: asyncio.Future[TurnDetectionEvent] | None = None
        self._ended: TurnDetectionEvent | None = None
        self._closed = False

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
        return self._detector.params.max_pause + 0.5

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._detector.params.threshold

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return None

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return True

    # chat_ctx since livekit-agents 1.8.13: Cue reads the audio only
    def predict(self, chat_ctx: Any = None) -> asyncio.Future[TurnDetectionEvent]:
        fut, self._fut = self._fut, None
        if fut is not None and not fut.done():
            fut.set_result(_event(0.0, time.time()))
        self._fut = asyncio.get_running_loop().create_future()
        if self._closed or self._ended is not None:
            self._fut.set_result(self._ended or _event(1.0, time.time()))
        return self._fut

    # livekit-agents 1.8.13 tells when the user's turn started: Cue finds its turns in the audio
    def on_start_of_turn(self, started_at: float) -> None:
        pass

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        # the session cancels when speech comes back: the last pause's answer no longer holds
        self._ended = None
        fut, self._fut = self._fut, None
        if fut is not None and not fut.done():
            fut.set_result(_event(0.0, time.time()))

    def flush(self, reason: str | None = None) -> None:
        self.cancel_inference()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if self._live is None:
            self._rate = frame.sample_rate
            self._live = alkove.Live(self._detector._models, self._rate, turn=self._detector.params)
        elif frame.sample_rate != self._rate:
            logger.error("a frame with another sample rate was already pushed")
            return
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = alkove.to_int16(
                pcm.reshape(-1, frame.num_channels).astype(np.float64).mean(axis=1) / 32768
            )
        _, events = self._live.push(pcm)
        for e in events:
            if e["type"] == "speechstart" or e["type"] == "turn" and e["pause"] < 0.15:
                # new speech or a new pause: what the last one decided no longer holds
                self._ended = None
            elif e["type"] == "turnend":
                self._ended = _event(e["probability"], time.time() - (e["at"] - e["end"]))
                if self._fut is not None and not self._fut.done():
                    self._fut.set_result(self._ended)

    def end_input(self) -> None:
        self._closed = True
        self.cancel_inference()

    async def aclose(self) -> None:
        self.end_input()


def _event(probability: float, last_speaking_time: float) -> TurnDetectionEvent:
    return TurnDetectionEvent(
        type="eot_prediction",
        end_of_turn_probability=probability,
        last_speaking_time=last_speaking_time,
    )
