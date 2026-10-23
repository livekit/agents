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

import math
from typing import Any

import numpy as np

import alkove
from livekit import rtc

from ._models import models as shared_models
from .log import logger

# LiveKit's gain control needs whole 10 ms frames: a hop of Focus held ahead, about 37 ms in all
LEAD = 0.02


class Focus(rtc.FrameProcessor[rtc.AudioFrame]):
    """Keeps the user's voice and removes the other voices and the noise.

    It keeps the closest voice, or the voice of a print made with `alkove.Timbre`. Pass it
    as `room_io.AudioInputOptions(noise_cancellation=alkove.Focus())`. It works at the
    session's sample rate and gives each frame back at the size it came in, about 37 ms
    behind.
    """

    def __init__(
        self,
        *,
        models: alkove.Models | None = None,
        voiceprint: Any = None,
        strength: float = 1.0,
        normalize: bool = False,
    ) -> None:
        """Create a Focus processor.

        Args:
            models: The models to use, with Focus and Cue. Defaults to those `download-files`
                put in the cache, shared by every session of the process.
            voiceprint: The print of the voice to keep (`alkove.Voiceprint` or its
                embedding). Defaults to the closest voice.
            strength: How much of Focus goes out, from 0 to 1, the rest being the raw
                microphone.
            normalize: Bring the voice that goes out to a steady level.

        Raises:
            ValueError: `models` lacks Focus or Cue.
        """
        self._models = models or shared_models(download=False)
        missing = [n for n in ("focus", "cue") if n not in self._models]
        if missing:
            raise ValueError(f"alkove.Focus needs the models {', '.join(missing)}")
        self._voiceprint = voiceprint
        self._strength = strength
        self._normalize = normalize
        self._enabled = True
        # pre-connect audio comes through the same processor as the live track, at its own rate
        self._streams: dict[int, tuple[alkove.Live, np.ndarray]] = {}
        self._track: str | None = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        """Whether LiveKit runs the frames through Focus."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        # LiveKit skips a disabled processor: it comes back on a fresh stream
        if value and not self._enabled:
            self._streams.clear()
        self._enabled = value

    def set_voiceprint(self, embedding: Any) -> None:
        """Keep the voice of another print, from the user's next silence.

        Args:
            embedding: The print (`alkove.Voiceprint` or its embedding), or `None` to go
                back to the closest voice.
        """
        self._voiceprint = embedding
        for live, _ in self._streams.values():
            live.set_voiceprint(embedding)

    def set_strength(self, s: float) -> None:
        """Set how much of Focus goes out.

        Args:
            s: From 0 to 1, the rest being the raw microphone: 0.8 is 80 % Focus.
        """
        self._strength = s
        for live, _ in self._streams.values():
            live.set_strength(s)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        rate, ch, n = frame.sample_rate, frame.num_channels, frame.samples_per_channel
        if rate not in self._streams:
            live = alkove.Live(
                self._models,
                rate,
                normalize=self._normalize,
                strength=self._strength,
                voiceprint=self._voiceprint,
            )
            self._streams[rate] = (live, np.zeros(math.ceil(rate * LEAD), dtype=np.float32))
        live, held = self._streams[rate]
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        mono = pcm if ch == 1 else pcm.reshape(-1, ch).astype(np.float64).mean(axis=1) / 32768
        held = np.concatenate((held, live.push(mono)[0]))
        if len(held) < n:
            if not self._warned:
                self._warned = True
                logger.warning("Focus gave less than a frame: silence put in its place")
            held = np.concatenate((np.zeros(n - len(held), dtype=np.float32), held))
        self._streams[rate] = (live, held[n:])
        data = alkove.to_int16(held[:n])
        if ch > 1:
            data = np.repeat(data, ch)
        return rtc.AudioFrame(
            data=data.tobytes(), sample_rate=rate, num_channels=ch, samples_per_channel=n
        )

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None:
        if publication_sid != self._track:
            self._track = publication_sid
            self._streams.clear()

    def _on_stream_info_cleared(self) -> None:
        self._track = None
        self._streams.clear()

    def _close(self) -> None:
        self._streams.clear()
