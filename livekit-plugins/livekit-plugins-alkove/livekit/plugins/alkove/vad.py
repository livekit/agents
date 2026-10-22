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

import time
from collections import deque

import numpy as np

import alkove
from livekit import agents, rtc

from ._models import models as shared_models
from .log import logger

WINDOW = 0.03


class VAD(agents.vad.VAD):
    """Alkove Cue as the session's VAD.

    It gives a speech probability every 30 ms. Speech starts and ends where Alkove's web SDK
    and `alkove.Processor` say so. Behind `Focus`, it hears only the user.
    """

    def __init__(
        self,
        *,
        models: alkove.Models | None = None,
        speech: alkove.SpeechParams | None = None,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
    ) -> None:
        """Create a VAD from models already loaded. Use `VAD.load()` in prewarm.

        Args:
            models: The models to use, with Cue. Defaults to those `download-files` put in the
                cache, shared by every session of the process.
            speech: When speech starts and ends. Defaults to
                `alkove.SpeechParams(confidence=0.5, exit_confidence=0.1, start_secs=0.2,
                stop_secs=0.6)`; `stop_secs` is the turn detector's `min_silence_duration`.
            prefix_padding_duration: Seconds of audio before the start of speech in its
                events' frames.
            max_buffered_speech: Seconds of speech kept at most for the end of speech's
                frames, from its start.

        Raises:
            ValueError: `models` lacks Cue.
        """
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=WINDOW))
        self._models = models or shared_models(download=False)
        if "cue" not in self._models:
            raise ValueError("alkove.VAD needs the model cue")
        self.speech = speech or alkove.SpeechParams()
        self.prefix_padding_duration = prefix_padding_duration
        self.max_buffered_speech = max_buffered_speech

    @classmethod
    def load(
        cls,
        *,
        models: alkove.Models | None = None,
        speech: alkove.SpeechParams | None = None,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
    ) -> VAD:
        """Load the models, downloading them if they are not in the cache, and create a VAD.

        Call it in prewarm. The arguments are those of `VAD()`.
        """
        return cls(
            models=models or shared_models(),
            speech=speech,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
        )

    @property
    def min_silence_duration(self) -> float:
        """The silence that ends speech, `speech.stop_secs`, as the turn detector reads it."""
        return self.speech.stop_secs

    @property
    def model(self) -> str:
        return str(alkove.MANIFEST["models"]["cue"]["name"])

    @property
    def provider(self) -> str:
        return "alkove"

    def stream(self) -> VADStream:
        """Create a stream for one audio track."""
        return VADStream(self)


class _Recent:
    """The last samples of the input, mono int16, by their index since the stream's start."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self.base = 0
        self.end = 0

    def push(self, pcm: np.ndarray) -> None:
        self._chunks.append(pcm)
        self.end += len(pcm)

    def keep(self, n: int) -> None:
        while self._chunks and self.end - self.base - len(self._chunks[0]) >= n:
            self.base += len(self._chunks.popleft())

    def slice(self, a: int, b: int) -> np.ndarray:
        a, b = max(a, self.base), min(b, self.end)
        parts, at = [], self.end
        for c in reversed(self._chunks):
            start = at - len(c)
            if start < b and at > a:
                parts.append(c[max(0, a - start) : b - start])
            if start <= a:
                break
            at = start
        return np.concatenate(parts[::-1]) if parts else np.zeros(0, dtype=np.int16)


class VADStream(agents.vad.VADStream):
    """Cue on one audio track, created by `VAD.stream()`."""

    def __init__(self, vad: VAD) -> None:
        self._opts = vad
        super().__init__(vad)

    async def _main_task(self) -> None:
        v = self._opts
        rate = 0
        processor: alkove.Processor | None = None
        tracker = alkove.SpeechTracker(v.speech, WINDOW)
        recent = _Recent()
        # from the start of speech less the prefix, at most `max_buffered_speech` past the prefix
        speech: list[np.ndarray] = []
        first = room = 0
        full = False
        onset: float | None = None
        speech_acc = silence_acc = 0.0
        speech_pub = silence_pub = 0.0

        def reset() -> None:
            nonlocal processor, tracker, recent, speech, first, room, full, onset
            nonlocal speech_acc, silence_acc, speech_pub, silence_pub
            processor = alkove.Processor(
                v._models, cue=True, speech=v.speech, trace=True, rate=rate
            )
            tracker = alkove.SpeechTracker(v.speech, WINDOW)
            recent = _Recent()
            speech, first, room, full, onset = [], 0, 0, False, None
            speech_acc = silence_acc = speech_pub = silence_pub = 0.0

        def at(t: float) -> int:
            return round(t * rate)

        def capped() -> None:
            nonlocal full
            if not full:
                full = True
                logger.warning(
                    "max_buffered_speech reached, ignoring further data for the current speech"
                )

        def as_frame(data: np.ndarray) -> rtc.AudioFrame:
            return rtc.AudioFrame(
                data=data.tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=len(data)
            )

        async for f in self._input_ch:
            if isinstance(f, self._FlushSentinel):
                if rate:
                    reset()
                continue
            if not isinstance(f, rtc.AudioFrame):
                continue
            if not rate:
                rate = f.sample_rate
                reset()
            elif f.sample_rate != rate:
                logger.error("a frame with another sample rate was already pushed")
                continue
            assert processor is not None
            pcm = np.frombuffer(f.data, dtype=np.int16)
            if f.num_channels > 1:
                pcm = alkove.to_int16(
                    pcm.reshape(-1, f.num_channels).astype(np.float64).mean(axis=1) / 32768
                )
            recent.push(pcm)
            if onset is not None:
                speech.append(pcm[: max(0, room)])
                room -= len(pcm)
                if room < 0:
                    capped()
            t0 = time.perf_counter()
            part = processor.push(pcm)
            rows = part.trace["cue"] if part.trace else []
            spent = (time.perf_counter() - t0) / max(1, len(rows))
            for row in rows:
                k = round(row["t"] / WINDOW)
                p = row["speech"]
                now = (k + 1) * WINDOW
                speaking = tracker.confirmed
                if speaking:
                    speech_pub += WINDOW
                else:
                    silence_pub += WINDOW
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=(k + 1) * 480,
                        timestamp=now,
                        speech_duration=speech_pub,
                        silence_duration=silence_pub,
                        frames=[as_frame(recent.slice(at(k * WINDOW), at(now)))],
                        probability=p,
                        inference_duration=spent,
                        speaking=speaking,
                        raw_accumulated_silence=silence_acc,
                        raw_accumulated_speech=speech_acc,
                    )
                )
                if p >= v.speech.confidence or (tracker.active and p >= v.speech.exit_confidence):
                    speech_acc += WINDOW
                    silence_acc = 0.0
                else:
                    silence_acc += WINDOW
                    speech_acc = 0.0
                e = tracker.push(p, k)
                if e is not None and e["type"] == "speechstart":
                    onset = e["start"]
                    speech_pub, silence_pub = now - onset, 0.0
                    first = max(recent.base, at(onset - v.prefix_padding_duration))
                    head = recent.slice(first, recent.end)
                    cap = at(v.max_buffered_speech + v.prefix_padding_duration)
                    speech, room, full = [head[:cap]], cap - len(head), False
                    if room < 0:
                        capped()
                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.START_OF_SPEECH,
                            samples_index=(k + 1) * 480,
                            timestamp=now,
                            speech_duration=speech_pub,
                            silence_duration=0.0,
                            frames=[as_frame(recent.slice(first, at(now)))],
                            speaking=True,
                        )
                    )
                elif e is not None:
                    speech_pub, silence_pub = 0.0, now - e["end"]
                    whole = np.concatenate(speech) if speech else np.zeros(0, dtype=np.int16)
                    end = at(now) - first
                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.END_OF_SPEECH,
                            samples_index=(k + 1) * 480,
                            timestamp=now,
                            speech_duration=e["end"] - e["start"],
                            silence_duration=silence_pub,
                            frames=[as_frame(whole[:end])],
                            speaking=False,
                        )
                    )
                    speech, onset = [], None
            recent.keep(at(v.prefix_padding_duration + v.speech.start_secs + 1))
