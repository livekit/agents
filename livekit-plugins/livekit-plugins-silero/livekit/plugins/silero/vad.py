# Copyright 2023 LiveKit, Inc.
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
import contextlib
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch
from livekit import agents, rtc

from .log import logger


class VAD(agents.vad.VAD):
    def __init__(self, *, model_path: str | None = None, use_onnx: bool = True) -> None:
        if model_path:
            model = torch.jit.load(model_path)
            model.eval()
        else:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:v4.0",
                model="silero_vad",
                onnx=use_onnx,
            )
        self._model = model

    def stream(
        self,
        *,
        min_speaking_duration: float = 0.2,
        min_silence_duration: float = 0.8,
        padding_duration: float = 0.1,
        sample_rate: int = 16000,
        max_buffered_speech: float = 45.0,
        threshold: float = 0.2,
    ) -> "VADStream":
        return VADStream(
            self._model,
            min_speaking_duration=min_speaking_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            sample_rate=sample_rate,
            max_buffered_speech=max_buffered_speech,
            threshold=threshold,
        )


# Based on https://github.com/snakers4/silero-vad/blob/94504ece54c8caeebb808410b08ae55ee82dba82/utils_vad.py#L428
class VADStream(agents.vad.VADStream):
    def __init__(
        self,
        model,
        *,
        min_speaking_duration: float,
        min_silence_duration: float,
        padding_duration: float,
        sample_rate: int,
        max_buffered_speech: float,
        threshold: float,
    ) -> None:
        self._min_speaking_duration = min_speaking_duration
        self._min_silence_duration = min_silence_duration
        self._padding_duration = padding_duration
        self._sample_rate = sample_rate
        self._max_buffered_speech = max_buffered_speech
        self._threshold = threshold

        if sample_rate not in [8000, 16000]:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        self._queue = asyncio.Queue[Optional[rtc.AudioFrame]]()
        self._event_queue = asyncio.Queue[Optional[agents.vad.VADEvent]]()
        self._model = model

        self._closed = False
        self._speaking = False
        self._waiting_start = False
        self._waiting_end = False
        self._current_sample = 0
        self._filter = agents.utils.ExpFilter(0.8)
        self._min_speaking_samples = min_speaking_duration * sample_rate
        self._min_silence_samples = min_silence_duration * sample_rate
        self._padding_duration_samples = padding_duration * sample_rate
        self._max_buffered_samples = max_buffered_speech * sample_rate

        self._queued_frames: deque[rtc.AudioFrame] = deque()
        self._original_frames: deque[rtc.AudioFrame] = deque()
        self._buffered_frames: List[rtc.AudioFrame] = []
        self._main_task = asyncio.create_task(self._run())

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        if not wait:
            self._main_task.cancel()

        self._queue.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self):
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break  # None is sent inside aclose

                self._queue.task_done()

                # resample to silero's sample rate
                resampled_frame = frame.remix_and_resample(
                    self._sample_rate, 1
                )  # TODO: This is technically wrong, fix when we have a better resampler
                self._original_frames.append(frame)
                self._queued_frames.append(resampled_frame)

                # run inference by chunks of 40ms until we run out of data
                while True:
                    available_length = sum(
                        f.samples_per_channel for f in self._queued_frames
                    )

                    samples_40ms = self._sample_rate // 1000 * 40
                    if available_length < samples_40ms:
                        break

                    await asyncio.shield(self._run_inference())

        except Exception:
            logger.exception("silero stream failed")
        finally:
            self._event_queue.put_nowait(None)

    async def _run_inference(self) -> None:
        # merge the first 4 frames (we know each is 10ms)
        if len(self._queued_frames) < 4:
            return

        original_frames = [self._original_frames.popleft() for _ in range(4)]
        merged_frame = agents.utils.merge_frames(
            [self._queued_frames.popleft() for _ in range(4)]
        )

        # convert data_40ms to tensor & f32
        tensor = torch.from_numpy(np.frombuffer(merged_frame.data, dtype=np.int16))
        tensor = tensor.to(torch.float32) / 32768.0

        # run inference
        start_time = time.time()
        raw_prob = await asyncio.to_thread(
            lambda: self._model(tensor, self._sample_rate).item()
        )
        probability = self._filter.apply(1.0, raw_prob)
        inference_duration = time.time() - start_time

        # inference done
        event = agents.vad.VADEvent(
            type=agents.vad.VADEventType.INFERENCE_DONE,
            samples_index=self._current_sample,
            probability=probability,
            raw_inference_prob=raw_prob,
            inference_duration=inference_duration,
        )
        self._event_queue.put_nowait(event)

        self._dispatch_event(original_frames, probability, raw_prob, inference_duration)
        self._current_sample += merged_frame.samples_per_channel

    def _dispatch_event(
        self,
        original_frames: List[rtc.AudioFrame],
        probability: float,
        raw_inference_prob: float,
        inference_duration: float,
    ):
        """
        Dispatches a VAD event based on the speech probability and the options
        Args:
            speech_prob: speech probability of the current frame
            original_frames: original frames of the current inference
        """

        samples_10ms = self._sample_rate / 100
        padding_count = int(
            self._padding_duration_samples // samples_10ms
        )  # number of frames to keep for the padding (one side)

        self._buffered_frames.extend(original_frames)
        if (
            not self._speaking
            and not self._waiting_start
            and len(self._buffered_frames) > padding_count
        ):
            self._buffered_frames = self._buffered_frames[
                len(self._buffered_frames) - padding_count :
            ]

        max_buffer_len = padding_count + max(
            int(self._max_buffered_samples // samples_10ms),
            int(self._min_speaking_samples // samples_10ms),
        )
        if len(self._buffered_frames) > max_buffer_len:
            self._buffered_frames = self._buffered_frames[
                len(self._buffered_frames) - max_buffer_len :
            ]

        if probability >= self._threshold:
            # speaking, wait for min_speaking_duration to trigger START_OF_SPEECH
            self._waiting_end = False
            if not self._waiting_start and not self._speaking:
                self._waiting_start = True
                self._start_speech = self._current_sample

            if self._waiting_start and (
                self._current_sample - self._start_speech >= self._min_speaking_samples
            ):
                self._waiting_start = False
                self._speaking = True

                # since we're waiting for the min_spaking_duration to trigger START_OF_SPEECH,
                # put the speech that were used to trigger the start here
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.START_OF_SPEECH,
                    samples_index=self._start_speech,
                    frames=self._buffered_frames[padding_count:],
                    speaking=True,
                )
                self._event_queue.put_nowait(event)

        # we don't check the speech_prob here
        event = agents.vad.VADEvent(
            type=agents.vad.VADEventType.INFERENCE_DONE,
            samples_index=self._current_sample,
            frames=original_frames,
            probability=probability,
            raw_inference_prob=raw_inference_prob,
            inference_duration=inference_duration,
            speaking=self._speaking,
        )
        self._event_queue.put_nowait(event)

        if probability < self._threshold:
            # stopped speaking, s for min_silence_duration to trigger END_OF_SPEECH,
            self._waiting_start = False
            if not self._waiting_end and self._speaking:
                self._waiting_end = True
                self._end_speech = self._current_sample

            if self._waiting_end and (
                self._current_sample - self._end_speech
                >= max(self._min_silence_samples, self._padding_duration_samples)
            ):
                self._waiting_end = False
                self._speaking = False
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.END_OF_SPEECH,
                    samples_index=self._end_speech,
                    duration=(self._end_speech - self._start_speech)
                    / self._sample_rate,
                    frames=self._buffered_frames,
                    speaking=False,
                )
                self._event_queue.put_nowait(event)

    async def __anext__(self) -> agents.vad.VADEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
