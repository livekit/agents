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

import pkg_resources
import asyncio
from dataclasses import dataclass
from typing import AsyncIterable
from enum import Enum
from livekit import rtc
import torch
import numpy as np

VAD_SAMPLE_RATE = 16000

VADEventType = Enum("VADEventType", "STARTED, FINISHED")


class VADPlugin:
    """Class for Voice Activity Detection (VAD)"""

    @dataclass
    class Event:
        type: VADEventType
        frames: [rtc.AudioFrame]

    def __init__(self, *, left_padding_ms: int, silence_threshold_ms: int):
        self._silence_threshold_ms = silence_threshold_ms
        self._left_padding_ms = left_padding_ms
        self._window_buffer = np.zeros(512, dtype=np.float32)
        self._window_buffer_scratch = np.zeros(512, dtype=np.float32)
        self._model = None
        self._copy_int16 = np.zeros(1, dtype=np.int16)
        self._copy_float32 = np.zeros(1, dtype=np.float32)
        self._silent_frame_count_since_voice = 0
        self._voice_frames = []
        self._left_padding_frames = []
        self._frame_queue = []
        self._talking_state = False

    async def close(self):
        pass

    async def start(
        self, frame_iterator: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[Event]:
        """Start detecting voice activity

        Args:
            frame_iterator (AsyncIterable[rtc.AudioFrame]): Stream of audio frames to check for voice

        Returns:
            AsyncIterable[Event]: Stream of voice events
        """
        async for frame in frame_iterator:
            event = await self._push_frame(frame)
            if event is not None:
                yield event

    async def _push_frame(self, frame: rtc.AudioFrame):
        if self._model is None:
            await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        self._frame_queue.append(frame)

        # run inference every 30ms
        if len(self._frame_queue) * 10 < 30:
            return

        frame_queue_copy = self._frame_queue.copy()
        self._frame_queue = []

        talking_detected = await asyncio.get_event_loop().run_in_executor(
            None, self._process_frames, frame_queue_copy
        )
        if self._talking_state:
            if talking_detected:
                self._silent_frame_count_since_voice = 0
                self._voice_frames.extend(frame_queue_copy)
            else:
                self._silent_frame_count_since_voice += len(frame_queue_copy)
                if (
                    self._silent_frame_count_since_voice * 10
                    > self._silence_threshold_ms
                ):
                    self._talking_state = False
                    result = []
                    result.extend(self._left_padding_frames)
                    result.extend(self._voice_frames)
                    self._reset_frames()
                    event = VADPlugin.Event(type=VADEventType.FINISHED, frames=result)
                    return event
        else:
            if talking_detected:
                self._talking_state = True
                self._voice_frames.extend(frame_queue_copy)
                event = VADPlugin.Event(
                    type=VADEventType.STARTED, frames=self._voice_frames
                )
                return event
            else:
                self._add_left_padding(frame_queue_copy)

        return None

    def _add_left_padding(self, frames: [rtc.AudioFrame]):
        current_padding_ms = 0
        for f in self._left_padding_frames:
            current_padding_ms += f.sample_rate * len(f.data) * 1000 / f.num_channels
        ms_length = 0
        for f in frames:
            ms_length += f.sample_rate * len(f.data) * 1000 / f.num_channels
            if ms_length + current_padding_ms > self._left_padding_ms:
                return
            self._left_padding_frames.append(f)

    def _reset_frames(self):
        self._left_padding_frames = []
        self._voice_frames = []

    def _process_frames(self, frames: [rtc.AudioFrame]) -> bool:
        resampled = [f.remix_and_resample(VAD_SAMPLE_RATE, 1) for f in frames]
        buffer_count = 0
        for f in resampled:
            buffer_count += len(f.data)

        assert (
            buffer_count <= 512
        ), "Buffer count should be less than the input to the VAD model"

        if self._copy_int16.shape[0] < buffer_count:
            self._copy_int16 = np.zeros(buffer_count, dtype=np.int16)
            self._copy_float32 = np.zeros(buffer_count, dtype=np.float32)

        for i, f in enumerate(resampled):
            self._copy_int16[
                i * len(f.data) : (i + 1) * len(f.data)
            ] = np.ctypeslib.as_array(f.data)
            self._copy_float32 = (
                self._copy_int16.astype(np.float32, copy=False) / 32768.0
            )

        self._window_buffer_scratch[:-buffer_count] = self._window_buffer[buffer_count:]
        self._window_buffer[-buffer_count:] = self._copy_float32
        self._window_buffer[:-buffer_count] = self._window_buffer_scratch[
            :-buffer_count
        ]
        tensor = torch.from_numpy(self._window_buffer)
        speech_prob = self._model(tensor, VAD_SAMPLE_RATE).item()
        return speech_prob > 0.5

    def _load_model(self):
        model_path = pkg_resources.resource_filename(
            "livekit.plugins.silero", "files/silero_vad.jit"
        )
        self._model = torch.jit.load(model_path)
