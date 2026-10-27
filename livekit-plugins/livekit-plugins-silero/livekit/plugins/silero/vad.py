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

from typing import Literal

from livekit import agents
from livekit.agents import inference
from livekit.agents.types import NOT_GIVEN, NotGivenOr

VADStream = agents.VADStream


class VAD(inference.VAD):
    """Silero voice activity detection: the core's local Silero model and detection machine
    (``livekit.agents.inference.VAD``), with this plugin's defaults."""

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[8000, 16000] = 16000,
        force_cpu: bool = True,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> VAD:
        """The Silero VAD. The model is the core's, loaded once per process (where the job
        processes share it, before they fork), so this is cheap to call per session.

        Args:
            min_speech_duration: Minimum duration of speech to start a new speech chunk.
            min_silence_duration: At the end of each speech, wait this duration before ending it.
            prefix_padding_duration: Duration of padding to add to the beginning of each speech
                chunk.
            max_buffered_speech: Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold: Threshold to consider a frame as speech.
            sample_rate: Kept for compatibility: the model runs at 16 kHz, on any input rate.
            force_cpu: Kept for compatibility: the model runs on the CPU.
            deactivation_threshold: The probability below which speech ends;
                ``max(activation_threshold - 0.15, 0.01)`` by default.

        Raises:
            ValueError: If an unsupported sample rate is provided.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")
        return cls(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
        )
