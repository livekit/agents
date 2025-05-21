# Copyright 2024 LiveKit, Inc.
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
import numpy as np
import whisper # type: ignore

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer
from livekit.agents.types import SpeechData, SpeechEvent, SpeechEventType

import logging # Use livekit.agents.log instead? For now, stdlib logging
logger = logging.getLogger(__name__)

# Default model if not specified
DEFAULT_MODEL = "base"
# Whisper operates on 16kHz audio internally, but can resample.
# We should aim to provide it with 16kHz if possible, or let it handle resampling.
# The frames from LiveKit are typically S16LE. Whisper expects float32.
WHISPER_SAMPLE_RATE = 16000

class STT(stt.STT):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        language: str | None = None, # Whisper can auto-detect if None
        model_path: str | None = None, # For custom model location
        device: str | None = None, # e.g., "cuda", "cpu"
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,  # Whisper processes audio in chunks but not a continuous stream for interim results
                interim_results=False,
            )
        )
        logger.info(f"Initializing Whisper STT with model: {model_name}, language: {language or 'auto-detect'}")
        try:
            self._model = whisper.load_model(model_name, download_root=model_path, device=device)
            logger.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}")
            # Propagate error or handle gracefully, e.g., by disabling the STT
            raise RuntimeError(f"Failed to load Whisper model '{model_name}'") from e

        self._language = language
        self._sample_rate = WHISPER_SAMPLE_RATE # Target sample rate for Whisper
        self._num_channels = 1 # Whisper expects mono audio

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
    ) -> SpeechEvent:
        logger.debug(f"Recognizing audio buffer of {len(buffer.frames)} frames.")

        # Combine audio frames from the buffer.
        # The buffer contains rtc.AudioFrame objects.
        # Each frame has data, sample_rate, num_channels, samples_per_channel.
        # We need to convert this to a single float32 NumPy array at Whisper's expected sample rate.

        if not buffer.frames:
            logger.warning("Received empty audio buffer for recognition.")
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[SpeechData(text="", language="")])

        # Assuming input frames are S16LE based on typical WebRTC/LiveKit usage.
        # Convert to a single byte array first.
        
        # Determine the original sample rate and channels from the first frame
        # Assuming all frames in the buffer have the same characteristics
        original_sample_rate = buffer.frames[0].sample_rate
        original_num_channels = buffer.frames[0].num_channels

        # Concatenate all frame data
        raw_audio_data = b"".join(frame.data for frame in buffer.frames)

        # Convert to NumPy array (int16)
        audio_s16 = np.frombuffer(raw_audio_data, dtype=np.int16)

        # Convert to float32 and normalize to [-1.0, 1.0]
        audio_float32 = audio_s16.astype(np.float32) / 32768.0
        
        # Resample if necessary and convert to mono
        # Whisper's load_audio function handles resampling and mono conversion if given a file path.
        # For direct array input, we need to ensure it's in the correct format.
        # The `transcribe` method can take a numpy array directly.
        # According to Whisper docs, the array should be a float32 mono waveform.

        if original_num_channels > 1:
            # Simple average for stereo to mono if not handled by a library function
            # This might not be ideal, but Whisper expects mono.
            # For more robust conversion, a library like librosa would be better.
            # However, trying to keep dependencies minimal if whisper itself can handle it.
            # The `transcribe` function itself seems to expect a mono float32 array.
            if audio_float32.ndim > 1 and audio_float32.shape[1] == original_num_channels:
                 audio_float32 = audio_float32.mean(axis=1) # Average channels
            elif audio_float32.ndim == 1 and original_num_channels > 1: # interleaved stereo
                audio_float32 = audio_float32.reshape(-1, original_num_channels).mean(axis=1)


        # Resampling: Whisper's processor will handle resampling if the sample rate is different.
        # No explicit resampling code needed here if `model.transcribe` handles it for numpy arrays.
        # The `log_mel_spectrogram` function in whisper.audio resamples.
        # `transcribe` calls `log_mel_spectrogram`.

        logger.debug(f"Audio data prepared: {audio_float32.shape}, dtype: {audio_float32.dtype}")

        try:
            # The `transcribe` method can accept a numpy array.
            # It expects a mono float32 array. Sample rate differences are handled internally.
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_float32,
                language=self._language, # Can be None for auto-detection
                fp16=False # Set to True if using CUDA and want faster/less memory, False for CPU
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            # Consider how to report this error, e.g., an empty transcript or raise
            # For now, return an empty transcript as STT might be part of a larger flow.
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[SpeechData(text="", language=self._language or "")])


        transcribed_text = result.get("text", "").strip()
        detected_language = result.get("language", self._language or "")

        logger.info(f"Transcription result: \"{transcribed_text}\" (Language: {detected_language})")

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    text=transcribed_text,
                    language=detected_language,
                )
            ],
        )

    # The base STT class handles the stream method by accumulating audio
    # and calling _recognize_impl when streaming=False. So, no need to override stream().

    async def close(self):
        # No explicit close needed for Whisper model unless managing GPU resources specifically.
        # Models are typically loaded in memory and Python's GC handles them.
        logger.info("Closing Whisper STT (no specific cleanup actions).")
        pass
