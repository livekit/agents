from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import torch
from transformers.pipelines import pipeline

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.utils import is_given

from .log import logger

# Define AudioBuffer type for type hints
AudioBuffer = utils.AudioBuffer

# Optional accelerate import
_accelerate_available = False
try:
    import accelerate
    _accelerate_available = True
except ImportError:
    logger.info("accelerate package not found, using default device mapping")

SAMPLE_RATE = 16_000  # Hz (match Whisper)
CHUNK_LENGTH = 10


class WhisperSTT(stt.STT):
    def __init__(
        self,
        *,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        chunk_length_s: int = CHUNK_LENGTH,
    ):
        """
        Create a new instance of Whisper STT.
        
        Args:
            model_name: The name of the Whisper model to use
            device: The device to use for inference (None for auto-detection)
            chunk_length_s: The length of audio chunks to process at once
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                device_map = "auto" if _accelerate_available else None
            else:
                device = "cpu"
                device_map = None
        else:
            device_map = None
            
        logger.info(f"Loading Whisper model '{model_name}'...")
        if device_map:
            self.transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device_map=device_map,
                chunk_length_s=chunk_length_s,
            )
        else:
            self.transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device=0 if device == "cuda:0" else -1,
                chunk_length_s=chunk_length_s,
            )
        logger.info("Whisper model loaded.")
        
        self.sample_rate = SAMPLE_RATE
        
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement the recognition method required by the STT interface"""
        
        # Handle different types of audio buffers
        if isinstance(buffer, list):
            # Concatenate audio frames
            audio_bytes = b''.join([frame.data.tobytes() for frame in buffer])
            sample_rate = buffer[0].sample_rate if buffer else self.sample_rate
        elif hasattr(buffer, 'data') and hasattr(buffer, 'sample_rate'):
            # Single audio frame
            audio_bytes = buffer.data.tobytes()
            sample_rate = buffer.sample_rate
        elif hasattr(buffer, 'to_numpy'):
            # AudioBuffer object
            audio_data = buffer.to_numpy()
            sample_rate = buffer.sample_rate
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Simple resampling - in production you'd want a better resampler
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * self.sample_rate / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Run transcription in a thread pool to avoid blocking
            result = await asyncio.to_thread(self.transcriber, audio_data)
            
            # Skip to result handling
            goto_result_handling = True
        else:
            # Unknown buffer type
            raise ValueError(f"Unsupported buffer type: {type(buffer)}")
            
        # Process audio bytes if we didn't already process a numpy array
        if not locals().get('goto_result_handling', False):
            # Convert to int16 numpy array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_int16.astype(np.float32) / 32768.0
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Simple resampling - in production you'd want a better resampler
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * self.sample_rate / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Run transcription in a thread pool to avoid blocking
            result = await asyncio.to_thread(self.transcriber, audio_data)
        
        # Extract text from result - handle different return formats
        if isinstance(result, dict):
            text = result.get('text', '').strip()
        elif hasattr(result, 'text'):
            text = result.text.strip()
        elif isinstance(result, str):
            text = result.strip()
        else:
            # Try to convert to string
            try:
                text = str(result).strip()
            except:
                text = ""
        
        # Create and return a SpeechEvent
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(
                language=language if is_given(language) else "en",
                text=text,
                confidence=1.0
            )]
        )