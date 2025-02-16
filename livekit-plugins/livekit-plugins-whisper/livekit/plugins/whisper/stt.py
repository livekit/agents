from .energy import AudioEnergyFilter
from .models import Model
from .utils.audio import exp_smoothing, calculate_audio_volume
from .log import logger


import asyncio
from dataclasses import dataclass
import io
import wave

import numpy as np

from livekit.agents import (
    stt,
    utils,
    APIConnectOptions,
    APIConnectionError,
    APIStatusError,
)
from livekit.agents.utils import AudioBuffer
from livekit.rtc.audio_frame import AudioFrame
from livekit import rtc

utils.time_ms

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Whisper, you need to `pip install pipecat-ai[whisper]`."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class WhisperSTTOptions:
    model: Model = Model.LARGE_V3_TURBO
    device: str = "auto"
    compute_type: str = "default"
    no_speech_prob: float = 0.4
    language_prob: float = 0.9
    energy_filter: AudioEnergyFilter | bool = False
    min_volume: float = 0.6
    max_silence_secs: float = 0.3
    max_buffer_secs: float = 1.5
    sample_rate: int = 24000
    num_channels: int = 1


class WhisperSTT(stt.STT):
    def __init__(
        self,
        *,
        opts: WhisperSTTOptions | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._opts = opts or WhisperSTTOptions()
        self._load()

        (self._content, self._wave) = self._new_wave()
        self._silence_secs = 0
        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download.
        """
        logger.debug("Loading Whisper model...")
        self._model = WhisperModel(
            self._opts.model.value,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
        )
        logger.debug("Loaded Whisper model")

    @classmethod
    def load(cls, opts: WhisperSTTOptions | None = None) -> "WhisperSTT":
        """
        Load and initialize the Whisper model.

        This method loads the Whisper model and prepares it for inference. When options are not provided,
        sane defaults are used.

        **Note:**
            This method is blocking and may take time to load the model into memory.
            It is recommended to call this method inside your prewarm mechanism.
        """
        return cls(opts=opts)

    def _new_wave(self):
        content = io.BytesIO()
        ww = wave.open(content, "wb")
        ww.setsampwidth(2)
        ww.setnchannels(self._opts.num_channels)
        ww.setframerate(self._opts.sample_rate)
        return (content, ww)

    def _get_smoothed_volume(self, chunk: bytes) -> float:
        volume = calculate_audio_volume(chunk, self._opts.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    async def _recognize_impl(
        self,
        frame: AudioFrame,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        # Try to filter out empty background noise

        # Process the frame in 20ms chunks
        chunk_duration = 20 / 1000  # 20ms in seconds
        chunk_size = int(
            chunk_duration * frame.sample_rate * frame.num_channels * 2
        )  # 2 bytes per sample
        volume = 0

        for i in range(0, len(frame.data), chunk_size):
            chunk = frame.data[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break  # Ignore incomplete chunk

            volume = self._get_smoothed_volume(chunk)

            if volume >= self._opts.min_volume:
                self._wave.writeframes(chunk)
                self._silence_secs = 0
            else:
                self._silence_secs += 0.02

            self._prev_volume = volume

        if volume >= self._opts.min_volume:
            # If volume is high enough, write new data to wave file
            self._wave.writeframes(frame.data)
            # self._wave = frame.to_wav_bytes()
            self._silence_secs = 0
        else:
            self._silence_secs += frame.duration
        self._prev_volume = volume

        # If buffer is not empty and we have enough data or there's been a long
        # silence, transcribe the audio gathered so far.
        buffer_secs = self._wave.getnframes() / self._opts.sample_rate

        response = None
        if self._content.tell() > 0 and (
            buffer_secs > self._opts.max_buffer_secs
            or self._silence_secs > self._opts.max_silence_secs
        ):
            self._silence_num_frames = 0
            self._wave.close()
            self._content.seek(0)
            # await self.process_generator(self.run_stt(self._content.read()))
            response = await self.run_stt(self._content.read())
            (self._content, self._wave) = self._new_wave()

        return response or stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text="",
                    language="",
                    confidence=0.0,
                )
            ],
        )

    async def run_stt(
        self,
        buffer: AudioBuffer,
    ) -> stt.SpeechEvent:
        """Transcribes given audio using Whisper"""
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            return APIConnectionError("Whisper model not available", status_code=500)

        try:
            # Divide by 32768 because we have signed 16-bit data.
            audio_float = (
                np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            )

            segments, info = await asyncio.to_thread(
                self._model.transcribe, audio_float
            )

            lang, lang_prob = None, None
            if info.language_probability > self._opts.language_prob:
                lang, lang_prob = info.language, info.language_probability

            text: str = ""
            for segment in segments:
                if segment.no_speech_prob < self._opts.no_speech_prob:
                    text += f"{segment.text} "

            logger.debug(f"Transcription: [{text}]")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text or "",
                        language=lang or "",
                        confidence=lang_prob or 0.0,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            return APIStatusError("Whisper STT error", status_code=500, body=str(e))
