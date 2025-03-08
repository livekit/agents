import asyncio
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from kokoro_onnx import Kokoro
from livekit.agents import tts, utils
from livekit.rtc.audio_frame import AudioFrame

from .log import logger

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.2
DEFAULT_LANGUAGE = "en-us"
TTS_NUM_CHANNELS = 1
SAMPLE_RATE = 24000


@dataclass
class _TTSOptions:
    voice: str | None = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED
    language: str = DEFAULT_LANGUAGE
    sample_rate: int = SAMPLE_RATE


class TTS(tts.TTS):
    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    MODEL_FOLDER = "kokoro_models"

    def __init__(
        self,
        *,
        voice: str | None = DEFAULT_VOICE,
        speed: str = DEFAULT_SPEED,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=TTS_NUM_CHANNELS,
        )
        self._loadkokoro()

        self._opts = _TTSOptions(voice, speed, language)

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            text=text,
            opts=self._opts,
            model=self._model,
        )

    def _loadkokoro(self):
        """Loads the Kokoro model. Note that if this is the first time
        this model is being run, it will take time to download.
        """
        logger.debug("Loading Kokoro model...")
        model_path, voices_path = self._download_model_files()
        self._model = Kokoro(model_path, voices_path)
        logger.debug("Loaded Kokoro model")

    @classmethod
    def loadkokoro(cls, **kwargs):
        return cls(**kwargs)

    def _download_model_files(self):
        os.makedirs(self.MODEL_FOLDER, exist_ok=True)

        model_path = os.path.join(self.MODEL_FOLDER, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(self.MODEL_FOLDER, "voices-v1.0.bin")

        if not os.path.exists(model_path):
            self._download_file(self.MODEL_URL, model_path)

        if not os.path.exists(voices_path):
            self._download_file(self.VOICES_URL, voices_path)

        return model_path, voices_path

    @staticmethod
    def _download_file(url, destination):
        logger.info(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Downloaded {destination}")


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        opts: _TTSOptions,
        model,
    ) -> None:
        super().__init__(tts=tts, input_text=text)
        self._opts = opts
        self._text = text
        self._segment_id = utils.shortuuid()
        self.model = model

    async def _run(self):
        request_id = utils.shortuuid()

        # try:
        stream = self.model.create_stream(
            self._text,
            self._opts.voice,
            self._opts.speed,
            self._opts.language,
        )

        async def push_data():
            try:
                async for audio_data, sample_rate in stream:
                    logger.debug(
                        f"Received audio chunk: shape={audio_data.shape}, dtype={audio_data.dtype}, sample_rate={sample_rate}"
                    )

                    # Convert to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)

                    # Create an AudioFrame
                    frame = AudioFrame(
                        data=audio_int16.tobytes(),
                        sample_rate=self._opts.sample_rate,
                        num_channels=TTS_NUM_CHANNELS,
                        samples_per_channel=len(audio_int16),
                    )

                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=self._segment_id,
                            frame=frame,
                        )
                    )
            except Exception as e:
                logger.error(f"Kokoro TTS error: {e}")
            finally:
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        segment_id=self._segment_id,
                        frame=None,
                    )
                )

        push_task = asyncio.create_task(push_data())
        try:
            await push_task
        finally:
            await utils.aio.gracefully_cancel(push_task)
