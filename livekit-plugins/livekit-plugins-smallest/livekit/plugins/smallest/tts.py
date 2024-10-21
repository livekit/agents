from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Any

import aiohttp
from livekit.agents import tts, utils

from .log import logger
from .models import TTSLanguages, TTSModels, TTSVoices, TTSEncoding

API_BASE_URL = "https://waves-api.smallest.ai/api/v1"
NUM_CHANNELS = 1

@dataclass
class _TTSOptions:
    model: TTSModels
    encoding: TTSEncoding
    sample_rate: int
    voice: TTSVoices
    api_key: str
    language: TTSLanguages
    add_wav_header: bool


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels = "lightning",
        language: TTSLanguages = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: TTSVoices = "emily",
        sample_rate: int = 24000,
        api_key: str | None = None,
        add_wav_header: bool = True,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of smallest.ai Waves TTS.
        Args:
            model (TTSModels, optional): The Waves TTS model to use. Defaults to "lightning".
            language (TTSLanguages, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (VoiceSettings, optional): The voice settings to use. Defaults to "emily".
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            api_key (str, optional): The smallest.ai API key. If not provided, it will be read from the SMALLEST_API_KEY environment variable.
            add_wav_header (bool, optional): If True, includes a WAV header in the audio output; otherwise, only raw audio data is returned.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError("SMALLEST_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=api_key,
            add_wav_header=add_wav_header
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the Waves API endpoint"""

    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        super().__init__()
        self._text, self._opts, self._session = text, opts, session
        self._initial_buffer = bytearray()
        self._final_buffer = bytearray()
        self._fade_in_samples = int(0.01 * opts.sample_rate)  # 10ms fade-in
        self._fade_out_samples = int(0.01 * opts.sample_rate)  # 10ms fade-out

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )
        request_id, segment_id = utils.shortuuid(), utils.shortuuid()

        data = _to_smallest_options(self._opts)
        data["text"] = self._text

        url = f"{API_BASE_URL}/{self._opts.model}/get_speech"
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

        async with self._session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"smallest.ai API error: {resp.status} - {error_text}")

            async for chunk, _ in resp.content.iter_chunks():
                self._initial_buffer.extend(chunk)
                if len(self._initial_buffer) >= self._fade_in_samples * 2:  # 16-bit samples
                    break

            # Apply fade-in to the initial buffer
            samples = np.frombuffer(self._initial_buffer, dtype=np.int16)
            fade_in = np.linspace(0, 1, self._fade_in_samples)
            samples[:self._fade_in_samples] = (samples[:self._fade_in_samples] * fade_in).astype(np.int16)

            # Send the faded-in initial buffer
            for frame in bstream.write(samples.tobytes()):
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )

            # process and send the rest of the audio while collecting the last buffer for fade-out
            last_chunk = None
            async for chunk, _ in resp.content.iter_chunks():
                if last_chunk is not None:
                    for frame in bstream.write(last_chunk):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id, segment_id=segment_id, frame=frame
                            )
                        )
                last_chunk = chunk  # keeping track of the last chunk for fade-out

            # apply fade-out to the last chunk
            if last_chunk:
                self._final_buffer.extend(last_chunk)
                samples = np.frombuffer(self._final_buffer, dtype=np.int16)
                fade_out = np.linspace(1, 0, self._fade_out_samples)
                samples[-self._fade_out_samples:] = (samples[-self._fade_out_samples:] * fade_out).astype(np.int16)

                # Send the faded-out final buffer
                for frame in bstream.write(samples.tobytes()):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id, segment_id=segment_id, frame=frame
                        )
                    )

            # flush any remaining audio
            for frame in bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )


def _to_smallest_options(opts: _TTSOptions) -> dict[str, Any]:
    return {
        "voice_id": opts.voice,
        "sample_rate": opts.sample_rate,
        "add_wav_header": opts.add_wav_header,
    }
