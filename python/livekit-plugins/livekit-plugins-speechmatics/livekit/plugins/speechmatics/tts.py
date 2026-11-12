from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tts,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .utils import get_tts_url

BASE_URL = "https://preview.tts.speechmatics.com"
NUM_CHANNELS = 1
DEFAULT_VOICE = "sarah"


@dataclass
class _TTSOptions:
    voice: str
    sample_rate: int
    base_url: str
    api_key: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = 16000,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Create a new instance of Speechmatics TTS.

        Args:
            voice (str): Voice model to use for synthesis. Options: "sarah", "theo", "megan". Defaults to "sarah".
            sample_rate (int): Sample rate of audio. Defaults to 16000.
            api_key (str): Speechmatics API key. If not provided, will look for SPEECHMATICS_API_KEY in environment.
            base_url (str): Base URL for Speechmatics TTS API. Defaults to "https://preview.tts.speechmatics.com"
            http_session (http.HttpSession): Optional HTTP session to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
        if not api_key:
            raise ValueError(
                "Speechmatics API key required. Set SPEECHMATICS_API_KEY or provide api_key."
            )

        self._opts = _TTSOptions(
            voice=voice,
            sample_rate=sample_rate,
            base_url=base_url,
            api_key=api_key,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Speechmatics"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice model to use for synthesis.
        """
        if is_given(voice):
            self._opts.voice = voice

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            options=self._opts,
            http_session=self._ensure_session,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        options: _TTSOptions,
        http_session: Callable[[], http.HttpSession],
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._get_http_session = http_session
        self._opts = replace(options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": self._input_text,
            }

            url = get_tts_url(self._opts.base_url, self._opts.voice, self._opts.sample_rate)

            async with self._get_http_session().post(
                url, json=payload, headers=headers, conn_options=self._conn_options
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                # Process the response in streaming chunks
                buffer = b""

                async for data in resp.iter_bytes():
                    if not data:
                        continue

                    buffer += data

                    # Emit all complete 2-byte int16 samples from buffer
                    if len(buffer) >= 2:
                        complete_samples = len(buffer) // 2
                        complete_bytes = complete_samples * 2

                        audio_data = buffer[:complete_bytes]
                        buffer = buffer[complete_bytes:]  # Keep remaining bytes for next iteration

                        output_emitter.push(audio_data)

                if buffer:
                    raise APIError(
                        "Speechmatics returned an incomplete PCM sample", retryable=False
                    )
                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
