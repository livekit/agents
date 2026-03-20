from __future__ import annotations

import base64
import os
from dataclasses import dataclass, replace
from typing import Literal

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from .models import TTSModels, TTSVoices

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_MODEL: TTSModels = "voxtral-mini-tts-2603"
DEFAULT_VOICE: TTSVoices = "en_paul_neutral"

RESPONSE_FORMATS = Literal["mp3", "wav", "pcm", "opus", "flac"]
DEFAULT_RESPONSE_FORMAT: RESPONSE_FORMATS = "mp3"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    response_format: RESPONSE_FORMATS


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: Mistral | None = None,
        response_format: RESPONSE_FORMATS = DEFAULT_RESPONSE_FORMAT,
    ) -> None:
        """
        Create a new instance of MistralAI TTS.

        Args:
            model: The MistralAI TTS model to use, default is voxtral-mini-tts-2603.
            voice: The voice ID to use for synthesis, default is en_paul_neutral.
            api_key: Your MistralAI API key. If not provided, will use the
                MISTRAL_API_KEY environment variable.
            client: Optional pre-configured MistralAI client instance.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(model=model, voice=voice, response_format=response_format)

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MistralAI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMATS] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model: The MistralAI TTS model to use.
            voice: The voice ID to use for synthesis.
            response_format: The audio format of the synthesized speech.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(response_format):
            self._opts.response_format = response_format

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """ChunkedStream for MistralAI TTS. Sends the full text to the API and
    returns the synthesized audio as a single chunk."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            resp = await self._tts._client.audio.speech.complete_async(
                model=self._opts.model,
                input=self.input_text,
                voice_id=self._opts.voice,
                response_format=self._opts.response_format,
            )

            output_emitter.initialize(
                request_id="",
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=f"audio/{self._opts.response_format}",
            )

            audio_bytes = base64.b64decode(resp.audio_data)
            output_emitter.push(audio_bytes)
            output_emitter.flush()

        except SDKError as e:
            if e.status_code in (408, 504):
                raise APITimeoutError() from e
            else:
                raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
