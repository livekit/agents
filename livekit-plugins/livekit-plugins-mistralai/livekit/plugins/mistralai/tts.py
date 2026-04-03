from __future__ import annotations

import base64
import os
import struct
import uuid
from dataclasses import dataclass, replace
from typing import Literal

import httpx

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


def _f32le_to_s16le(data: bytes) -> bytes:
    n = len(data) // 4
    floats = struct.unpack(f"<{n}f", data)
    return struct.pack(f"<{n}h", *(max(-32768, min(32767, int(s * 32767))) for s in floats))


SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_MODEL: TTSModels = "voxtral-mini-tts-2603"
DEFAULT_VOICE: TTSVoices = "en_paul_neutral"

RESPONSE_FORMATS = Literal["mp3", "wav", "pcm", "opus", "flac"]
DEFAULT_RESPONSE_FORMAT: RESPONSE_FORMATS = "pcm"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str | None
    ref_audio: str | None
    response_format: RESPONSE_FORMATS


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: Mistral | None = None,
        response_format: RESPONSE_FORMATS = DEFAULT_RESPONSE_FORMAT,
    ) -> None:
        """
        Create a new instance of MistralAI TTS.

        Args:
            model: The MistralAI TTS model to use, default is voxtral-mini-tts-2603.
            voice: The voice ID to use for synthesis. Mutually exclusive with
                ``ref_audio``. Defaults to ``en_paul_neutral`` when neither is given.
            ref_audio: Base64-encoded audio sample (3–25 s) for zero-shot voice
                cloning. Mutually exclusive with ``voice``.
            api_key: Your MistralAI API key. If not provided, will use the
                MISTRAL_API_KEY environment variable.
            client: Optional pre-configured MistralAI client instance.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")

        resolved_voice: TTSVoices | str | None = voice if is_given(voice) else None
        resolved_ref_audio: str | None = ref_audio if is_given(ref_audio) else None
        if resolved_voice is None and resolved_ref_audio is None:
            resolved_voice = DEFAULT_VOICE

        self._opts = _TTSOptions(
            model=model,
            voice=resolved_voice,
            ref_audio=resolved_ref_audio,
            response_format=response_format,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
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
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMATS] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model: The MistralAI TTS model to use.
            voice: The voice ID to use for synthesis. Clears ``ref_audio``.
            ref_audio: Base64-encoded audio sample for zero-shot voice cloning.
                Clears ``voice``.
            response_format: The audio format of the synthesized speech.
        """
        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
            self._opts.ref_audio = None
        if is_given(ref_audio):
            self._opts.ref_audio = ref_audio
            self._opts.voice = None
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
            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=f"audio/{self._opts.response_format}",
            )
            if self._opts.ref_audio is not None:
                stream = await self._tts._client.audio.speech.complete_async(
                    model=self._opts.model,
                    input=self.input_text,
                    ref_audio=self._opts.ref_audio,
                    response_format=self._opts.response_format,
                    timeout_ms=int(self._conn_options.timeout * 1000),
                    stream=True,
                )
            else:
                stream = await self._tts._client.audio.speech.complete_async(
                    model=self._opts.model,
                    input=self.input_text,
                    voice_id=self._opts.voice or DEFAULT_VOICE,
                    response_format=self._opts.response_format,
                    timeout_ms=int(self._conn_options.timeout * 1000),
                    stream=True,
                )
            async for ev in stream:
                if ev.event == "speech.audio.delta":
                    data = base64.b64decode(ev.data.audio_data)
                    if self._opts.response_format == "pcm":
                        data = _f32le_to_s16le(data)
                    output_emitter.push(data)
                elif ev.event == "speech.audio.done":
                    self._set_token_usage(
                        input_tokens=ev.data.usage.prompt_tokens,
                        output_tokens=ev.data.usage.completion_tokens,
                    )

            output_emitter.flush()

        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except SDKError as e:
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
