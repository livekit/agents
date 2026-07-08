from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass, replace
from typing import cast

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/tts"
QWEN_BASE_URL = "https://api.simplismart.live/v1/audio/speech"
DEFAULT_ORPHEUS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
DEFAULT_QWEN_MODEL = "qwen-tts"
DEFAULT_ORPHEUS_VOICE = "tara"
DEFAULT_QWEN_VOICE = "Chelsie"


@dataclass
class _SimplismartTTSOptions:
    temperature: float
    top_p: float
    repetition_penalty: float
    max_tokens: int


@dataclass
class _QwenTTSOptions:
    language: str
    leading_silence: bool


@dataclass
class _TTSOptions:
    model: str
    voice: str
    simplismart_options: _SimplismartTTSOptions | None = None
    qwen_options: _QwenTTSOptions | None = None


def _is_qwen_model(model: str) -> bool:
    return "qwen" in model.lower()


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: TTSModels | str = DEFAULT_ORPHEUS_MODEL,
        voice: str | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        # sample_rate is used by the audio framework for playback; not sent to the server
        sample_rate: int = 24000,
        # Simplismart TTS options
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        max_tokens: int = 1000,
        # Qwen 3 TTS options
        language: str = "English",
        leading_silence: bool = True,
    ) -> None:
        """Initialize SimpliSmart TTS.

        SimpliSmart hosts multiple TTS models. The model name determines which endpoint
        and payload format to use. Defaults are set for the Orpheus model
        (``"canopylabs/orpheus-3b-0.1-ft"``).

        Args:
            base_url: Base URL for the TTS endpoint.
            model: TTS model identifier.
            voice: Voice/speaker identifier.
            api_key: API key for authentication (defaults to ``SIMPLISMART_API_KEY`` env var).
            http_session: Optional aiohttp session for reuse.
            sample_rate: Expected sample rate of the returned PCM audio (default: 24000).
                Used by the framework for playback; not sent to the server.
            temperature: Controls output randomness.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeated tokens.
            max_tokens: Maximum number of output tokens.
            language: Qwen 3 TTS only — language for synthesis (default: ``"English"``).
            leading_silence: Qwen 3 TTS only — whether to include leading silence (default: ``True``).
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        is_qwen = _is_qwen_model(model)

        self._base_url = (
            base_url
            if base_url is not None
            else (QWEN_BASE_URL if is_qwen else SIMPLISMART_BASE_URL)
        )
        self._opts = _TTSOptions(
            model=model,
            voice=voice
            if voice is not None
            else (DEFAULT_QWEN_VOICE if is_qwen else DEFAULT_ORPHEUS_VOICE),
        )

        if is_qwen:
            self._opts.qwen_options = _QwenTTSOptions(
                language=language,
                leading_silence=leading_silence,
            )
        else:
            self._opts.simplismart_options = _SimplismartTTSOptions(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            )

        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SimpliSmart"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._opts.qwen_options is not None:
            qwen_opts = self._opts.qwen_options
            payload: dict = {
                "model": self._opts.model,
                "text": self._input_text,
                "language": qwen_opts.language,
                "voice": self._opts.voice,
                "leading_silence": qwen_opts.leading_silence,
            }
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/L16",
            }
        else:
            simplismart_opts = cast(_SimplismartTTSOptions, self._opts.simplismart_options)
            payload = {
                "prompt": self._input_text,
                "voice": self._opts.voice,
                "model": self._opts.model,
                "temperature": simplismart_opts.temperature,
                "top_p": simplismart_opts.top_p,
                "repetition_penalty": simplismart_opts.repetition_penalty,
                "max_tokens": simplismart_opts.max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
            }

        logger.debug("TTS request to %s (model: %s)", self._tts._base_url, self._opts.model)

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Simplismart TTS API error: %s - %s", resp.status, error_text)
                    raise APIStatusError(
                        message=f"Simplismart TTS API Error: {error_text}",
                        status_code=resp.status,
                        request_id=None,
                        body=error_text,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )

                async for audio_data, _ in resp.content.iter_chunks():
                    if audio_data:
                        output_emitter.push(audio_data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            logger.error("Simplismart TTS API timeout: %s", e)
            raise APITimeoutError("Simplismart TTS API request timed out") from e
        except aiohttp.ClientError as e:
            logger.error("Simplismart TTS API client error: %s", e)
            raise APIConnectionError(f"Simplismart TTS API connection error: {e}") from e
        except APIStatusError:
            raise
        except Exception as e:
            logger.error("Error during Simplismart TTS processing: %s", traceback.format_exc())
            raise APIConnectionError(f"Unexpected error in Simplismart TTS: {e}") from e
