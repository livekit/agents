from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tts,
    utils,
)

from .log import logger
from .models import TTSModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/tts"
QWEN_BASE_URL = "https://api.simplismart.live/v1/audio/speech"
DEFAULT_ORPHEUS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
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
        http_session: http.HttpSession | None = None,
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
            http_session: Optional HttpSession for reuse.
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

        resolved_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not resolved_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._api_key = resolved_key
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SimpliSmart"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            opts=replace(self._opts),
            api_key=self._api_key,
            base_url=self._base_url,
            ensure_session=self._ensure_session,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        api_key: str,
        base_url: str,
        ensure_session: Callable[[], http.HttpSession],
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self._ensure_session = ensure_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._opts.qwen_options is not None:
            qwen_opts = self._opts.qwen_options
            payload: dict[str, str | bool | float | int] = {
                "model": self._opts.model,
                "text": self._input_text,
                "language": qwen_opts.language,
                "voice": self._opts.voice,
                "leading_silence": qwen_opts.leading_silence,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
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
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

        logger.debug("TTS request to %s (model: %s)", self._base_url, self._opts.model)

        try:
            async with self._ensure_session().post(
                self._base_url,
                json=payload,
                headers=headers,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )

                async for audio_data in resp.iter_bytes():
                    if audio_data:
                        output_emitter.push(audio_data)

                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("Unexpected error in Simplismart TTS", detail=str(e)) from e
