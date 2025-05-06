from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSModel


@dataclass
class _TTSOptions:
    model: TTSModel | str
    language: str
    sample_rate: int
    voice: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        user_id: str | None = None,
        model: TTSModel | str = "PlayDialog-turbo",
        voice: str = "Atlas-PlayAI",
        language: str = "english",
        sample_rate: int = 24000,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialize the PlayAI TTS engine.

        Args:
            api_key (str): PlayAI API key.
            user_id (str): PlayAI user ID.
            model (TTSModel): TTS model, defaults to "PlayDialog-turbo".
            voice (str): Voice manifest URL.
            language (str): language, defaults to "english".
            sample_rate (int): sample rate (Hz), A number greater than or equal to 8000,
                and must be less than or equal to 48000
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._api_key = api_key or os.environ.get("PLAYHT_API_KEY")
        self._user_id = user_id or os.environ.get("PLAYHT_USER_ID")

        if not self._api_key or not self._user_id:
            raise ValueError(
                "PlayHT API key and user ID are required. Set environment variables PLAYHT_API_KEY "
                "and PLAYHT_USER_ID or pass them explicitly."
            )

        self._opts = _TTSOptions(
            voice=voice, model=model, sample_rate=sample_rate, language=language
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModel | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        try:
            async with self._tts._ensure_session().post(
                "https://api.play.ht/api/v2/tts/stream",
                # headers from https://github.com/playht/pyht/blob/master/pyht/client.py
                headers={
                    "authorization": f"Bearer {self._tts._api_key}",
                    "x-user-id": self._tts._user_id,
                    "accept": "audio/wav",
                },
                json={
                    "text": self._input_text,
                    "voice": self._opts.voice,
                    "voice_engine": self._opts.model,
                    "output_format": "wav",
                    "sample_rate": self._opts.sample_rate,
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/wav",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
