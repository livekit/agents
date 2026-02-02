import asyncio
import os

import aiohttp

# from .log import logger
from pydantic import BaseModel

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .models import TTSModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/tts"


class SimplismartTTSOptions(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    max_tokens: int = 1000


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str = SIMPLISMART_BASE_URL,
        model: TTSModels | str = "canopylabs/orpheus-3b-0.1-ft",
        voice: str = "tara",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        max_tokens: int = 1000,
    ) -> None:
        """
        Configuration options for SimpliSmart TTS (Text-to-Speech).

        Attributes:
            temperature (float): Controls the randomness in the model output. Lower values make output more deterministic.
            top_p (float): Nucleus sampling probability threshold. Limits the sampling pool of predicted tokens.
            repetition_penalty (float): Penalty applied to repeated text to reduce repetition.
            max_tokens (int): Maximum number of output tokens allowed in the synthesized speech.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

        self._base_url = base_url
        self._model = model
        self._voice = voice
        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._session = http_session

        self._opts = SimplismartTTSOptions(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

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
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = tts._opts
        self._model = tts._model

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = self._opts.model_dump()
        payload["prompt"] = self._input_text
        payload["voice"] = self._tts._voice
        payload["model"] = self._tts._model

        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
        }

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
                resp.raise_for_status()
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
                async for audio_data, _ in resp.content.iter_chunks():
                    output_emitter.push(audio_data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
