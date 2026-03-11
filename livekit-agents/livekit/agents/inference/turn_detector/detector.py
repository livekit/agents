from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from ... import utils
from ...language import LanguageCode
from ...log import logger
from ...types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ...utils import resolve_env_var
from .languages import LANGUAGES
from .types import TurnDetectorEncoding

if TYPE_CHECKING:
    from .stream import MultiModalTurnDetectionStream

TURN_DETECTOR_HTTP_ENV = "LIVEKIT_TURN_DETECTOR_HTTP"

INFERENCE_TIMEOUT = 1
INFERENCE_INTERVAL = 0.1

DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_ENCODING: TurnDetectorEncoding = "pcm_s16le"
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


@dataclass
class TurnDetectionEvent:
    type: Literal["eou_prediction"]
    end_of_turn_probability: float
    last_speaking_time: float


@dataclass
class TurnDetectorOptions:
    encoding: TurnDetectorEncoding
    sample_rate: int
    base_url: str
    api_key: str
    api_secret: str
    conn_options: APIConnectOptions


class MultiModalTurnDetector:
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        encoding: TurnDetectorEncoding = DEFAULT_ENCODING,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        lk_base_url = resolve_env_var(base_url, "LIVEKIT_INFERENCE_URL", default=DEFAULT_BASE_URL)
        lk_api_key = resolve_env_var(
            api_key, "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY", default=""
        )
        lk_api_secret = resolve_env_var(
            api_secret, "LIVEKIT_INFERENCE_API_SECRET", "LIVEKIT_API_SECRET", default=""
        )
        if not lk_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET env var"
            )
        if not lk_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY env var"
            )

        self._opts = TurnDetectorOptions(
            encoding=encoding,
            sample_rate=sample_rate,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            conn_options=conn_options,
        )

        self._session = http_session
        self._streams: weakref.WeakSet[MultiModalTurnDetectionStream] = weakref.WeakSet()
        self._latest_eou_probability = asyncio.Future[float]()

    @property
    def model(self) -> str:
        return "multimodal"

    @property
    def provider(self) -> str:
        return "livekit"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> MultiModalTurnDetectionStream:
        from .stream import HTTPStream, WSStream

        # TODO: @chenghao-mou replace this with the EOT remote url env check
        use_http = os.getenv(TURN_DETECTOR_HTTP_ENV, "").lower() in ("1", "true", "yes")

        stream: MultiModalTurnDetectionStream
        if use_http:
            stream = HTTPStream(
                detector=self,
                opts=self._opts,
                conn_options=conn_options,
            )
        else:
            stream = WSStream(
                detector=self,
                opts=self._opts,
                conn_options=conn_options,
            )
        self._streams.add(stream)
        return stream

    async def unlikely_threshold(self, language: LanguageCode | None) -> float:
        return LANGUAGES.get(language.language if language is not None else "en", 0.3)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return language is not None and language.language in LANGUAGES

    async def predict_end_of_turn(
        self,
        chat_ctx: Any,
        *,
        timeout: float | None = None,
    ) -> float:
        """chat_ctx should have been streamed to the model by now."""
        try:
            return await asyncio.wait_for(self._latest_eou_probability, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("EOU prediction timed out", extra={"timeout": timeout})
            return 0.0

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        if self._session:
            await self._session.close()
        self._session = None
