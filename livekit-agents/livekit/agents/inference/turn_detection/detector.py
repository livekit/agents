from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import aiohttp

from livekit import rtc

from ... import utils
from ...language import LanguageCode
from ...types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from .languages import LANGUAGES

if TYPE_CHECKING:
    from .base import BaseAudioTurnDetectionStream


DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


@dataclass
class TurnDetectionEvent:
    type: Literal["eot_prediction"]
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None = None


@dataclass
class TurnDetectorOptions:
    sample_rate: int
    base_url: str
    api_key: str
    api_secret: str
    conn_options: APIConnectOptions


class AudioTurnDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        lk_base_url = utils.resolve_env_var(
            base_url, "LIVEKIT_INFERENCE_URL", default=DEFAULT_BASE_URL
        )
        lk_api_key = utils.resolve_env_var(
            api_key, "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY", default=""
        )
        lk_api_secret = utils.resolve_env_var(
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
            sample_rate=sample_rate,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            conn_options=conn_options,
        )

        self._session = http_session
        self._streams: weakref.WeakSet[BaseAudioTurnDetectionStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "eot-audio"

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
    ) -> BaseAudioTurnDetectionStream:
        from .stream import AudioTurnDetectionStream

        stream: BaseAudioTurnDetectionStream = AudioTurnDetectionStream(
            detector=self,
            opts=self._opts,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        lang_key = language.language if language is not None else "en"
        return LANGUAGES.get(lang_key)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self.unlikely_threshold(language) is not None

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        self._session = None
