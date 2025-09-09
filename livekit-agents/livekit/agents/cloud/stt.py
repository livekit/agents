from __future__ import annotations

import os
from dataclasses import dataclass

from .. import stt
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .models import STTModels


@dataclass
class STTOptions:
    model: NotGivenOr[STTModels | str]
    language: NotGivenOr[str]
    base_url: str
    api_key: str
    api_secret: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
        )

        lk_base_url = base_url if is_given(base_url) else os.environ.get("LIVEKIT_URL")
        if not lk_base_url:
            raise ValueError(
                "LIVEKIT_URL is required, either as argument or set LIVEKIT_URL environmental variable"
            )

        lk_api_key = api_key if is_given(api_key) else os.environ.get("LIVEKIT_API_KEY")
        if not lk_api_key:
            raise ValueError(
                "LIVEKIT_API_KEY is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        lk_api_secret = api_secret if is_given(api_secret) else os.environ.get("LIVEKIT_API_SECRET")
        if not lk_api_secret:
            raise ValueError(
                "LIVEKIT_API_SECRET is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        self._opts = STTOptions(
            model=model,
            language=language,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
        )
