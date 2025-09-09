from __future__ import annotations

import os
from dataclasses import dataclass

from .. import tts
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .models import TTSModels


@dataclass
class _TTSOptions:
    model: NotGivenOr[TTSModels | str]
    voice: NotGivenOr[str]
    language: NotGivenOr[str]
    base_url: str
    api_key: str
    api_secret: str


DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
    ):
        """Livekit Cloud Inference TTS

        Args:
            model (TTSModels | str, optional): TTS model to use, in "provider/model" format, use a default one if not provided
            voice (str, optional): Voice to use, use a default one if not provided
            language (str, optional): Language of the TTS model.
            base_url (str, optional): LIVEKIT_URL, if not provided, read from environment variable.
            api_key (str, optional): LIVEKIT_API_KEY, if not provided, read from environment variable.
            api_secret (str, optional): LIVEKIT_API_SECRET, if not provided, read from environment variable.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=DEFAULT_NUM_CHANNELS,
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

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
        )
