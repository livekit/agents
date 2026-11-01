"""Adaptive interruption detection: the detector is the core's (``_core.InterruptionDetector``);
its generated event and error are exposed here."""

from __future__ import annotations

from typing import Any

from livekit.agents._generated.interruption_errors import (
    InterruptionDetectionError as InterruptionDetectionError,
)

from .. import _core
from .._generated.overlap import OverlappingSpeechEvent as OverlappingSpeechEvent
from ._utils import Credentials

__all__ = ["AdaptiveInterruptionDetector", "InterruptionDetectionError", "OverlappingSpeechEvent"]


class AdaptiveInterruptionDetector(_core.InterruptionDetector):
    """The gateway model that tells an interruption from a backchannel while the agent speaks.
    ``base_url``, ``api_key`` and ``api_secret`` default to the server's settings."""

    def __new__(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        **options: Any,
    ) -> AdaptiveInterruptionDetector:
        credentials = Credentials.resolve(base_url, api_key, api_secret)
        return super().__new__(
            cls,
            base_url=credentials.base_url,
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            **options,
        )
