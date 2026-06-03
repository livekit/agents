from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from livekit.agents import (
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ..models import STTLanguages

if TYPE_CHECKING:
    from livekit.agents.types import NotGivenOr


class CartesiaRecognizeStream(stt.RecognizeStream):
    """Concrete instances are created by `cartesia.STT.stream()`"""

    @abstractmethod
    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> None: ...
