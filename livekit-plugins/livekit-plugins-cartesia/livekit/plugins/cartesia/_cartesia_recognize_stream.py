from abc import abstractmethod

from livekit.agents import (
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .models import STTLanguages


class CartesiaRecognizeStream(stt.RecognizeStream):
    """This ABC exists for backward compatibility"""

    @abstractmethod
    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> None: ...
