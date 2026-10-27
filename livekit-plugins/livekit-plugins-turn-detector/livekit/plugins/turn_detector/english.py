from __future__ import annotations

from .base import EOUModelBase, _EUORunnerBase
from .models import EOUModelType


class _EUORunnerEn(_EUORunnerBase):
    @classmethod
    def model_type(cls) -> EOUModelType:
        return "en"

    def _normalize_text(self, text: str) -> str:
        """
        The english model is trained on the original chat context without normalization.
        """
        if not text:
            return ""

        return text


class EnglishModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(model_type="en", unlikely_threshold=unlikely_threshold)
        _EUORunnerEn.load()

    def _runner(self) -> type[_EUORunnerBase]:
        return _EUORunnerEn
