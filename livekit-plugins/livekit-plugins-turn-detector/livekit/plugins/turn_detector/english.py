from __future__ import annotations

from livekit.agents import Plugin
from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, EOUPlugin, _EUORunnerBase
from .models import EOUModelType


class _EUORunnerEn(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_en"

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

    def _inference_method(self) -> str:
        return _EUORunnerEn.INFERENCE_METHOD


_InferenceRunner.register_runner(_EUORunnerEn)
Plugin.register_plugin(EOUPlugin(_EUORunnerEn))
