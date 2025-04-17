from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, _EUORunnerBase


class _EUORunnerEn(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_en"

    def __init__(self):
        super().__init__("en")


class EnglishModel(EOUModelBase):
    def __init__(self, custom_threshold: float | None = None):
        super().__init__(model_type="en", custom_threshold=custom_threshold)

    def _inference_method(self) -> str:
        return _EUORunnerEn.INFERENCE_METHOD


_InferenceRunner.register_runner(_EUORunnerEn)
