from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, _EUORunnerBase


class _EUORunnerEn(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_en"

    def __init__(self):
        super().__init__("en")


class EnglishModel(EOUModelBase):
    def __init__(self, **kwargs):
        super().__init__(model_type="en", **kwargs)

    def _inference_method(self) -> str:
        return _EUORunnerEn.INFERENCE_METHOD


_InferenceRunner.register_runner(_EUORunnerEn)
