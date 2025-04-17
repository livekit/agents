from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, _EUORunnerBase


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    def __init__(self):
        super().__init__("multilingual")


class MultilingualModel(EOUModelBase):
    def __init__(self, **kwargs):
        super().__init__(model_type="multilingual", **kwargs)

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD


_InferenceRunner.register_runner(_EUORunnerMultilingual)
