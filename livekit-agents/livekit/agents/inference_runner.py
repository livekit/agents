from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, Type


class _RunnerMeta(Protocol):
    INFERENCE_METHOD: ClassVar[str]


_RunnersDict = dict[str, Type["_InferenceRunner"]]


# kept private until we stabilize the API (only used for EOU today)
class _InferenceRunner(ABC, _RunnerMeta):
    registered_runners: _RunnersDict = {}

    @classmethod
    def register_runner(cls, runner_class: Type["_InferenceRunner"]) -> None:
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("InferenceRunner must be registered on the main thread")

        if runner_class.INFERENCE_METHOD in cls.registered_runners:
            raise ValueError(
                f"InferenceRunner {runner_class.INFERENCE_METHOD} already registered"
            )

        cls.registered_runners[runner_class.INFERENCE_METHOD] = runner_class

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the runner. This is used to load models, etc."""
        ...

    @abstractmethod
    def run(self, data: bytes) -> bytes | None:
        """Run inference on the given data."""
        ...
