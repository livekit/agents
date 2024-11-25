from __future__ import annotations

import threading
from abc import ABC, ABCMeta, abstractmethod
from typing import Type


class _RunnerMeta(ABCMeta):
    @property
    @abstractmethod
    def METHOD(cls) -> str: ...


# kept private until we stabilize the API (only used for EOU today)
class _InferenceRunner(ABC, metaclass=_RunnerMeta):
    registered_runners: dict[str, Type["_InferenceRunner"]] = {}

    @classmethod
    def register_runner(cls, runner_class: Type["_InferenceRunner"]) -> None:
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("InferenceRunner must be registered on the main thread")

        if runner_class.METHOD in cls.registered_runners:
            raise ValueError(
                f"InferenceRunner {runner_class.METHOD} already registered"
            )

        cls.registered_runners[runner_class.METHOD] = runner_class

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the runner. This is used to load models, etc."""
        ...

    @abstractmethod
    def run(self, data: bytes) -> bytes | None:
        """Run inference on the given data."""
        ...
