"""
Provides base classes for implementing inference runners that handle:
- Model initialization
- Data processing
- Thread-safe execution
- Plugin registration system

Used internally by LiveKit agents for AI/ML operations like:
- Voice activity detection
- Speech-to-text
- NLP processing
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, Type


class _RunnerMeta(Protocol):
    """Protocol enforcing inference method identifier in runner implementations"""
    INFERENCE_METHOD: ClassVar[str]  # Unique identifier for runner type


_RunnersDict = dict[str, Type["_InferenceRunner"]]


# kept private until we stabilize the API (only used for EOU today)
class _InferenceRunner(ABC, _RunnerMeta):
    """
    Abstract base class for inference runners. Implement this to create:
    - Model-specific initialization
    - Custom inference pipelines
    - Hardware-accelerated processing
    
    Note: Currently internal API - stability not guaranteed
    
    Usage:
    1. Subclass and implement abstract methods
    2. Set INFERENCE_METHOD class variable
    3. Register with register_runner()
    """
    
    registered_runners: _RunnersDict = {}  # Global registry of available runners

    @classmethod
    def register_runner(cls, runner_class: Type["_InferenceRunner"]) -> None:
        """Register a runner implementation. Must be called on main thread.
        
        Args:
            runner_class: Subclass of _InferenceRunner to register
            
        Raises:
            RuntimeError: If called from non-main thread
            ValueError: For duplicate registration attempts
        """
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("InferenceRunner must be registered on the main thread")

        if runner_class.INFERENCE_METHOD in cls.registered_runners:
            raise ValueError(
                f"InferenceRunner {runner_class.INFERENCE_METHOD} already registered"
            )

        cls.registered_runners[runner_class.INFERENCE_METHOD] = runner_class

    @abstractmethod
    def initialize(self) -> None:
        """One-time initialization for resource setup. Called before any run() calls.
        
        Typical uses:
        - Loading ML models
        - Allocating GPU buffers
        - Initializing hardware accelerators
        """
        ...

    @abstractmethod
    def run(self, data: bytes) -> bytes | None:
        """Execute inference on input data.
        
        Args:
            data: Serialized input (format depends on implementation)
            
        Returns:
            bytes | None: Serialized output or None for no result
            
        Note:
            Implementation must be thread-safe - may be called concurrently
        """
        ...
