from typing import Union

from .base import LLMError, STTError, TTSError

Error = Union[
    LLMError,
    STTError,
    TTSError,
]

__all__ = ["LLMError", "STTError", "TTSError", "Error"]
