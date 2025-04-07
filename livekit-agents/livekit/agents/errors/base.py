from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict


class _Error(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp: float
    label: str
    error: str
    recoverable: bool


class LLMError(_Error):
    type: Literal["llm_error"] = "llm_error"


class STTError(_Error):
    type: Literal["stt_error"] = "stt_error"


class TTSError(_Error):
    type: Literal["tts_error"] = "tts_error"


Errors = Union[
    LLMError,
    STTError,
    TTSError,
]
