# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/amd.rs

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

# Record declarations:


@unique
class AMDCategory(StrEnum):
    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    UNCERTAIN = "uncertain"


@dataclass(kw_only=True, slots=True)
class AMDPredictionEvent:
    speech_duration: float
    category: AMDCategory
    reason: Literal["detection_timeout", "no_speech_timeout", "short_greeting", "llm", "participant_missing"]
    transcript: str
    delay: float
    type: Literal["amd_prediction"] = "amd_prediction"

    @property
    def is_human(self) -> bool:
        from .. import _core

        return _core.amd_prediction_is_human(self)

    @property
    def is_machine(self) -> bool:
        from .. import _core

        return _core.amd_prediction_is_machine(self)


class DetectionOptions(TypedDict, total=False):
    """Per-`AMD` overrides of the classifier's timings and prompt (`voice/amd/detector.py`)."""

    human_speech_threshold: float
    human_silence_threshold: float
    machine_silence_threshold: float
    no_speech_threshold: float
    timeout: float
    max_endpointing_delay: float
    prompt: str


TranscriptSource: TypeAlias = Literal["stt", "amd_stt"]
"""With AMD's own STT, session transcripts are dropped, or the greeting is seen twice."""


class AMDConfig(TypedDict):
    """What `AMD` attaches to the session: the user's overrides and the settings the host resolved."""

    detection: DetectionOptions
    """The user's overrides; every one left out is the classifier's own."""
    max_endpointing_delay: NotRequired[float]
    """Already resolved by the host (`amd_resolve_max_endpointing_delay`)."""
    source: NotRequired[TranscriptSource]
    """``"stt"`` for the session's finals, ``"amd_stt"`` for a recogniser of AMD's own."""
    wait_until_finished: bool
    interrupt_on_machine: bool
    ivr_detection: bool


AMDCategory.__module__ = "livekit.agents.voice.amd"
DetectionOptions.__module__ = "livekit.agents.voice.amd"


DEFAULT_LLM_MODEL: Final = "google/gemini-3.1-flash-lite"
"""The LLM an `AMD` auto-selects: the one the benchmark ran on."""
DEFAULT_STT_MODEL: Final = "cartesia/ink-whisper"
"""The STT an `AMD` auto-selects: the one the benchmark ran on."""
