# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/amd.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from enum import Enum, unique
from typing import Literal, TypedDict

from pydantic import BaseModel

from .. import _bindings


@unique
class AMDCategory(str, Enum):
    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    UNCERTAIN = "uncertain"


class AMDPredictionEvent(BaseModel):
    type: Literal["amd_prediction"] = "amd_prediction"
    speech_duration: float
    category: AMDCategory
    reason: str
    transcript: str
    delay: float

    @property
    def is_human(self) -> bool:
        return _bindings.amd_prediction_is_human(self)

    @property
    def is_machine(self) -> bool:
        return _bindings.amd_prediction_is_machine(self)


# --- the user's tuning, as the host takes it --------------------------------------------------

class DetectionOptions(TypedDict, total=False):
    human_speech_threshold: float
    human_silence_threshold: float
    machine_silence_threshold: float
    no_speech_threshold: float
    timeout: float
    max_endpointing_delay: float
    prompt: str


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
AMDCategory.__module__ = "livekit.agents.voice.amd.classifier"
AMDPredictionEvent.__module__ = "livekit.agents.voice.amd.classifier"
DetectionOptions.__module__ = "livekit.agents.voice.amd.detector"
