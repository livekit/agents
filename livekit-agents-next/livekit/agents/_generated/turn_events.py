# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/language/eot_thresholds.rs
#     rust/crates/lk-agents-abi/src/turn/eot.rs
#     rust/crates/lk-agents-abi/src/turn/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal, TypeAlias

TurnDetectorModel: TypeAlias = Literal["turn-detector-v1", "turn-detector-v1-mini"]


@dataclass
class TurnDetectionEvent:
    type: Literal['eot_prediction']
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None = None
    """Latest input audio creation time -> prediction receive time."""
    inference_duration: float | None = None
    """Server-side model inference time."""
    backchannel_probability: float | None = None
    """How appropriate it is for the agent to backchannel at this pause.
        ``None`` when the detector does not produce one (e.g. the local mini model).
    """


@dataclass(kw_only=True, slots=True)
class EotPrediction:
    """One end-of-turn prediction. Optional timing and backchannel data depend on the detector.

    Read from the generated prediction record returned by a host detector's `_core_predict()`.
    """

    end_of_turn_probability: float
    """How likely the user has finished their turn, `[0, 1]`."""
    last_speaking_time: float = 0.0
    detection_delay: float | None = None
    """Latest input audio creation time -> prediction receive time."""
    inference_duration: float | None = None
    """Server-side model inference time."""
    backchannel_probability: float | None = None
    """How appropriate it is to backchannel at this pause; `None` when the detector has none."""


ThresholdOverride: TypeAlias = float | dict[str, float]
"""One threshold for every language, or thresholds per language merged over the defaults."""


TurnDetectionEvent.__annotations__ = get_annotations(TurnDetectionEvent, eval_str=True)
TurnDetectionEvent.__module__ = "livekit.agents.voice.io"
