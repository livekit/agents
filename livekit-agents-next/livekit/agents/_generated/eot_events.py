# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/turn/eot.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class EotResolve:
    """A resolved request and its prediction measurements."""

    id: str
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None
    inference_duration: float | None
    backchannel_probability: float | None


@dataclass(frozen=True, kw_only=True, slots=True)
class EotStartInference:
    """Start inference for one request."""

    id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class EotFlush:
    """Flush the detector's buffered audio."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EotSwapToLocal:
    """Replace the cloud transport with the local mini model."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EotDegrade:
    """Drop buffered audio and stop when no detector remains."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EotWarn:
    """A detector warning and its optional cause."""

    warning: str
    reason: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class PredictionEvent:
    """The measurements a transport reports for one request."""

    id: str
    probability: float
    detection_delay: float | None
    inference_duration: float | None
    backchannel_probability: float | None
