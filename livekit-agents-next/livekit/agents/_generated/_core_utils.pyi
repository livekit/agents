# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from ..types import NotGivenOr

class ExpFilter:
    def __init__(
        self,
        alpha: float,
        max_val: NotGivenOr[float] = ...,
        min_val: NotGivenOr[float] = ...,
        initial: NotGivenOr[float] = ...,
    ) -> None: ...
    def reset(
        self,
        alpha: NotGivenOr[float] = ...,
        initial: NotGivenOr[float] = ...,
        min_val: NotGivenOr[float] = ...,
        max_val: NotGivenOr[float] = ...,
    ) -> None: ...
    def apply(self, exp: float, sample: NotGivenOr[float] = ...) -> float: ...
    @property
    def value(self) -> float | None: ...
    def update_base(self, alpha: float) -> None: ...

class MovingAverage:
    def __init__(self, window_size: int) -> None: ...
    def add_sample(self, sample: float) -> None: ...
    def get_avg(self) -> float: ...
    def reset(self) -> None: ...
    def size(self) -> int: ...
