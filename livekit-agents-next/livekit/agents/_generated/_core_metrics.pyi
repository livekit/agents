# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from .metrics import AgentMetrics

def runtime_busy_seconds() -> list[float]:
    """Each async worker's total busy seconds so far; empty before the runtime started."""

def metrics_log_fields(metrics: AgentMetrics) -> tuple[str, dict[str, object]] | None:
    """`log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
    (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
    """
