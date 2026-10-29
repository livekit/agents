# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from .chat_items import MetricsReport
from .metrics import AgentMetrics
from .telemetry import OtelValue

def metrics_log_fields(metrics: AgentMetrics) -> tuple[str, dict[str, object]] | None:
    """`log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
    (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
    """

def otel_turn_records(
    report: MetricsReport | None,
) -> list[tuple[str, str, OtelValue, dict[str, str]]]:
    """`_record_turn_metrics(report)` minus the recording. `None` and `{}` yield nothing; a
    present `0.0` yields a record.
    """

def otel_usage_records(metrics: AgentMetrics) -> list[tuple[str, str, OtelValue, dict[str, str]]]:
    """`collect_usage(ev)` minus the recording: the counter increments it would make. Empty for
    unhandled types, unknown types and honest zeros alike (the reference gates on truthiness).
    """

class ModelUsageCollector:
    """`metrics.ModelUsageCollector` -- the live per-model rollup."""
    def __init__(self) -> None: ...
    def collect(self, metrics: AgentMetrics) -> None: ...
    def flatten(self) -> list[dict[str, object]]: ...
