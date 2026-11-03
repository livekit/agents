"""Telemetry API; exporter implementations load when telemetry is configured."""

import importlib
from typing import TYPE_CHECKING, Any

from . import metrics as metrics, trace_types as trace_types, utils as utils
from ._context import (
    capture_content_enabled as capture_content_enabled,
    set_capture_content as set_capture_content,
    tracer as tracer,
)
from ._diagnostics import diagnostics as diagnostics

if TYPE_CHECKING:
    from ._providers import set_tracer_provider as set_tracer_provider
    from ._redaction import (
        RedactingLogExporter as RedactingLogExporter,
        RedactingSpanExporter as RedactingSpanExporter,
    )


def __getattr__(name: str) -> Any:
    module = {
        "RedactingLogExporter": "_redaction",
        "RedactingSpanExporter": "_redaction",
        "set_tracer_provider": "_providers",
    }.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# the documented surface (docs/telemetry.md); the submodules are the framework's own
__all__ = [
    "diagnostics",
    "RedactingSpanExporter",
    "RedactingLogExporter",
    "set_tracer_provider",
    "set_capture_content",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
