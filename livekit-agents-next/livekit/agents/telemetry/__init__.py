from . import (
    metrics as metrics,
    trace_types as trace_types,
    utils as utils,
)
from ._export import diagnostics
from .traces import (
    RedactingLogExporter,
    RedactingSpanExporter,
    _upload_session_report as _upload_session_report,
    capture_content_enabled as capture_content_enabled,
    set_capture_content,
    set_tracer_provider,
    tracer as tracer,
)

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
