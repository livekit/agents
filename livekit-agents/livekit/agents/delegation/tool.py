"""What a delegation leaves on the call that made it, and the note that keeps a call going.

``A2ADelegate`` (``a2a.py``) is the tool itself; this is what it writes on the call and what it
says on a call while the expert works.
"""

from __future__ import annotations

from ..llm.chat_context import FunctionCall
from .delegate import DelegationResult

TASK_ID_KEY = "lk.delegation.task_id"
CONTEXT_ID_KEY = "lk.delegation.context_id"
ENDPOINT_KEY = "lk.delegation.endpoint"
TRANSPORT_KEY = "lk.delegation.transport"
PROFILE_KEY = "lk.delegation.profile"
"""``FunctionCall.extra``: the expert's task that answered the call and its context, where the
expert was reached, over which binding, under which profile (``livekit`` or ``a2a``), so a
dashboard joins the delegation to the call as it ran, and the next delegation to a stock expert
goes back to the context it keeps."""


def record_delegation(function_call: FunctionCall, result: DelegationResult) -> None:
    """Writes the expert's task id and how it was reached on the call that delegated."""
    recorded = (
        (TASK_ID_KEY, result.task_id),
        (CONTEXT_ID_KEY, result.context_id),
        (ENDPOINT_KEY, result.endpoint),
        (TRANSPORT_KEY, result.transport),
        (PROFILE_KEY, result.profile),
    )
    for key, value in recorded:
        if value:
            function_call.extra[key] = value


DISPATCHED = (
    'Acknowledge in a few natural words, "one moment", "sure, let me check", "okay, looking '
    'now", varying the wording, restating none of the request, asking nothing and promising '
    "nothing about the outcome. The answer is a separate entry, not this one."
)


__all__ = [
    "CONTEXT_ID_KEY",
    "DISPATCHED",
    "ENDPOINT_KEY",
    "PROFILE_KEY",
    "TASK_ID_KEY",
    "TRANSPORT_KEY",
    "record_delegation",
]
