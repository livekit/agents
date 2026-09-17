"""Speaking A2A, with the LiveKit agent session extension.

One wire protocol: every byte that crosses a process boundary is A2A JSON, and the extension
at ``https://livekit.io/a2a/ext/agent-session/v1`` is the whole of what LiveKit adds to it. A
chat client sending a person's text and a voice agent sending a delegation use the same
types, the same client and the same server. Delegation is one use of the profile, not the
whole of it.

Needs the ``a2a`` extra: ``pip install 'livekit-agents[a2a]'``.
"""

from ..voice.served_request import Directive
from ._client import A2AClient, TaskStream
from ._codec import (
    from_a2a_events,
    from_a2a_request,
    to_a2a_events,
    to_a2a_request,
)
from ._extension import (
    ANSWER_ARTIFACT_NAME,
    DIRECTIVE,
    EXTENSION_URI,
    KIND,
    REASON,
    VERBATIM,
    agent_card,
)
from ._server import A2ASessionContext, A2ASessionHandler
from ._types import TaskInput, TaskState, TaskUpdate

__all__ = [
    "ANSWER_ARTIFACT_NAME",
    "DIRECTIVE",
    "EXTENSION_URI",
    "KIND",
    "REASON",
    "VERBATIM",
    "A2AClient",
    "A2ASessionContext",
    "A2ASessionHandler",
    "Directive",
    "TaskInput",
    "TaskState",
    "TaskStream",
    "TaskUpdate",
    "agent_card",
    "from_a2a_events",
    "from_a2a_request",
    "to_a2a_events",
    "to_a2a_request",
]
