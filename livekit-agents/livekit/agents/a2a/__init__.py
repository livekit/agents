"""Speaking A2A: the text door, its client, and the LiveKit agents extension.

Every byte that crosses a process boundary is A2A JSON, and the extension at
``https://livekit.io/a2a/ext/agents/v1`` is the whole of what LiveKit adds to it.
``@server.a2a_session`` serves an endpoint; ``AgentClient`` talks to one.
"""

from ..storage.context import A2ASessionContext
from .auth import AgentThreadGrant, Credentials, Forbidden, Principal, Unauthorized
from .client import AgentClient, Reply
from .codec import TaskUpdate, updates_from_events
from .endpoint import AGENT_CARD_PATH, RPC_PATH, A2AEndpoint, a2a_session
from .extension import EXTENSION_URI, EXTENSIONS_HEADER, Extensions, agent_card
from .methods import A2AService, CallContext, TurnOutcome

__all__ = [
    "a2a_session",
    "AGENT_CARD_PATH",
    "EXTENSIONS_HEADER",
    "EXTENSION_URI",
    "RPC_PATH",
    "A2AEndpoint",
    "A2AService",
    "A2ASessionContext",
    "AgentClient",
    "AgentThreadGrant",
    "CallContext",
    "Credentials",
    "Extensions",
    "Forbidden",
    "Principal",
    "Reply",
    "TaskUpdate",
    "TurnOutcome",
    "Unauthorized",
    "agent_card",
    "updates_from_events",
]
