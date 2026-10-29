"""Speaking A2A: the text door, its client, and the LiveKit agents extension.

Every byte that crosses a process boundary is A2A JSON, and the extension at
``https://livekit.io/a2a/ext/agents/v1`` is the whole of what LiveKit adds to it.
``@server.a2a_session`` serves an endpoint; ``AgentClient`` talks to one.
"""

from .._generated.a2a_client import (
    Artifact,
    DataPart,
    FilePart,
    Message,
    MessageMetadata,
    TaskUpdate,
    TextPart,
)
from ..storage.context import A2ASessionContext
from .auth import AgentThreadGrant, Credentials, Forbidden, Principal, Unauthorized
from .client import AgentClient, Reply
from .endpoint import a2a_session
from .remote import RemoteAgent

# the documented surface (docs/clients.md, docs/server.md, docs/tools.md): the endpoint and card
__all__ = [
    "a2a_session",
    "A2ASessionContext",
    "AgentClient",
    "Artifact",
    "DataPart",
    "FilePart",
    "Message",
    "MessageMetadata",
    "RemoteAgent",
    "TextPart",
    "AgentThreadGrant",
    "Credentials",
    "Forbidden",
    "Principal",
    "Reply",
    "TaskUpdate",
    "Unauthorized",
]
