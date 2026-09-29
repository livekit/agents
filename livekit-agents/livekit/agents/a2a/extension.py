"""The LiveKit agent session extension to A2A: its URI, its metadata keys, and its card."""

from __future__ import annotations

from typing import Any

from ..log import logger

try:
    from a2a.types import a2a_pb2 as pb
    from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
    from google.protobuf import json_format, struct_pb2
except ImportError as e:
    raise ImportError(
        "The 'a2a-sdk' package is required to speak A2A but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[a2a]'"
    ) from e


__all__ = [
    "ANSWER_ARTIFACT_NAME",
    "CALLER",
    "CONVERSATION",
    "DIRECTIVE",
    "EXTENSION_DESCRIPTION",
    "EXTENSION_URI",
    "KIND",
    "KIND_CHAT_CTX",
    "KIND_CLOSE",
    "KIND_CHAT_ITEM",
    "KIND_DELEGATION",
    "REASON",
    "VERBATIM",
    "agent_card",
    "as_dict",
    "offers_extension",
    "pb",
    "struct",
    "text_of",
    "value",
]

EXTENSION_URI = "https://livekit.io/a2a/ext/agent-session/v1"
"""Identifies the profile, and prefixes every key of it.

A client asks for it with the ``A2A-Extensions`` header; not echoed back means not active.
"""

EXTENSION_DESCRIPTION = (
    "LiveKit agent session profile: chat history, typed chat items, verbatim text, directives."
)


def _key(name: str) -> str:
    """A metadata key of this profile, namespaced under its URI so nothing collides."""
    return f"{EXTENSION_URI}/{name}"


KIND = _key("kind")
"""On a message: ``delegation`` when an agent is asking, absent for a person's turn.
On a part: ``chat_ctx`` for the chat history, ``chat_item`` for one typed item."""

VERBATIM = _key("verbatim")
"""On a status message or an artifact: say the text as written rather than phrasing it."""

DIRECTIVE = _key("directive")
"""On a terminal ``COMPLETED`` event: what the caller does after saying the text."""

REASON = _key("reason")
"""On a ``CancelTaskRequest``: why the caller is stopping the task."""

CONVERSATION = _key("conversation")
"""On a message: the caller's conversation id, whose database the expert persists into."""

CALLER = _key("caller")
"""On a message: the caller's session id, which the expert's session names as its parent."""

KIND_DELEGATION = "delegation"
KIND_CLOSE = "close"
"""On a message: the context is over, and the server may drop it now rather than wait
for it to go idle."""
KIND_CHAT_CTX = "chat_ctx"
KIND_CHAT_ITEM = "chat_item"

ANSWER_ARTIFACT_NAME = "answer"
"""The one artifact that is the answer. A caller reads it whole at the terminal status."""


def struct(value: dict[str, Any]) -> struct_pb2.Struct:
    """A metadata field: A2A types them ``Struct``."""
    return json_format.ParseDict(value, struct_pb2.Struct())


def value(data: dict[str, Any]) -> struct_pb2.Value:
    """A data part's payload: A2A types it ``Value``, not ``Struct``."""
    return json_format.ParseDict(data, struct_pb2.Value())


def as_dict(field: Any) -> dict[str, Any]:
    """Whichever of the two, read back as JSON."""
    return dict(json_format.MessageToDict(field))


def text_of(parts: Any) -> str:
    return "\n".join(part.text for part in parts if part.WhichOneof("content") == "text")


def agent_card(
    *,
    url: str,
    name: str,
    description: str,
    version: str = "1.0.0",
) -> pb.AgentCard:
    """The card an endpoint serves: one ``delegate`` skill, text in and text out.

    The extension is offered and never required, so a client that ignores it still works.
    """
    return pb.AgentCard(
        name=name,
        description=description,
        version=version,
        capabilities=pb.AgentCapabilities(
            streaming=True,
            extensions=[
                pb.AgentExtension(
                    uri=EXTENSION_URI, description=EXTENSION_DESCRIPTION, required=False
                )
            ],
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            pb.AgentSkill(
                id="delegate", name="delegate", description=description, tags=["delegation"]
            )
        ],
        supported_interfaces=[
            pb.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.HTTP_JSON,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
    )


def offers_extension(card: pb.AgentCard) -> bool:
    """Whether a card offers this profile, so a client knows to ask for it."""
    from a2a.extensions.common import find_extension_by_uri

    if find_extension_by_uri(card, EXTENSION_URI) is None:
        logger.debug(
            "the agent card does not offer the agent session extension, falling back to plain A2A",
            extra={"card": card.name},
        )
        return False
    return True
