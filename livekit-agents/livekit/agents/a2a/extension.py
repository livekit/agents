"""The LiveKit agents extension to A2A.

A2A extensions are named by a URI, declared on the agent card, activated by a client with the
``A2A-Extensions`` header, echoed by the server, and keep their fields in the ``metadata`` maps
under keys prefixed by the URI. This module holds the URI, every key written short
(``livekit/<name>``), the per-request negotiation, the protobuf helpers and the agent card an
endpoint publishes. The specification is ``docs/a2a-live-ext.md``. A stock client that sends
no header gets plain A2A, and a stock server that offers no extension is read as plain A2A.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from google.protobuf import json_format, struct_pb2
from google.protobuf.message import Message as ProtoMessage
from pydantic import JsonValue

EXTENSION_URI = "https://livekit.io/a2a/ext/agents/v1"
"""The Live extension. A key on the wire is ``f"{EXTENSION_URI}/{name}"``."""

EXTENSIONS_HEADER = "A2A-Extensions"
JSONRPC_BINDING = TransportProtocol.JSONRPC.value
"""``JSONRPC``: JSON-RPC 2.0 over HTTP, server-sent events for the streams; mandatory."""

LIVEKIT_PREFIX = "livekit/"

Metadata = dict[str, JsonValue]
"""A ``metadata`` map as this package reads and writes it: JSON under short keys."""


def shorten(wire_key: str) -> str:
    """A wire key to its short form; anything else unchanged."""
    if wire_key.startswith(EXTENSION_URI + "/"):
        return LIVEKIT_PREFIX + wire_key[len(EXTENSION_URI) + 1 :]
    return wire_key


def widen(short_key: str) -> str:
    """A short key to the wire; anything else unchanged."""
    if short_key.startswith(LIVEKIT_PREFIX):
        return f"{EXTENSION_URI}/{short_key[len(LIVEKIT_PREFIX) :]}"
    return short_key


# the Live extension's keys, short
KIND = "livekit/kind"
CALLER_LANE = "livekit/callerLane"
"""On a delegation in the caller's thread: the caller's lane, whose history holds the person's
own words. The expert's handler may read it; the framework gives the expert's model none of it."""
INTERRUPT = "livekit/interrupt"
"""On a message to a ``live`` inbox: ``false`` when it waits for the reply under way instead of
stopping it; absent is ``true``. A delegation sends ``false``."""
INTERRUPTED = "livekit/interrupted"
INTERRUPTION_REASON = "livekit/interruptionReason"
MERGED_INTO = "livekit/mergedInto"
REASON = "livekit/reason"
QUEUED = "livekit/queued"
"""On a ``working`` task whose question waits, unshown, behind the one in front of the person."""
VERBATIM = "livekit/verbatim"
"""``true`` on a text part that reaches the person exactly as written; the agent facing them
words every other part."""

KIND_DELEGATION = "delegation"
KIND_CHAT_CTX = "chat_ctx"

ANSWER_ARTIFACT_NAME = "answer"
"""The one artifact that is the answer; a caller reads it whole at the terminal status."""


@dataclass(frozen=True)
class Extensions:
    """Whether a request activated the extension: read from ``A2A-Extensions``, echoed back.

    A stock client that sends nothing gets ``Extensions()``: plain A2A, text only. The
    server-owned keys (``interrupted``, ``mergedInto``) are written either way, since
    ``metadata`` is core A2A; the typed chat context part follows the extension.
    """

    livekit: bool = False

    @classmethod
    def from_header(cls, value: str | None) -> Extensions:
        if not value:
            return cls()

        requested = {part.strip() for part in value.split(",") if part.strip()}
        return cls(livekit=EXTENSION_URI in requested)

    @classmethod
    def from_headers(cls, headers: Mapping[str, str] | None) -> Extensions:
        """From a Starlette ``Headers`` or any mapping with a case-insensitive ``get``."""
        if headers is None:
            return cls()

        value = headers.get(EXTENSIONS_HEADER)
        if value is None:
            value = headers.get(EXTENSIONS_HEADER.lower())
        return cls.from_header(value)

    def header(self) -> str | None:
        """The echo: the URI when activated, ``None`` when not."""
        return EXTENSION_URI if self.livekit else None


# -- protobuf helpers --


def struct(value: Mapping[str, JsonValue]) -> struct_pb2.Struct:
    """A metadata field: A2A types them ``Struct``."""
    return json_format.ParseDict(dict(value), struct_pb2.Struct())


def value(data: JsonValue) -> struct_pb2.Value:
    """A data part's payload: A2A types it ``Value``."""
    holder = json_format.ParseDict({"data": data}, struct_pb2.Struct())
    return holder.fields["data"]


def as_dict(field: ProtoMessage) -> Metadata:
    """A ``Struct`` or ``Value`` back as JSON; an empty dict for anything that is not a map."""
    out = json_format.MessageToDict(field)
    return out if isinstance(out, dict) else {}


def to_json(message: ProtoMessage) -> dict[str, JsonValue]:
    """A proto message as the JSON the HTTP binding carries (camelCase, enums by name)."""
    return json_format.MessageToDict(message, preserving_proto_field_name=False)


# -- the agent card --

SecuritySchemes = Mapping[str, Mapping[str, JsonValue]]
"""The developer's own schemes, by name, as the JSON of a ``SecurityScheme``."""


def agent_card(
    *,
    url: str,
    name: str,
    description: str,
    version: str = "1.0.0",
    public: bool = True,
    security_schemes: SecuritySchemes | None = None,
) -> pb.AgentCard:
    """The card an endpoint serves: one ``delegate`` skill, text in and text out, the
    extension offered and not required, JSON-RPC over HTTP+SSE as the one interface.
    ``url`` is the HTTP interface, ``<endpoint>/v1``."""
    interface = pb.AgentInterface(
        url=url, protocol_binding=JSONRPC_BINDING, protocol_version=PROTOCOL_VERSION_1_0
    )
    skill = pb.AgentSkill(
        id="delegate", name="delegate", description=description, tags=["delegation"]
    )
    card = pb.AgentCard(
        name=name,
        description=description,
        version=version,
        capabilities=pb.AgentCapabilities(streaming=True, extensions=[_offered_extension()]),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[skill],
        supported_interfaces=[interface],
    )

    _add_security(card, public=public, security_schemes=security_schemes)
    return card


def _offered_extension() -> pb.AgentExtension:
    return pb.AgentExtension(
        uri=EXTENSION_URI,
        description=(
            "LiveKit agents extension: durable sessions as contexts, typed chat items, "
            "interrupt and merge semantics."
        ),
        required=False,
    )


def _add_security(
    card: pb.AgentCard, *, public: bool, security_schemes: SecuritySchemes | None
) -> None:
    """A LiveKit project token is always accepted; the developer's own schemes beside it."""
    note = "" if public else "; required: this endpoint is not public"
    bearer = pb.HTTPAuthSecurityScheme(
        scheme="bearer",
        bearer_format="JWT",
        description="A LiveKit project token or agent session grant" + note,
    )
    card.security_schemes["livekit"].http_auth_security_scheme.CopyFrom(bearer)

    for scheme_name, scheme in (security_schemes or {}).items():
        json_format.ParseDict(dict(scheme), card.security_schemes[scheme_name])

    if not public:
        requirement = card.security_requirements.add()
        requirement.schemes["livekit"].list.extend([])


__all__ = [
    "ANSWER_ARTIFACT_NAME",
    "EXTENSIONS_HEADER",
    "EXTENSION_URI",
    "INTERRUPTED",
    "INTERRUPTION_REASON",
    "JSONRPC_BINDING",
    "KIND",
    "KIND_CHAT_CTX",
    "KIND_DELEGATION",
    "MERGED_INTO",
    "REASON",
    "Extensions",
    "Metadata",
    "SecuritySchemes",
    "agent_card",
    "as_dict",
    "pb",
    "shorten",
    "struct",
    "to_json",
    "value",
    "widen",
]
