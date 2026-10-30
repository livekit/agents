"""Host SDK adapters for LiveKit A2A metadata and cards.

The keys are generated from the native schema. The optional extension header advertises peer
capability; it grants no authority and is not required to use the documented metadata fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import TransportProtocol
from google.protobuf import json_format, struct_pb2
from google.protobuf.message import Message as ProtoMessage
from pydantic import JsonValue

from .._core import _core_a2a_agent_card, _core_a2a_extensions_header
from .._generated.a2a_card import EndpointDescriptor
from .._generated.a2a_metadata import (
    EXTENSION_URI,
    EXTENSIONS_HEADER,
)

JSONRPC_BINDING = TransportProtocol.JSONRPC.value
Metadata = dict[str, JsonValue]


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
        return cls(livekit=_core_a2a_extensions_header(value) is not None)

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


def agent_card(
    *,
    url: str,
    name: str,
    description: str,
    version: str = "1.0.0",
) -> pb.AgentCard:
    """Adapt the native card to the Python A2A SDK's protobuf representation."""
    descriptor = EndpointDescriptor(name=name, description=description, version=version, url=url)
    return json_format.Parse(_core_a2a_agent_card(descriptor), pb.AgentCard())
