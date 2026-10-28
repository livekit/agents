"""What crosses ``_core``'s storage boundary (plan 5.2, 13.8), as a developer reads it.

The errors are ``errors.py``'s classes; the rows, records and the turn's context are the generated
models of ``_generated/storage*.py``. This module holds the two things that are JSON on one side
and Python on the other: a chat item (``ChatItem`` against its protobuf JSON) and a lane's
JSON data held in the state column. Application fields are loaded and saved explicitly.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from typing import Any, TypeVar

from google.protobuf import json_format
from pydantic import BaseModel, JsonValue, TypeAdapter

from livekit.protocol.agent_pb import agent_session as agent_pb

from .._generated.storage_class import ItemRow
from .._proto import decode_chat_item, encode_chat_item
from ..llm import ChatContext, ChatItem
from .errors import (
    NotSerializable,
    RunRejected,
    StorageError,
    StorageUnavailable,
    ThreadNotFound,
    TooLarge,
    VersionConflict,
)

S = TypeVar("S")

TERMINAL_TASK_STATES: frozenset[str] = frozenset({"completed", "failed", "canceled", "rejected"})
"""Plan 5.5: the task states nothing moves out of."""

STATE_LIMIT = 64 * 1024
"""Plan 5.3 rule 6: state under 64 KiB."""


def item_to_payload(item: ChatItem) -> dict[str, Any]:
    """A chat item as ``_core`` holds it: ``livekit.agent.ChatContext.ChatItem`` protobuf JSON."""
    return json_format.MessageToDict(encode_chat_item(item))


def payload_to_item(payload: dict[str, Any]) -> ChatItem:
    return decode_chat_item(json_format.ParseDict(payload, agent_pb.ChatContext.ChatItem()))


def history_of(rows: Iterable[ItemRow]) -> ChatContext:
    """The stored items as the chat context the model reads."""
    return ChatContext([payload_to_item(row.payload) for row in rows])


def dump_state(state: S, state_type: type[S]) -> JsonValue:
    """The state as JSON; ``NotSerializable`` names the field, ``TooLarge`` the limit."""
    try:
        data: JsonValue = TypeAdapter(state_type).dump_python(state, mode="json")
    except Exception as e:
        field = _offending_field(state)
        raise NotSerializable(
            f"field `{field}` is not serializable: {e}; state is JSON-shaped, services belong "
            "in unmarked runtime context fields"
        ) from e

    if len(json.dumps(data, separators=(",", ":")).encode()) > STATE_LIMIT:
        raise TooLarge(f"state exceeds the limit of {STATE_LIMIT} bytes")
    return data


def load_state(stored: JsonValue, state_type: type[S]) -> S:
    """The stored snapshot as ``state_type``; default-constructed for a fresh lane."""
    if stored is None:
        return state_type()
    return TypeAdapter(state_type).validate_python(stored)


def _offending_field(state: object) -> str:
    """The first field of a dataclass, model or dict that does not dump as JSON."""
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        values = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
    elif isinstance(state, BaseModel):
        values = {name: getattr(state, name) for name in type(state).model_fields}
    elif isinstance(state, dict):
        values = {str(k): v for k, v in state.items()}
    else:
        return "state"

    probe: TypeAdapter[object] = TypeAdapter(object)
    for name, value in values.items():
        try:
            probe.dump_python(value, mode="json")
        except Exception:
            return name
    return "state"


__all__ = [
    "STATE_LIMIT",
    "TERMINAL_TASK_STATES",
    "NotSerializable",
    "RunRejected",
    "ThreadNotFound",
    "StorageError",
    "StorageUnavailable",
    "TooLarge",
    "VersionConflict",
    "dump_state",
    "history_of",
    "item_to_payload",
    "payload_to_item",
    "load_state",
]
