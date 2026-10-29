"""What crosses ``_core``'s storage boundary (plan 5.2, 13.8), as a developer reads it.

The errors are ``errors.py``'s classes; the rows, records and the turn's context are the generated
models of ``_generated/storage*.py``. This module holds what is JSON on one side and Python on
the other: a chat item (``ChatItem`` against its protobuf JSON). Application fields are loaded and
saved explicitly; storage checks their size.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from google.protobuf import json_format

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

TERMINAL_TASK_STATES: frozenset[str] = frozenset({"completed", "failed", "canceled", "rejected"})
"""Plan 5.5: the task states nothing moves out of."""


def item_to_payload(item: ChatItem) -> dict[str, Any]:
    """A chat item as ``_core`` holds it: ``livekit.agent.ChatContext.ChatItem`` protobuf JSON."""
    return json_format.MessageToDict(encode_chat_item(item))


def payload_to_item(payload: dict[str, Any]) -> ChatItem:
    return decode_chat_item(json_format.ParseDict(payload, agent_pb.ChatContext.ChatItem()))


def history_of(rows: Iterable[ItemRow]) -> ChatContext:
    """The stored items as the chat context the model reads."""
    return ChatContext([payload_to_item(row.payload) for row in rows])


__all__ = [
    "TERMINAL_TASK_STATES",
    "NotSerializable",
    "RunRejected",
    "ThreadNotFound",
    "StorageError",
    "StorageUnavailable",
    "TooLarge",
    "VersionConflict",
    "history_of",
    "item_to_payload",
    "payload_to_item",
]
