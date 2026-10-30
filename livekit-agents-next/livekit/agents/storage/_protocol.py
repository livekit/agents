"""A chat item as ``_core``'s storage boundary holds it: ``ChatItem`` against its protobuf
JSON."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from google.protobuf import json_format

from livekit.protocol.agent_pb import agent_session as agent_pb

from .._generated.storage_class import ItemRow
from .._proto import decode_chat_item, encode_chat_item
from ..llm import ChatContext, ChatItem


def item_to_payload(item: ChatItem) -> dict[str, Any]:
    """A chat item as ``_core`` holds it: ``livekit.agent.ChatContext.ChatItem`` protobuf JSON."""
    return json_format.MessageToDict(encode_chat_item(item))


def payload_to_item(payload: dict[str, Any]) -> ChatItem:
    return decode_chat_item(json_format.ParseDict(payload, agent_pb.ChatContext.ChatItem()))


def history_of(rows: Iterable[ItemRow]) -> ChatContext:
    """The stored items as the chat context the model reads."""
    return ChatContext([payload_to_item(row.payload) for row in rows])
