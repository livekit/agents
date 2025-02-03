from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from livekit.agents import llm

from .log import logger


@dataclass
class _ConversationItem:
    """A node in the conversation linked list"""

    message: llm.ChatMessage
    _prev: Optional[_ConversationItem] = field(default=None, repr=False)
    _next: Optional[_ConversationItem] = field(default=None, repr=False)


class _RemoteConversationItems:
    """Manages conversation items in a doubly-linked list"""

    def __init__(self) -> None:
        self._head: Optional[_ConversationItem] = None
        self._tail: Optional[_ConversationItem] = None
        self._id_to_item: OrderedDict[str, _ConversationItem] = OrderedDict()

    @classmethod
    def from_chat_context(cls, chat_ctx: llm.ChatContext) -> _RemoteConversationItems:
        """Create ConversationItems from a ChatContext"""
        items = cls()
        for msg in chat_ctx.messages:
            items.append(msg)
        return items

    def to_chat_context(self) -> llm.ChatContext:
        """Export to a ChatContext"""
        chat_ctx = llm.ChatContext()
        current = self._head
        while current:
            chat_ctx.messages.append(current.message.copy())
            current = current._next
        return chat_ctx

    def append(self, message: llm.ChatMessage) -> None:
        """Add a message to the end of the conversation"""
        if message.id is None:
            raise ValueError("Message must have an id")

        if message.id in self._id_to_item:
            raise ValueError(f"Message with id {message.id} already exists")

        item = _ConversationItem(message=message)
        item._prev = self._tail
        item._next = None

        if self._tail:
            self._tail._next = item
        self._tail = item

        if not self._head:
            self._head = item

        self._id_to_item[message.id] = item

    def insert_after(self, prev_item_id: str | None, message: llm.ChatMessage) -> None:
        """Insert a message after the specified message ID.
        If prev_item_id is None, append to the end."""
        if message.id is None:
            raise ValueError("Message must have an id")

        if message.id in self._id_to_item:
            raise ValueError(f"Message with id {message.id} already exists")

        if prev_item_id is None:
            # Append to end instead of inserting at head
            self.append(message)
            return

        prev_item = self._id_to_item.get(prev_item_id)
        if not prev_item:
            logger.error(
                f"Previous message with id {prev_item_id} not found, ignore it"
            )
            return

        new_item = _ConversationItem(message=message)
        new_item._prev = prev_item
        new_item._next = prev_item._next
        prev_item._next = new_item
        if new_item._next:
            new_item._next._prev = new_item
        else:
            self._tail = new_item

        self._id_to_item[message.id] = new_item

    def delete(self, item_id: str) -> None:
        """Delete a message by its ID"""
        item = self._id_to_item.get(item_id)
        if not item:
            logger.error(f"Message with id {item_id} not found for deletion")
            return

        if item._prev:
            item._prev._next = item._next
        else:
            self._head = item._next

        if item._next:
            item._next._prev = item._prev
        else:
            self._tail = item._prev

        del self._id_to_item[item_id]

    def get(self, item_id: str) -> llm.ChatMessage | None:
        """Get a message by its ID"""
        item = self._id_to_item.get(item_id)
        return item.message if item else None

    @property
    def messages(self) -> list[llm.ChatMessage]:
        """Return all messages in order"""
        return [item.message for item in self._id_to_item.values()]
