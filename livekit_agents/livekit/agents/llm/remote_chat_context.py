from __future__ import annotations

from dataclasses import dataclass, field

from .chat_context import ChatContext, ChatItem

__all__ = ["RemoteChatContext"]


@dataclass
class _RemoteChatItem:
    item: ChatItem
    _prev: _RemoteChatItem | None = field(default=None, repr=False)
    _next: _RemoteChatItem | None = field(default=None, repr=False)


class RemoteChatContext:
    def __init__(self) -> None:
        self._head: _RemoteChatItem | None = None
        self._tail: _RemoteChatItem | None = None
        self._id_to_item: dict[str, _RemoteChatItem] = {}

    def to_chat_ctx(self) -> ChatContext:
        items: list[ChatItem] = []
        current_node = self._head
        while current_node is not None:
            items.append(current_node.item)
            current_node = current_node._next

        return ChatContext(items=items)

    def get(self, item_id: str) -> _RemoteChatItem | None:
        return self._id_to_item.get(item_id)

    def insert(self, previous_item_id: str | None, message: ChatItem) -> None:
        """
        Insert `message` after the node with ID `previous_item_id`.
        If `previous_item_id` is None, insert at the head.
        """
        item_id = message.id

        if item_id in self._id_to_item:
            raise ValueError(f"Item with ID {item_id} already exists.")

        new_node = _RemoteChatItem(item=message)

        if previous_item_id is None:
            if self._head is not None:
                new_node._next = self._head
                self._head._prev = new_node
            else:
                self._tail = new_node

            self._head = new_node
            self._id_to_item[item_id] = new_node
            return

        prev_node = self._id_to_item.get(previous_item_id)
        if prev_node is None:
            raise ValueError(f"previous_item_id `{previous_item_id}` not found")

        new_node._prev = prev_node
        new_node._next = prev_node._next

        prev_node._next = new_node

        if new_node._next is not None:
            new_node._next._prev = new_node
        else:
            self._tail = new_node

        self._id_to_item[item_id] = new_node

    def delete(self, item_id: str) -> None:
        node = self._id_to_item.get(item_id)
        if node is None:
            raise ValueError(f"item_id `{item_id}` not found")

        prev_node = node._prev
        next_node = node._next

        if self._head == node:
            self._head = next_node
            if self._head is not None:
                self._head._prev = None
        else:
            if prev_node is not None:
                prev_node._next = next_node

        if self._tail == node:
            self._tail = prev_node
            if self._tail is not None:
                self._tail._next = None
        else:
            if next_node is not None:
                next_node._prev = prev_node

        del self._id_to_item[item_id]
