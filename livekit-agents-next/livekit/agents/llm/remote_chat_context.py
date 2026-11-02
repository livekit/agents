from __future__ import annotations

from typing import cast

from .. import _core
from .._generated.remote_chat_context import _RemoteChatItem
from .chat_context import ChatContext, ChatItem

__all__ = ["RemoteChatContext"]


class RemoteChatContext:
    """Server acknowledgement order, retaining the original mutable Python items."""

    def __init__(self) -> None:
        self._order = _core._core_RemoteChatContext()

    def to_chat_ctx(self) -> ChatContext:
        views = cast(list[_RemoteChatItem], self._order.items())
        return ChatContext(items=[view.item for view in views])

    def get(self, item_id: str) -> _RemoteChatItem | None:
        return cast(_RemoteChatItem | None, self._order.get(item_id))

    @property
    def tail_id(self) -> str | None:
        return self._order.tail_id

    def insert(self, previous_item_id: str | None, message: ChatItem) -> None:
        self._order.insert(previous_item_id, message.id, _RemoteChatItem(message))

    def delete(self, item_id: str) -> None:
        self._order.delete(item_id)
