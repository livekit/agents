# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] declarations in lk-agents-abi.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .. import _core
from .chat_items import ChatItem


class SpeechView:
    if TYPE_CHECKING:

        @property
        def _native(self) -> _core.SpeechHandle: ...

    @property
    def id(self) -> str:
        """`speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
        the id it was constructed with.
        """
        return self._native.id

    @property
    def num_steps(self) -> int:
        return self._native.num_steps

    @property
    def chat_items(self) -> list[ChatItem]:
        """What this speech committed to the conversation; a copy."""
        return self._native.chat_items

    @property
    def scheduled(self) -> bool:
        return self._native.scheduled

    @property
    def interrupted(self) -> bool:
        """`True` from the tick `interrupt()` was called until the speech is done, and once the loop
        has interrupted the speech for any reason (an interruption, a supersede, the close). A done
        speech reads the loop's verdict.
        """
        return self._native.interrupted

    def done(self) -> bool:
        return self._native.done()
