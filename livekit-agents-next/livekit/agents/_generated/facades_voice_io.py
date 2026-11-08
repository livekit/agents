# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from .. import _bindings, _core

if TYPE_CHECKING:
    from .chat_items import ChatItem


class SpeechHandle:
    """`voice.SpeechHandle`'s reads of its native speech."""

    if TYPE_CHECKING:
        _native: _core.SpeechHandle

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

    @property
    def allow_interruptions(self) -> bool:
        return self._native.allow_interruptions

    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        self._native.allow_interruptions = value

    def exception(self) -> BaseException | None:
        """The error the speech failed with, or `None`; raises `asyncio.InvalidStateError` before it
        is done.
        """
        return self._native.exception()

    def interrupt(self, *, force: bool = False) -> Self:
        """Interrupt the speech."""
        self._native.interrupt(force=force)
        return self

    async def _wait_for_generation(self, step: int | None = None) -> None:
        """One step of the turn, not the whole thing; `None` is the latest."""
        await self._native.wait_for_generation(step)
