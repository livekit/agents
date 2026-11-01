# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] declarations in lk-agents-abi.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .. import _core
from .usage import Usage


class SessionCommands:
    if TYPE_CHECKING:

        @property
        def _owner(self) -> _core._core_SessionOwner: ...

        @property
        def _running(self) -> _core.VoiceSession: ...

    async def set_participant(self, identity: str | None) -> None:
        """`RoomIO.set_participant(identity)`: point the audio input and the user transcript at
        someone else. `None` unlinks.
        """
        return await self._running.set_participant(identity)

    def clear_user_turn(self) -> None:
        """`AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it."""
        return self._running.clear_user_turn()

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        """`AgentSession.interrupt(force=)`: the current and queued speeches up to the first
        uninterruptible; answers when they are done, or with the refusal.
        """
        return asyncio.ensure_future(self._running.interrupt(force=force))

    async def drain(self) -> None:
        """`AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
        speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
        """
        await self._running.drain()

    async def wait_for_idle(self) -> None:
        """`AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`)."""
        return await self._running.wait_for_idle()

    def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> asyncio.Future[str]:
        """`AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
        transcript. Runs whether or not awaited; a newer commit supersedes it.
        """
        return asyncio.ensure_future(
            self._running.commit_user_turn(
                transcript_timeout=transcript_timeout,
                stt_flush_duration=stt_flush_duration,
                skip_reply=skip_reply,
            )
        )

    @property
    def usage(self) -> Usage:
        """Empty before startup; the final snapshot remains readable after close."""
        return self._owner.usage

    @property
    def keyterms(self) -> list[str]:
        """No keyterms are applied before a voice driver exists."""
        return self._owner.keyterms
