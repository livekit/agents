from __future__ import annotations

import asyncio
import contextlib

from .. import utils


class SpeechHandle:
    def __init__(
        self, *, speech_id: str, allow_interruptions: bool, step_index: int
    ) -> None:
        self._id = speech_id
        self._step_index = step_index
        self._allow_interruptions = allow_interruptions
        self._interrupt_fut = asyncio.Future()
        self._done_fut = asyncio.Future()
        self._play_fut = asyncio.Future()
        self._playout_done_fut = asyncio.Future()

    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0) -> SpeechHandle:
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    def _authorize_playout(self) -> None:
        self._play_fut.set_result(None)

    def done(self) -> bool:
        return self._done_fut.done()

    def interrupt(self) -> None:
        if not self._allow_interruptions:
            raise ValueError("This generation handle does not allow interruptions")

        if self.done():
            return

        self._done_fut.set_result(None)
        self._interrupt_fut.set_result(None)

    async def wait_for_playout(self) -> None:
        await asyncio.shield(self._playout_done_fut)

    def _mark_playout_done(self) -> None:
        self._playout_done_fut.set_result(None)

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            # will raise InvalidStateError if the future is already done (interrupted)
            self._done_fut.set_result(None)
