from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from .. import utils
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .events import AgentStateChangedEvent, UserStateChangedEvent


_FillerSource = str | Callable[[int], SpeechHandle | str | None]


class _FillerScheduler:
    """Background task that fires filler speech during a long-running step."""

    def __init__(
        self,
        source: _FillerSource,
        *,
        session: AgentSession,
        speech_handle: SpeechHandle,
        delay: float,
        interval: float | None,
        max_steps: int | None,
    ) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if interval is not None and interval < 0:
            raise ValueError("interval must be non-negative when set")

        self._session = session
        self._speech_handle = speech_handle
        self._source = source
        self._delay = delay
        self._interval = interval
        self._max_steps = max_steps

        self._speaking_ev = asyncio.Event()
        self._main_task = asyncio.create_task(self._run(), name="_FillerScheduler._run")
        self._created_speeches: list[SpeechHandle] = []

    async def aclose(self) -> None:
        if not self._main_task.done():
            await utils.aio.cancel_and_wait(self._main_task)

    def reset_dwell(self) -> None:
        """Abort the current idle dwell — the next iteration restarts from wait_for_idle.

        Called by ``ctx.update()`` to signal that the tool just took the floor, so any
        pending filler should hold off until idle resumes for a fresh ``delay`` window.
        """
        self._speaking_ev.set()

    async def _run(self) -> None:
        def _on_agent(ev: AgentStateChangedEvent) -> None:
            if ev.new_state in ("speaking", "thinking"):
                self._speaking_ev.set()

        def _on_user(ev: UserStateChangedEvent) -> None:
            if ev.new_state == "speaking":
                self._speaking_ev.set()

        self._session.on("agent_state_changed", _on_agent)
        self._session.on("user_state_changed", _on_user)

        async def _loop() -> None:
            while True:
                await self._session.wait_for_idle()
                self._speaking_ev.clear()
                try:
                    await asyncio.wait_for(self._speaking_ev.wait(), timeout=self._delay)
                    continue  # reset dwell delay
                except asyncio.TimeoutError:
                    pass

                # no await allowed below
                src = self._source
                if callable(src):
                    step = len(self._created_speeches)
                    handle = src(step)
                    if isinstance(handle, str):
                        handle = self._session.say(handle)
                else:
                    handle = self._session.say(src)

                if handle is not None:
                    self._created_speeches.append(handle)

                if self._interval is None or (
                    self._max_steps is not None and len(self._created_speeches) >= self._max_steps
                ):
                    break

                await asyncio.sleep(self._interval)

        loop_task = asyncio.create_task(_loop(), name="_FillerScheduler._loop")
        try:
            await self._speech_handle.wait_if_not_interrupted([loop_task])
        finally:
            if not loop_task.done():
                await utils.aio.cancel_and_wait(loop_task)
            self._session.off("agent_state_changed", _on_agent)
            self._session.off("user_state_changed", _on_user)

    async def _await_impl(self) -> list[SpeechHandle]:
        await self._main_task
        return self._created_speeches

    def __await__(self) -> Generator[None, None, list[SpeechHandle]]:
        return self._await_impl().__await__()
