from __future__ import annotations

import asyncio
import logging

from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, utils

from .codec import deserialize_chat

logger = logging.getLogger(__name__)

_DEFAULT_BOT_NAME = "LemonSlice Avatar"


def format_chat_user_input(*, sender: str, text: str) -> str:
    return f"[{sender}]: {text}"


class MeetingChatRelay:
    """Queue meeting chat until the agent session is running, then generate replies."""

    def __init__(
        self,
        session: AgentSession,
        *,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._session = session
        if utils.is_given(bot_name) and (name := str(bot_name).strip()):
            self._bot_name = name.lower()
        else:
            self._bot_name = _DEFAULT_BOT_NAME.lower()
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._drain_task: asyncio.Task[None] | None = None

    def submit_json(self, payload: str) -> None:
        try:
            self._loop.call_soon_threadsafe(self._enqueue_json, payload)
        except RuntimeError:
            pass

    def _enqueue_json(self, payload: str) -> None:
        message = deserialize_chat(payload)
        if message is None:
            return
        if message.sender.strip().lower() == self._bot_name:
            return
        user_input = format_chat_user_input(sender=message.sender, text=message.text)
        try:
            self._queue.put_nowait(user_input)
        except asyncio.QueueFull:
            logger.warning("meeting chat relay queue full; dropping message")

    async def _drain(self) -> None:
        while True:
            user_input = await self._queue.get()
            await self._wait_for_session_started()
            logger.info("meeting chat relay: user_input=%r", user_input[:120])
            try:
                await self._session.interrupt()
                self._session.generate_reply(user_input=user_input)
            except Exception:
                logger.warning("meeting chat relay: generate_reply failed", exc_info=True)

    async def _wait_for_session_started(self) -> None:
        while not self._session._started:
            await asyncio.sleep(0.05)

    def start(self) -> None:
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain())

    async def aclose(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            await asyncio.gather(self._drain_task, return_exceptions=True)
            self._drain_task = None
