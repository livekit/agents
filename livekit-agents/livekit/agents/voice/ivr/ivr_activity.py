from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...log import logger
from ...utils.aio.debounce import Debounced

if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_session import AgentSession
    from ..events import (
        AgentState,
        AgentStateChangedEvent,
        UserInputTranscribedEvent,
        UserState,
        UserStateChangedEvent,
    )


DEFAULT_SILENCE_REMINDER_MESSAGE = (
    "<system_reminder>"
    "Now that the phone call is become silent for a while, say something or perform some action to proceed with the call. "
    "</system_reminder>"
)


class IVRActivity:
    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        self._session = session
        self._max_silence_duration = max_silence_duration
        self._current_user_state: Optional[UserState] = None  # noqa: UP007
        self._current_agent_state: Optional[AgentState] = None  # noqa: UP007
        self._send_silence_reminder_debounced = Debounced(
            self._send_silence_reminder, self._max_silence_duration
        )

    async def start(self) -> None:
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)

    async def update_agent(self, agent: Agent) -> None:
        from ...beta.tools.send_dtmf import send_dtmf_events

        agent._tools.append(send_dtmf_events)

    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        self._current_user_state = ev.new_state
        self._schedule_silence_check()

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        self._current_agent_state = ev.new_state
        self._schedule_silence_check()

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        pass

    def _schedule_silence_check(self) -> None:
        if self._current_agent_state == self._current_user_state == "listening":
            logger.info("Both agent and user are listening, scheduling silence check")
            self._send_silence_reminder_debounced.schedule()
        else:
            logger.info("Either agent or user is not listening, canceling silence check")
            self._send_silence_reminder_debounced.cancel()

    async def _send_silence_reminder(self) -> None:
        logger.info("Sending silence reminder")
        self._session.generate_reply(user_input=DEFAULT_SILENCE_REMINDER_MESSAGE)

    async def aclose(self) -> None:
        self._send_silence_reminder_debounced.cancel()
        self._session.off("user_state_changed", self._on_user_state_changed)
        self._session.off("agent_state_changed", self._on_agent_state_changed)
