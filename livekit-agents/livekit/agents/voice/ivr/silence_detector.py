from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from livekit.rtc import EventEmitter

from ...log import logger
from ...utils.aio.debounce import Debounced

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..events import AgentStateChangedEvent, UserStateChangedEvent

EventTypes = Literal["silence_detected"]


class SilenceDetector(EventEmitter[EventTypes]):
    """Silence detector.

    This detector checks for silence in the user / agent interaction.

    Args:
        session: The agent session.
        max_silence_duration: The maximum duration of silence to detect in seconds. Default ``5.0`` seconds.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        super().__init__()
        self._session = session
        self._max_silence_duration = max_silence_duration
        self._current_user_state: Optional[str] = None  # noqa: UP007
        self._current_agent_state: Optional[str] = None  # noqa: UP007
        self._debounced_emit = Debounced(self._emit_silence_detected, self._max_silence_duration)

    async def start(self) -> None:
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("agent_state_changed", self._on_agent_state_changed)

    async def aclose(self) -> None:
        self._debounced_emit.cancel()
        self._session.off("user_state_changed", self._on_user_state_changed)
        self._session.off("agent_state_changed", self._on_agent_state_changed)

    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        self._current_user_state = ev.new_state
        self._schedule_check()

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        self._current_agent_state = ev.new_state
        self._schedule_check()

    def _schedule_check(self) -> None:
        if self._current_user_state == self._current_agent_state == "listening":
            logger.info(
                "SilenceDetector: user_state=%s, agent_state=%s, scheduling silence check",
                self._current_user_state,
                self._current_agent_state,
            )
            self._debounced_emit.schedule()
        else:
            logger.info(
                "SilenceDetector: user_state=%s, agent_state=%s, canceling silence check",
                self._current_user_state,
                self._current_agent_state,
            )
            self._debounced_emit.cancel()

    async def _emit_silence_detected(self) -> None:
        logger.info("SilenceDetector: emitting silence_detected event")
        self.emit("silence_detected", None)
