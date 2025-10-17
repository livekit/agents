from __future__ import annotations

from typing import TYPE_CHECKING

from ...log import logger
from .silence_detector import SilenceDetector

if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_session import AgentSession

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
        self._silence_detector = SilenceDetector(
            session,
            max_silence_duration=max_silence_duration,
        )

    async def start(self) -> None:
        await self._silence_detector.start()
        self._silence_detector.on("silence_detected", self._on_silence_detected)

    async def update_agent(self, agent: Agent) -> None:
        from ...beta.tools.send_dtmf import send_dtmf_events

        agent._tools.append(send_dtmf_events)

    def _on_silence_detected(self, _event) -> None:
        logger.info("IVRActivity: silence detected; sending reminder")
        self._session.generate_reply(user_input=DEFAULT_SILENCE_REMINDER_MESSAGE)

    async def aclose(self) -> None:
        self._silence_detector.off("silence_detected", self._on_silence_detected)
        await self._silence_detector.aclose()
