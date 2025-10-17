from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents.voice.ivr.loop_detector import TfidfLoopDetector

from ...log import logger
from .silence_detector import SilenceDetector

if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_session import AgentSession

DEFAULT_SILENCE_REMINDER_MESSAGE = (
    "<system_notification>"
    "Now that the phone call is become silent for a while, say something or perform some action to proceed with the call. "
    "</system_notification>"
)

DEFAULT_LOOP_DETECTED_MESSAGE = (
    "<system_notification>"
    "Speech loop has been detected from the automated IVR system, say something or perform some action to proceed with the call. "
    "</system_notification>"
)


class IVRActivity:
    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        self._session = session
        self._silence_detector = SilenceDetector(session, max_silence_duration=max_silence_duration)
        self._loop_detector = TfidfLoopDetector(session)

    async def start(self) -> None:
        self._silence_detector.on("silence_detected", self._on_silence_detected)
        self._loop_detector.on("loop_detected", self._on_loop_detected)

        await self._silence_detector.start()
        await self._loop_detector.start()

    async def update_agent(self, agent: Agent) -> None:
        from ...beta.tools.send_dtmf import send_dtmf_events

        agent._tools.append(send_dtmf_events)

    def _on_silence_detected(self, _) -> None:
        logger.info("IVRActivity: silence detected; sending notification")
        self._session.generate_reply(user_input=DEFAULT_SILENCE_REMINDER_MESSAGE)

    def _on_loop_detected(self, _) -> None:
        logger.info("IVRActivity: speech loop detected; sending notification")
        self._loop_detector.reset()
        self._session.generate_reply(
            user_input=DEFAULT_LOOP_DETECTED_MESSAGE, allow_interruptions=False
        )

    async def aclose(self) -> None:
        self._silence_detector.off("silence_detected", self._on_silence_detected)
        self._loop_detector.off("loop_detected", self._on_loop_detected)

        await self._silence_detector.aclose()
        await self._loop_detector.aclose()
