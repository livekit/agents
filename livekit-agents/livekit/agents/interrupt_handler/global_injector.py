import asyncio
import logging
from .pipeline import InterruptionPipeline


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("InterruptionPipeline").setLevel(logging.DEBUG)
logging.getLogger("GlobalInterruptionInjector").setLevel(logging.DEBUG)
logger = logging.getLogger("GlobalInterruptionInjector")


class GlobalInterruptionInjector:
    """
    Injects the universal interruption pipeline into ANY LiveKit agent session.
    This attaches:
     - TTS start/end listeners
     - Unified transcription handler
     - Emits clean 'user_input' events instead of raw ASR
    """

    def __init__(self):
        self.pipeline = InterruptionPipeline()

    def attach(self, session):
        @session.on("tts_started")   # NOT tts-started
        def _(ev):
            self.pipeline.set_agent_speaking(True)

        @session.on("tts_ended")     # NOT tts-finished
        def _(ev):
            self.pipeline.set_agent_speaking(False)

        # Intercept all ASR events
        @session.on("transcription")
        def _(ev):
            asyncio.create_task(self._handle_transcription(session, ev))

        logger.info("[GLOBAL] Interruption injector attached to session.")

    async def _handle_transcription(self, session, ev):
        text = ev.text or ""
        confidence = ev.confidence or 0.0

        result = self.pipeline.process(text, confidence)

        if result == "ignore":
            return

        if result == "interrupt":
            await session.interrupt_tts()
            return

        if result == "speech":
            session.emit("user_input", text)
    def attach_to_next_session(self, agent):
        """
        Called inside Agent.__init__().
        Agent has no session yet, so we patch _get_activity_or_raise().
        When the session appears later, we automatically attach.
        """

        original_get_activity = agent._get_activity_or_raise

        def wrapped_get_activity():
            activity = original_get_activity()
            session = activity.session

            # Only attach once per session
            if not getattr(session, "_interrupt_injected", False):
                self.attach(session)
                session._interrupt_injected = True

            return activity

        # Monkey-patch Agent._get_activity_or_raise
        agent._get_activity_or_raise = wrapped_get_activity