import logging
import asyncio
from agent.state import AgentState
from interrupt_handler.middleware import InterruptFilteringMiddleware

logger = logging.getLogger("session_manager")

class SessionManager:
    def __init__(self, session, agent, config):
        self.session = session
        self.agent = agent
        self.config = config
        self.state = AgentState()

        # semantic interruption filter
        self.interrupt_filter = InterruptFilteringMiddleware(
            conf_threshold=float(getattr(config, "confidence_threshold", 0.6))
        )

    async def initialize(self, room):

        # LiveKit 1.3.x emits "conversation_message_added"
        @self.session.on("conversation_message_added")
        def _on_message(ev):
            asyncio.create_task(self._handle_message(ev))

        await self.session.start(agent=self.agent, room=room)
        logger.info("[SESSION] Started LiveKit session.")

    # ----------------------------------------------------
    # MAIN HANDLER — LiveKit 1.3.x event model
    # ----------------------------------------------------
    async def _handle_message(self, ev):
        msg = ev.message  # correct object in LiveKit 1.3.x

        # ignore system + assistant messages
        if msg.role != "user":
            return

        # get transcript text
        text = (msg.text or "").strip()

        # confidence exists only for transcripts
        confidence = getattr(msg, "confidence", 1.0)

        logger.info(f"[USER TRANSCRIPT] '{text}' (conf={confidence:.2f})")

        await self.handle_user_transcript(text, confidence)

    # ----------------------------------------------------
    # SEMANTIC INTERRUPTION LAYER
    # ----------------------------------------------------
    async def handle_user_transcript(self, text, confidence):
        decision = await self.interrupt_filter.should_interrupt(
            text=text,
            confidence=confidence,
            agent_is_speaking=self.state.agent_is_speaking,
        )

        if decision == "command" and self.state.agent_is_speaking:
            logger.info("[INTERRUPT] Command detected → stopping agent.")
            await self.session.interrupt()
            return

        if decision == "filler_ignored" and self.state.agent_is_speaking:
            logger.info("[FILLER IGNORED] Not interrupting.")
            return

        # Normal speech → end user turn (allows agent to respond)
        logger.info("[END USER TURN] Normal speech.")
        await self.session.end_user_turn()

    # ----------------------------------------------------
    # AGENT TTS GENERATION
    # ----------------------------------------------------
    async def say(self, text):
        logger.info(f"[AGENT] Speaking: {text}")

        self.state.agent_is_speaking = True

        handle = await self.session.generate_reply(
            instructions=text,
            allow_interruptions=False,
        )

        await handle._wait_for_generation()
        self.state.agent_is_speaking = False
