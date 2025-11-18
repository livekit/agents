import logging
import asyncio
from livekit.agents import ConversationItemAddedEvent
from agent.state import AgentState
from interrupt_handler.middleware import InterruptFilteringMiddleware

logger = logging.getLogger("session_manager")


class SessionManager:
    def __init__(self, session, agent, config):
        self.session = session
        self.agent = agent
        self.config = config
        self.state = AgentState()

        self.interrupt_filter = InterruptFilteringMiddleware(
            conf_threshold=float(
                getattr(config, "confidence_threshold", 0.6)
            )
        )

    async def initialize(self, room):
        """
        Attach event listener for conversation items (v1.3).
        """

        @self.session.on("conversation_item_added")
        def _on_conversation_item_added(event: ConversationItemAddedEvent):
            # MUST use create_task — LiveKit does NOT allow async callbacks
            asyncio.create_task(self._handle_conversation_message(event))

        await self.session.start(agent=self.agent, room=room)
        logger.info("[SESSION] Started LiveKit session.")

    async def _handle_conversation_message(self, event):
        """
        Minimal replacement for transcript handler.
        """

        # Correct participant access
        participant = event.item.participant

        # Ignore assistant messages
        if participant.identity == "assistant":
            return

        text = event.item.text_content or ""
        confidence = 1.0  # LiveKit doesn't expose ASR confidence here

        logger.info(
            f"[USER TRANSCRIPT] '{text}' (conf={confidence:.2f})"
        )

        await self.handle_user_transcript(text, confidence)

    async def handle_user_transcript(self, text, confidence):
        """
        Your existing interruption logic.
        """

        should_interrupt = await self.interrupt_filter.should_interrupt(
            text=text,
            confidence=confidence,
            agent_is_speaking=self.state.agent_is_speaking,
        )

        if should_interrupt and self.state.agent_is_speaking:
            logger.info("[INTERRUPT] Approved.")
            await self.session.interrupt()
        else:
            logger.info("[NO INTERRUPT] Rejected.")

    async def say(self, text):
        logger.info(f"[AGENT] Speaking: {text}")
        self.state.agent_is_speaking = True

        handle = await self.session.generate_reply(
            instructions=text,
            allow_interruptions=False,
        )

        await handle.wait_for_completion()
        self.state.agent_is_speaking = False
