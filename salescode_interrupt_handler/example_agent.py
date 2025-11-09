
import os

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    llm,
    inference,
)

from .config import InterruptConfig
from .interrupt_handler import FillerAwareInterruptController
from .logging_utils import get_logger

logger = get_logger("example_agent")


class DemoAgent(Agent):
    async def on_enter(self) -> None:
        logger.info("DemoAgent started.")

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        # In this SDK version `text_content` is a property (string), not a function.
        raw_text = new_message.text_content
        user_text = (raw_text or "").strip().lower()

        if not user_text:
            logger.debug(
                "on_user_turn_completed: empty/none user message, nothing to respond."
            )
            return

        logger.info("on_user_turn_completed: user said: %s", user_text)

        reply = f"You said: {user_text}. Let me help you with that."

        # Our custom interrupt controller manages interruptions
        await self.session.generate_reply(
            user_input=reply,
            allow_interruptions=False,
        )


async def entrypoint(ctx: JobContext):
    """
    Entrypoint used by LiveKit worker / CLI.
    """
    await ctx.connect()

    # Use models compatible with your LiveKit inference setup.
    stt_model = os.getenv("LK_STT_MODEL", "assemblyai/universal-streaming:en")
    llm_model = os.getenv("LK_LLM_MODEL", "openai/gpt-4o-mini")
    tts_model = os.getenv("LK_TTS_MODEL", "cartesia/sonic-3")

    session = AgentSession(
        stt=inference.STT.from_model_string(stt_model),
        llm=inference.LLM.from_model_string(llm_model),
        tts=inference.TTS.from_model_string(tts_model),
    )

    agent = DemoAgent(
        instructions=(
            "You are a natural, friendly sales assistant. "
            "Respond clearly, concisely, and leave space for the user to interrupt."
        ),
    )

    # Attach interruption controller
    FillerAwareInterruptController(
        session=session,
        config=InterruptConfig(),
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(),
    )

