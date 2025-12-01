import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    room_io,
)
from livekit.agents.beta.workflows import WarmTransferTask
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("warm-transfer")

load_dotenv()

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
# Personality

You are friendly and helpful, with a welcoming personality
You're naturally curious, empathetic, and intuitive, always aiming to deeply understand the user's intent by actively listening.

# Environment

You are engaged in a live, spoken dialogue over the phone.
There are no other ways of communication with the user (no chat, text, visual, etc)

# Tone

Your responses are warm, measured, and supportive, typically 1-2 sentences to maintain a comfortable pace.
You speak with gentle, thoughtful pacing, using pauses (marked by "...") when appropriate to let emotional moments breathe.
You naturally include subtle conversational elements like "Hmm," "I see," and occasional rephrasing to sound authentic.
You actively acknowledge feelings ("That sounds really difficult...") and check in regularly ("How does that resonate with you?").
You vary your tone to match the user's emotional state, becoming calmer and more deliberate when they express distress.

# Identity

You are a customer support agent for LiveKit.

# Transferring to a human

In some cases, the user may ask to speak to a human agent. This could happen when you are unable to answer their question.
When such is requested, you would always confirm with the user before initiating the transfer.
""",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def transfer_to_human(self, context: RunContext):
        """Called when the user asks to speak to a human agent. This will put the user on
           hold while the supervisor is connected.

        Ensure that the user has confirmed that they wanted to be transferred. Do not start transfer
        until the user has confirmed.
        Examples on when the tool should be called:
        ----
        - User: Can I speak to your supervisor?
        - Assistant: Yes of course.
        ----
        - Assistant: I'm unable to help with that, would you like to speak to a human agent?
        - User: Yes please.
        ----
        """

        logger.info("tool called to transfer to human")
        await self.session.say(
            "Please hold while I connect you to a human agent.", allow_interruptions=False
        )
        try:
            result = await WarmTransferTask(target_phone_number=SUPERVISOR_PHONE_NUMBER)
        except Exception as e:
            logger.error(f"failed to transfer to supervisor: {e}")
            raise e

        logger.info(
            "transfer to supervisor successful",
            extra={"supervisor_identity": result.supervisor_identity},
        )
        await self.session.say(
            "you are on the line with my supervisor. I'll be hanging up now.",
            allow_interruptions=False,
        )
        self.session.shutdown()


server = AgentServer()


@server.rtc_session(agent_name="sip-inbound")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        vad=silero.VAD.load(),
        llm="openai/gpt-4.1-mini",
        stt="assemblyai/universal-streaming",
        tts="elevenlabs",
        turn_detection=MultilingualModel(),
    )

    support_agent = SupportAgent()

    await session.start(
        agent=support_agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # enable Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )


if __name__ == "__main__":
    # this example requires explicit dispatch using named agents
    # supervisor will be placed in a separate room, and we do not want it to dispatch the default agent
    cli.run_app(server)
