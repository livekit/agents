import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    room_io,
)
from livekit.agents.beta.workflows import WarmTransferTask
from livekit.agents.llm import ToolError, function_tool
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("warm-transfer")

load_dotenv()

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def transfer_to_human(self) -> None:
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
            assert SIP_TRUNK_ID is not None
            assert SUPERVISOR_PHONE_NUMBER is not None

            result = await WarmTransferTask(
                target_phone_number=SUPERVISOR_PHONE_NUMBER,
                sip_trunk_id=SIP_TRUNK_ID,
                sip_number=SIP_NUMBER,
                chat_ctx=self.chat_ctx,
                # add extra instructions for summarization
                # you can also customize the entire instructions by overriding the `get_instructions` method
                extra_instructions=SUMMARY_INSTRUCTIONS,
            )
        except ToolError as e:
            logger.error(f"failed to transfer to supervisor with tool error: {e}")
            raise e
        except Exception as e:
            logger.exception("failed to transfer to supervisor")
            raise ToolError(f"failed to transfer to supervisor with error: {e}") from e

        logger.info(
            "transfer to supervisor successful",
            extra={"supervisor_identity": result.human_agent_identity},
        )
        await self.session.say(
            "you are on the line with my supervisor. I'll be hanging up now.",
            allow_interruptions=False,
        )
        self.session.shutdown()


server = AgentServer()


@server.rtc_session(agent_name="sip-inbound")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm="openai/gpt-4.1-mini",
        stt="assemblyai/universal-streaming",
        tts="cartesia/sonic-3",
        turn_detection=MultilingualModel(),
    )

    support_agent = SupportAgent()

    await session.start(
        agent=support_agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # enable Krisp BVC noise cancellation
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
            delete_room_on_close=False,  # keep the room open for the customer and supervisor
        ),
    )


INSTRUCTIONS = """
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
"""


SUMMARY_INSTRUCTIONS = """
Introduce the conversation from your perspective as the AI assistant who participated in this call:

WHO you're talking to (name, role, company if mentioned)
WHY they contacted you (goal, problem, request)
WHY a human agent is requested or needed at this point
Brief summary in 100-200 characters from a first-person perspective"""


if __name__ == "__main__":
    # this example requires explicit dispatch using named agents
    # supervisor will be placed in a separate room, and we do not want it to dispatch the default agent
    cli.run_app(server)
