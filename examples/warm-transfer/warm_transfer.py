import asyncio
import logging
import os
from typing import Literal

from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    JobContext,
    PlayHandle,
    RunContext,
    cli,
    llm,
    room_io,
    stt,
    tts,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

load_dotenv()

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"

# status enums representing the two sessions
SupervisorStatus = Literal["inactive", "summarizing", "merged", "failed"]
CustomerStatus = Literal["active", "escalated", "passive"]
_supervisor_identity = "supervisor-sip"


class SessionManager:
    """
    Helper class to orchestrate the session flow
    """

    def __init__(
        self,
        *,
        ctx: JobContext,
        customer_room: rtc.Room,
        customer_session: AgentSession,
        supervisor_contact: str,
        lkapi: api.LiveKitAPI,
    ):
        self.ctx = ctx
        self.customer_session = customer_session
        self.customer_room = customer_room
        self.background_audio = BackgroundAudioPlayer()
        self.hold_audio_handle: PlayHandle | None = None

        self.supervisor_session: AgentSession | None = None
        self.supervisor_room: rtc.Room | None = None
        self.supervisor_contact = supervisor_contact
        self.lkapi = lkapi

        self.customer_status: CustomerStatus = "active"
        self.supervisor_status: SupervisorStatus = "inactive"

    async def start(self) -> None:
        await self.background_audio.start(
            room=self.customer_room, agent_session=self.customer_session
        )

    async def start_transfer(self):
        if self.customer_status != "active":
            return

        self.customer_status = "escalated"

        self.start_hold()

        try:
            # dial human supervisor in a new room
            supervisor_room_name = self.customer_room.name + "-supervisor"
            self.supervisor_room = rtc.Room()
            token = (
                api.AccessToken()
                .with_identity("summary-agent")
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=supervisor_room_name,
                        can_update_own_metadata=True,
                        can_publish=True,
                        can_subscribe=True,
                    )
                )
            )

            logger.info(
                f"connecting to supervisor room {supervisor_room_name}",
                extra={"token": token.to_jwt(), "url": os.getenv("LIVEKIT_URL")},
            )

            await self.supervisor_room.connect(os.getenv("LIVEKIT_URL"), token.to_jwt())
            # if supervisor hung up for whatever reason, we'd resume the customer conversation
            self.supervisor_room.on("disconnected", self.on_supervisor_room_close)

            self.supervisor_session = AgentSession(
                vad=silero.VAD.load(),
                llm=_create_llm(),
                stt=_create_stt(),
                tts=_create_tts(),
                turn_detection=MultilingualModel(),
            )

            supervisor_agent = SupervisorAgent(prev_ctx=self.customer_session.history)
            supervisor_agent.session_manager = self
            await self.supervisor_session.start(
                agent=supervisor_agent,
                room=self.supervisor_room,
                room_options=room_io.RoomOptions(
                    close_on_disconnect=True,
                ),
            )

            # dial the supervisor
            await self.lkapi.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=SIP_TRUNK_ID,
                    sip_call_to=self.supervisor_contact,
                    room_name=supervisor_room_name,
                    participant_identity=_supervisor_identity,
                    wait_until_answered=True,
                )
            )
            self.supervisor_status = "summarizing"

        except Exception:
            logger.exception("could not start transfer")
            self.customer_status = "active"
            await self.set_supervisor_failed()

    def on_supervisor_room_close(self, reason: rtc.DisconnectReason):
        asyncio.create_task(self.set_supervisor_failed())

    def on_customer_participant_disconnected(self, participant: rtc.RemoteParticipant):
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return

        logger.info(f"participant disconnected: {participant.identity}, deleting room")
        self.customer_room.off(
            "participant_disconnected", self.on_customer_participant_disconnected
        )
        self.ctx.delete_room()

    async def set_supervisor_failed(self):
        self.supervisor_status = "failed"

        # when we've encountered an error during the transfer, agent would need to recover
        # from there.
        self.stop_hold()
        self.customer_session.generate_reply(
            instructions="let the user know that we are unable to connect them to a supervisor right now."
        )

        if self.supervisor_session:
            await self.supervisor_session.aclose()
            self.supervisor_session = None

    async def merge_calls(self):
        if self.supervisor_status != "summarizing":
            return

        try:
            # we no longer care about the supervisor session. it's supposed to be over
            self.supervisor_room.off("disconnected", self.on_supervisor_room_close)
            await self.lkapi.room.move_participant(
                api.MoveParticipantRequest(
                    room=self.supervisor_room.name,
                    identity=_supervisor_identity,
                    destination_room=self.customer_room.name,
                )
            )

            self.stop_hold()

            # now both users are in the same room, we'll ensure that whenever anyone leaves,
            # the entire call is terminates
            self.customer_room.on(
                "participant_disconnected", self.on_customer_participant_disconnected
            )

            # wait for this to be fully played out and close the agent's session with customer
            await self.customer_session.say(
                "you are on the line with my supervisor. I'll be hanging up now."
            )
            await self.customer_session.aclose()

            if self.supervisor_session:
                await self.supervisor_session.aclose()
                self.supervisor_session = None

            logger.info("calls merged")
        except Exception:
            logger.exception("could not merge calls")
            await self.set_supervisor_failed()

    def stop_hold(self):
        if self.hold_audio_handle:
            self.hold_audio_handle.stop()
            self.hold_audio_handle = None

        self.customer_session.input.set_audio_enabled(True)
        self.customer_session.output.set_audio_enabled(True)

    def start_hold(self):
        self.customer_session.input.set_audio_enabled(False)
        self.customer_session.output.set_audio_enabled(False)
        self.hold_audio_handle = self.background_audio.play(
            AudioConfig("hold_music.mp3", volume=0.8),
            loop=True,
        )


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_support_agent_instructions,
        )
        self.session_manager: SessionManager | None = None

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
        await self.session.say("Please hold while I connect you to a human agent.")
        await self.session_manager.start_transfer()

        # no generation required from the function call
        return None


class SupervisorAgent(Agent):
    def __init__(self, prev_ctx: llm.ChatContext) -> None:
        prev_convo = ""
        context_copy = prev_ctx.copy(
            exclude_empty_message=True, exclude_instructions=True, exclude_function_call=True
        )
        for msg in context_copy.items:
            if msg.role == "user":
                prev_convo += f"Customer: {msg.text_content}\n"
            else:
                prev_convo += f"Assistant: {msg.text_content}\n"
        # to make it easier to test, uncomment to use a mock conversation history
        #         prev_convo = """
        # Customer: I'm having a problem with my account.
        # Assistant: what's wrong?
        # Customer: I'm unable to login.
        # Assistant: I see, looks like your account has been locked out.
        # Customer: Can you help me?
        # Assistant: I'm not able to help with that, would you like to speak to a human agent?
        # Customer: Yes please.
        # """

        super().__init__(
            instructions=_supervisor_agent_instructions + "\n\n" + prev_convo,
        )
        self.prev_ctx = prev_ctx
        self.session_manager: SessionManager | None = None

    async def on_enter(self):
        """Summarize the current conversation and explain the situation to the supervisor."""
        logger.info("supervisor agent entered")
        # since we are dialing out to a supervisor, let them speak first, and the agent will summarize the conversation

    @function_tool
    async def connect_to_customer(self, context: RunContext):
        """Called when the supervisor has agreed to start speaking to the customer.

        The agent should explicitly confirm that they are ready to connect.
        """
        await self.session.say("connecting you to the customer now.")
        await self.session_manager.merge_calls()
        return None

    @function_tool
    async def voicemail_detected(self, context: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        self.session_manager.set_supervisor_failed()


server = AgentServer()


@server.rtc_session(agent_name="sip-inbound")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        vad=silero.VAD.load(),
        llm=_create_llm(),
        stt=_create_stt(),
        tts=_create_tts(),
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
        ),
    )

    session_manager = SessionManager(
        ctx=ctx,
        customer_room=ctx.room,
        customer_session=session,
        supervisor_contact=SUPERVISOR_PHONE_NUMBER,
        lkapi=ctx.api,
    )
    support_agent.session_manager = session_manager

    await session_manager.start()


def _create_llm() -> llm.LLM:
    return openai.LLM(model="gpt-4.1-mini")


def _create_stt() -> stt.STT:
    return deepgram.STT(model="nova-3", language="multi")


def _create_tts() -> tts.TTS:
    return cartesia.TTS()


if __name__ == "__main__":
    # this example requires explicit dispatch using named agents
    # supervisor will be placed in a separate room, and we do not want it to dispatch the default agent
    cli.run_app(server)

_common_instructions = """
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
"""

_support_agent_instructions = (
    _common_instructions
    + """
# Identity

You are a customer support agent for LiveKit.

# Transferring to a human

In some cases, the user may ask to speak to a human agent. This could happen when you are unable to answer their question.
When such is requested, you would always confirm with the user before initiating the transfer.
"""
)

_supervisor_agent_instructions = (
    _common_instructions
    + """
# Identity

You are an agent that is reaching out to a supervisor for help. There has been a previous conversation
between you and a customer, the conversation history is included below.

# Goal

Your main goal is to give the supervisor sufficient context about why the customer had called in,
so that the supervisor could gain sufficient knowledge to help the customer directly.

# Context

In the conversation, user refers to the supervisor, customer refers to the person who's transcript is included.
Remember, you are not speaking to the customer right now, you are speaking to the supervisor.

Once the supervisor has confirmed, you should call the tool `connect_to_customer` to connect them to the customer.

Start by giving them a summary of the conversation so far, and answer any questions they might have.

## Conversation history with customer
"""
)
