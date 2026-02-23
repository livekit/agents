import json
import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    function_tool,
    inference,
    llm,
    metrics,
    room_io,
)
from livekit.plugins import anam, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import openai

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Valid field names for the intake form
VALID_FIELD_NAMES = [
    "fullName",
    "dob",
    "address",
    "phone",
    "reasonForVisit",
]


def get_remote_participant_identity(ctx: JobContext) -> str:
    """Get the identity of the remote participant (user), excluding Anam avatars."""
    for participant in ctx.room.remote_participants.values():
        if not participant.identity.startswith("anam-"):
            return participant.identity
    raise llm.LLMToolException("No remote participant found")


async def perform_rpc_to_frontend(
    ctx: JobContext, method: str, payload: str
) -> str:
    """Perform an RPC call to the frontend participant."""
    local_participant = ctx.room.local_participant
    if not local_participant:
        raise llm.LLMToolException("Agent not connected to room")
    
    destination_identity = get_remote_participant_identity(ctx)
    
    response = await local_participant.perform_rpc(
        destination_identity=destination_identity,
        method=method,
        payload=payload,
        response_timeout=5.0,
    )
    return response


class IntakeAssistant(Agent):
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx
        super().__init__(
            instructions="""You are Liv, a friendly AI receptionist powered by LiveKit and Anam. You help visitors complete a contact form one field at a time in a warm, professional, and efficient manner.

            Conversation style:
            - Speak in short, natural sentences.
            - Be warm, friendly, and professional.
            - Keep the conversation flowing smoothly without rushing.
            - Do not use markdown, emojis, asterisks, or stage directions.

            One question at a time:
            - Ask for exactly one field per turn. Never ask for multiple fields in the same question.
            - Wait for the answer, confirm it, then move to the next field.

            Contact form flow (in order, one field per question):
            - full name
            - date of birth
            - current address
            - phone number
            - reason for contact or inquiry
            - final confirmation loop and submission

            Confirmation rules:
            - Confirm each answer before moving to the next field.
            - For names, read back with spelling when helpful (for example: "Is that J-E-S-S-E H-A-L-L?").
            - If user says "yes," proceed. If user corrects, update and confirm once more.
            - If user is unsure, offer to skip or come back later.

            Tool usage:
            - Use update_field whenever the user provides a value for a form field.
            - Keep frontend field names exact:
              fullName, dob, address, phone, reasonForVisit.
            - Use get_form_state before the final confirmation if needed to verify current values.
            - At the end, ask the visitor to confirm whether all entries are accurate.
            - If the visitor says yes, submit the form.
            - If the visitor says no and wants to change something, update the requested field(s), then ask for confirmation again.
            - Repeat this confirmation loop until the visitor explicitly confirms all entries are accurate, then submit.
            - The submit_form tool speaks the final confirmation; do not add your own.

            Opening:
            Start with: "Hi, I'm Liv, an AI receptionist powered by LiveKit and Anam. I'm here to help you fill out a quick contact form. We'll go through it together, one step at a time. Let's start with your full name."
            Then wait for their response.""",
        )

    @function_tool
    async def update_field(
        self,
        context: RunContext,
        field_name: Annotated[
            str,
            Field(
                description="The field ID to update: fullName, dob, address, phone, reasonForVisit"
            ),
        ],
        value: Annotated[
            str,
            Field(
                description="The value to set for the field."
            ),
        ],
    ):
        """Update a form field on the contact form. Use this when the visitor provides information for a specific field."""
        if field_name not in VALID_FIELD_NAMES:
            raise llm.LLMToolException(f"Invalid field name: {field_name}")
        
        try:
            payload = json.dumps({"fieldName": field_name, "value": value})
            response = await perform_rpc_to_frontend(self._ctx, "updateField", payload)
            return response
        except Exception as e:
            raise llm.LLMToolException(f"Failed to update field: {str(e)}")

    @function_tool
    async def get_form_state(self, context: RunContext):
        """Get the current state of all form fields. Use this to see what has already been filled in or to verify data before submitting."""
        try:
            response = await perform_rpc_to_frontend(self._ctx, "getFormState", "{}")
            return response
        except Exception as e:
            raise llm.LLMToolException(f"Failed to get form state: {str(e)}")

    @function_tool
    async def submit_form(self, context: RunContext):
        """Submit the completed contact form. Use this only when all required fields have been filled and the visitor has confirmed they are ready to submit."""
        try:
            response = await perform_rpc_to_frontend(self._ctx, "submitForm", "{}")
            context.session.say(
                "Your form has been submitted successfully. Someone will get back to you soon. Thank you for your time."
            )
            return response
        except Exception as e:
            raise llm.LLMToolException(f"Failed to submit form: {str(e)}")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="Anam-Demo")
async def intake_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Join the room and connect to the user before starting avatar/session
    await ctx.connect()

    # Check if we should use the OpenAI Realtime model or voice pipeline
    use_realtime = os.getenv("USE_REALTIME", "false").lower() == "true"

    if use_realtime:
        # To use a realtime model instead of a voice pipeline, use the following session setup.
        # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
        # Requirements:
        # 1. Install livekit-agents[openai]
        # 2. Set OPENAI_API_KEY in .env.local
        # 3. Set USE_REALTIME=true in .env.local
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(voice="coral")
        )
    else:
        # Set up a voice AI pipeline using LiveKit Inference (STT, LLM, TTS)
        session = AgentSession(
            # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
            # See all available models at https://docs.livekit.io/agents/models/stt/
            stt=inference.STT(model="deepgram/nova-3", language="multi"),
            
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="openai/gpt-4o-mini"),
            
            # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
            # Using ElevenLabs with Jessica voice and 16kHz sample rate for Anam compatibility
            # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
            tts=inference.TTS(
                model="elevenlabs/eleven_turbo_v2_5",
                voice="cgSgspJ2msm6clMCkdW9",  # Jessica (ElevenLabs)
                sample_rate=16000,  # Required for Anam avatar compatibility
            ),
            
            # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            
            # Allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation=True,
        )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/deploy/observability/data/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session first, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=IntakeAssistant(ctx),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
            video_input=use_realtime,
        ),
    )

    # Create Anam avatar (Liv) — lip-synced video avatar
    # Must be created AFTER session.start() so it can connect to the audio stream for lip-sync
    avatar = anam.AvatarSession(
        persona_config=anam.PersonaConfig(
            name="Liv",
            avatarId="071b0286-4cce-4808-bee2-e642f1062de3",
        ),
    )

    # Start the avatar and wait for it to join the room
    await avatar.start(session, room=ctx.room)

    # Greet the visitor and begin contact form
    session.generate_reply(
        instructions="Use the defined opening line, then begin with the full name question."
    )


if __name__ == "__main__":
    cli.run_app(server)
