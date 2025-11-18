import logging
import asyncio 

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)

# === START: MODIFIED IMPORTS ===
# We are importing the specific plugin classes here to instantiate them directly,
# bypassing the problematic local inference path setup.
from livekit.plugins import deepgram # Added for direct STT instantiation
from livekit.plugins import openai    # Added for direct LLM instantiation
from livekit.plugins import cartesia  # Added for direct TTS instantiation
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
# === END: MODIFIED IMPORTS ===

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

# =========================================================================
# === START: INTERRUPT HANDLER MODIFICATIONS ===

# 1. Define the configurable list of filler words
IGNORED_WORDS = {'uh', 'umm', 'hmm', 'haan', 'yeah', 'like', 'uhh', 'um', 'mhm'} # Added common fillers

def is_filler_only(transcribed_text: str) -> bool:
    """Checks if the transcription contains only words on the ignored list."""
    words = transcribed_text.lower().split()
    if not words:
        return True # Treat empty transcription as filler
    
    # Check every word against the ignored list
    for word in words:
        if word not in IGNORED_WORDS:
            return False # Found a non-filler word, so it's a real interruption
            
    return True # Only fillers were found

# === END: INTERRUPT HANDLER MODIFICATIONS ===
# =========================================================================


class MyAgent(Agent):
# ... (MyAgent class remains unchanged) ...
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    # === START: MODIFIED AGENTSESSION CONFIG ===
    session = AgentSession(
        # STT: Instantiate Deepgram plugin directly to run locally
        stt=deepgram.STT(),
        # LLM: Instantiate OpenAI plugin directly to run locally
        llm=openai.LLM(),
        # TTS: Instantiate Cartesia plugin directly to run locally
        tts=cartesia.TTS(),
        
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
        
        # resume_false_interruption is crucial for your code to work
        resume_false_interruption=True, 
        false_interruption_timeout=1.0,
    )
    # === END: MODIFIED AGENTSESSION CONFIG ===

    # =========================================================================
    # === START: INTERCEPTION LOGIC (ASYNC/SYNC FIX) ===
    
    async def _process_smart_interruption(event):
        """Contains the core logic, run as an asynchronous task."""
        
        # Only process final, confirmed transcriptions
        if not event.final:
            return

        transcript_text = event.text.lower()

        # Check the agent's state: is it currently speaking?
        if session.state == "speaking":
            
            if is_filler_only(transcript_text):
                # 1. Filler-Only: BLOCK the interruption signal
                
                logger.info(f"LOG: IGNORED INTERRUPTION. Resuming agent speech. Filler: '{transcript_text}'")
                
                # CRITICAL: Call this method to override the VAD pause signal and force the agent to continue TTS.
                await session.resume_interruption(reason="filler-filter-override")
                
            else:
                # 2. Genuine Interruption: LET IT PASS
                logger.info(f"LOG: VALID INTERRUPTION. Agent will stop. Command: '{transcript_text}'")
                # No action needed, the VAD pause signal proceeds normally.
        
        else:
            # Agent is listening (not speaking)
            logger.debug(f"Registered user speech (Agent was listening). Text: '{transcript_text}'")

    
    @session.on("user_input_transcribed")
    def handle_smart_interruption(event):
        """
        LiveKit requires this to be synchronous. We defer the async logic
        to resolve the concurrency error.
        """
        # IMMEDIATELY create a task to run the actual async logic in the background
        asyncio.create_task(_process_smart_interruption(event))

    # === END: INTERCEPTION LOGIC (ASYNC/SYNC FIX) ===
    # =========================================================================

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)