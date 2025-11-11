import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from examples.voice_agents.basic_agent_interruption_handler import InterruptionPolicy, InterruptionFilter, InterruptionDecision
import os
# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
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


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["interrupt_policy"] = InterruptionPolicy.from_env()

async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=float(os.getenv("FALSE_INTERRUPTION_TIMEOUT", "1.0")),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    agent_speaking = {"flag": False}

    def _mark_speaking(val: bool):
        agent_speaking["flag"] = val
        logger.debug(f"agent_speaking={val}")

    try:
        @session.on("agent_speech_started")
        def _on_agent_speech_started(_ev):
            _mark_speaking(True)

        @session.on("agent_speech_finished")
        def _on_agent_speech_finished(_ev):
            _mark_speaking(False)
    except Exception:
        
        logger.debug("agent_speech_* events not available in this SDK build")

    # Build the filter
    interrupt_filter = InterruptionFilter(ctx.proc.userdata["interrupt_policy"])
    
    @session.on("transcription")
    def _on_transcription(ev):
        """
        Expected ev-like shape:
          ev.text: str
          ev.confidence: Optional[float]
          ev.is_final: Optional[bool]
          ev.source: "user" | "agent" (some builds provide this)
        We only act on USER speech.
        """
        text = getattr(ev, "text", "") or ""
        confidence = getattr(ev, "confidence", None)
        source = getattr(ev, "source", "user")

        if not text.strip():
            return
        if source != "user":
            return

        decision = interrupt_filter.decide(
            text=text,
            confidence=confidence,
            agent_speaking=agent_speaking["flag"],
        )

        if decision == InterruptionDecision.IGNORE:
            # Let the agent continue; if a false interruption has paused TTS,
            # resume it quickly (uses built-in resume_false_interruption).
            try:
                session.resume_speaking()
            except Exception:
                # older versions might not expose resume_speaking; harmless to skip
                pass
            logger.info(f"[IGNORED FILLER] {text!r} (conf={confidence})")

        elif decision == InterruptionDecision.INTERRUPT:
            # Stop current TTS and hand control to LLM turn-taking
            try:
                session.interrupt()
            except Exception:
                # Fallback: some builds use stop_speaking()
                try:
                    session.stop_speaking()
                except Exception:
                    pass
            logger.info(f"[INTERRUPT] {text!r} (conf={confidence})")

        else:
            # Agent quiet, register as normal input
            logger.debug(f"[REGISTER] {text!r} (conf={confidence})")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
