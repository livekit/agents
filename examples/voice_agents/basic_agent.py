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
from livekit.agents.llm import function_tool
# from livekit.agents.voice.agent import FILLERS 
# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()
# logger = logging.getLogger("basic-agent")
logging.basicConfig(level=logging.INFO)


import os

class FillerManager:
    def __init__(self) -> None:
        base = os.getenv("LIVEKIT_IGNORED_FILLERS", "uh,umm,hmm,haan,accha").split(",")
        self._fillers_by_lang = {
            "default": {b.strip().lower() for b in base if b.strip()},
            "hinglish": {"umm", "haan", "hmm", "accha"},
            "en": {"uh", "umm", "hmm"},
            "hi": {"haan", "accha"},
        }
        self._min_conf = float(os.getenv("LIVEKIT_FILLER_CONFIDENCE", "0.6"))

    def get_all_for(self, lang: str | None):
        lang = (lang or "default").split("-")[0].lower()
        merged = set(self._fillers_by_lang.get("default", set()))
        merged |= self._fillers_by_lang.get("hinglish", set())
        merged |= self._fillers_by_lang.get(lang, set())
        return merged

    def get_min_conf(self) -> float:
        return self._min_conf

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

FILLERS = FillerManager()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        llm="google/gemini-2.0-flash",
        tts="cartesia/sonic-3:6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

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
    async def _on_transcription(ev):
        alt = ev.alternatives[0]
        transcript = (alt.text or "").strip()
        confidence = getattr(alt, "confidence", None)
        language = getattr(alt, "language", "default")

        tokens = [t for t in transcript.lower().split() if t]
        fillers = FILLERS.get_all_for(language)
        min_conf = FILLERS.get_min_conf()

        def is_filler_only() -> bool:
            if not tokens:
                return True
            return all(tok in fillers for tok in tokens)

        agent_is_speaking = session.is_speaking

        if agent_is_speaking:
            # explicit commands win
            if any(k in transcript.lower() for k in ["stop", "wait", "hold on", "no not that"]):
                logger.info("[interruption] user said stop -> interrupting")
                session.interrupt_speech()
                return

            # filler-only → ignore
            if is_filler_only():
                logger.debug(f"[ignored filler] {transcript}")
                return

            # low-confidence background → ignore
            if confidence is not None and confidence < min_conf:
                logger.debug(f"[ignored low-conf] {transcript} ({confidence})")
                return

            # real speech → interrupt
            logger.info("[interruption] valid user speech -> interrupting")
            session.interrupt_speech()
        else:
            # agent is quiet → let it pass
            logger.debug(f"[user speech while agent quiet] {transcript}")
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
