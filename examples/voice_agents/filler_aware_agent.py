"""
Filler-aware voice agent that filters filler words during agent speech.

Environment Variables:
    FILLER_WORDS: Comma-separated filler words (default: "uh,umm,hmm,haan,huh")
    FILLER_CONFIDENCE_THRESHOLD: Min confidence (default: "0.5")
    ENABLE_FILLER_FILTERING: Enable/disable (default: "true")
"""

import asyncio
import logging
import os

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
from livekit.agents.voice import FillerFilterConfig, FillerFilteredAgentActivity
from livekit.plugins import silero

# Turn detection: Use VAD-based by default (no model download required)
# For multilingual turn detection, uncomment the line below and run:
# python3 filler_aware_agent.py download-files
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("filler-aware-agent")
logger.setLevel(logging.INFO)

load_dotenv()


def load_filler_config() -> FillerFilterConfig:
    filler_words_str = os.getenv("FILLER_WORDS", "uh,umm,hmm,haan,huh")
    filler_words = [w.strip() for w in filler_words_str.split(",") if w.strip()]

    confidence_threshold = float(os.getenv("FILLER_CONFIDENCE_THRESHOLD", "0.5"))

    enable_filtering = os.getenv("ENABLE_FILLER_FILTERING", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    config = FillerFilterConfig(
        filler_words=filler_words,
        confidence_threshold=confidence_threshold,
        enable_filtering=enable_filtering,
        log_filtered=True,
        log_interruptions=True,
    )

    logger.info(f"Filler filter config loaded: {config}")
    return config


class FillerAwareAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Alex. You are a helpful voice assistant that demonstrates "
                "intelligent interruption handling. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You are friendly and professional."
            )
        )

    async def on_enter(self):
        logger.info("FillerAwareAgent entered session")
        self.session.generate_reply()

    @function_tool
    async def get_time(self, context: RunContext):
        """Get the current time."""
        import datetime

        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        logger.info(f"Time requested: {time_str}")
        return f"The current time is {time_str}."

    @function_tool
    async def tell_joke(self, context: RunContext):
        """Tell a joke to the user."""
        joke = "Why don't scientists trust atoms? Because they make up everything!"
        logger.info("Joke requested")
        return joke


class FillerAwareAgentSession(AgentSession):
    def __init__(self, *args, filler_config: FillerFilterConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._filler_config = filler_config or FillerFilterConfig()
        logger.info("FillerAwareAgentSession initialized")

    async def _update_activity(
        self,
        agent: Agent,
        *,
        previous_activity: str = "close",
        new_activity: str = "start",
        blocked_tasks: list[asyncio.Task] | None = None,
        wait_on_enter: bool = True,
    ) -> None:
        async with self._activity_lock:
            self._agent = agent

            if new_activity == "start":
                previous_agent = self._activity.agent if self._activity else None
                if agent._activity is not None and (
                    agent is not previous_agent or previous_activity != "close"
                ):
                    raise RuntimeError("cannot start agent: an activity is already running")

                self._next_activity = FillerFilteredAgentActivity(
                    agent, self, filler_config=self._filler_config
                )
                logger.info("Created FillerFilteredAgentActivity with filler filtering enabled")
            elif new_activity == "resume":
                if agent._activity is None:
                    raise RuntimeError("cannot resume agent: no existing active activity to resume")
                self._next_activity = agent._activity

            if self._root_span_context is not None:
                from opentelemetry import context as otel_context

                otel_context.attach(self._root_span_context)

            previous_activity_v = self._activity
            if self._activity is not None:
                if previous_activity == "close":
                    await self._activity.drain()
                    await self._activity.aclose()
                elif previous_activity == "pause":
                    await self._activity.pause(blocked_tasks=blocked_tasks or [])

            self._activity = self._next_activity
            self._next_activity = None

            run_state = self._global_run_state
            if run_state:
                run_state._agent_handoff(
                    old_agent=previous_activity_v.agent if previous_activity_v else None,
                    new_agent=self._activity.agent,
                )

            if new_activity == "start" and wait_on_enter:
                await self._activity.agent.on_enter()

            await self._activity.start()


def prewarm(proc: JobProcess):
    logger.info("Prewarming models...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"Starting filler-aware agent in room: {ctx.room.name}")

    filler_config = load_filler_config()

    session = FillerAwareAgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=0,
        resume_false_interruption=True,
        false_interruption_timeout=2.0,
        filler_config=filler_config,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Session usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        logger.info(f"Agent state changed: {ev.old_state} → {ev.new_state}")

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        logger.info(
            f"User transcript: '{ev.transcript}' (final={ev.is_final}, language={ev.language})"
        )

    await session.start(
        agent=FillerAwareAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    logger.info("Filler-aware agent session started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
