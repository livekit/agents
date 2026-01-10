"""
Azure Voice Agent - Using Azure Speech Services and Azure OpenAI
"""

import logging
import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    MetricsCollectedEvent,
    tts as tts_module,
)
from livekit.plugins import azure, openai, silero
from livekit.plugins.turn_detector import multilingual

logger = logging.getLogger("azure-agent")
logger.setLevel(logging.INFO)

load_dotenv()


# Logging TTS wrapper that shows sentence breaks
class SentenceLoggingTTS(tts_module.TTS):
    def __init__(self, wrapped_tts: tts_module.TTS):
        super().__init__(
            capabilities=wrapped_tts.capabilities,
            sample_rate=wrapped_tts.sample_rate,
            num_channels=wrapped_tts.num_channels,
        )
        self._wrapped = wrapped_tts
        self._sentence_count = 0

    def synthesize(self, text: str, *, conn_options=None) -> tts_module.ChunkedStream:
        self._sentence_count += 1
        logger.info(f"📝 Sentence #{self._sentence_count}: '{text}'")

        if conn_options:
            return self._wrapped.synthesize(text, conn_options=conn_options)
        return self._wrapped.synthesize(text)

    def update_options(self, **kwargs):
        if hasattr(self._wrapped, 'update_options'):
            return self._wrapped.update_options(**kwargs)

    @property
    def model(self) -> str:
        return self._wrapped.model

    @property
    def provider(self) -> str:
        return self._wrapped.provider


@function_tool
async def get_weather(
    context: RunContext,
    location: str,
):
    """Called when the user asks about weather.

    Args:
        location: The city or location to get weather for
    """
    logger.info(f"Getting weather for {location}")
    # In a real scenario, you would call a weather API here
    return f"The weather in {location} is sunny with a temperature of 72°F."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Create the agent with instructions and tools
    agent = Agent(
        instructions="Your name is Azure Assistant. You interact with users via voice. "
        "Keep your responses concise and conversational. "
        "Do not use emojis, asterisks, markdown, or special characters in your responses. "
        "You are helpful, friendly, and professional.",
        tools=[get_weather],
    )

    # Create Azure TTS with sentence logging
    azure_tts = azure.TTS(
        voice=os.getenv("AZURE_SPEECH_VOICE", "en-US-JennyNeural"),
    )

    # Wrap synthesize method to log sentences
    original_synthesize = azure_tts.synthesize
    sentence_count = [0]  # Use list to allow mutation in closure

    def logging_synthesize(text: str, *, conn_options=None):
        sentence_count[0] += 1
        logger.info(f"📝 Sentence #{sentence_count[0]}: '{text}'")
        if conn_options:
            return original_synthesize(text, conn_options=conn_options)
        return original_synthesize(text)

    azure_tts.synthesize = logging_synthesize

    # Create agent session with Azure services
    session = AgentSession(
        # Azure Speech-to-Text
        stt=azure.STT(),

        # Azure OpenAI for LLM
        llm=openai.LLM.with_azure(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        ),

        # Azure Text-to-Speech
        tts=azure_tts,

        # Voice Activity Detection
        vad=silero.VAD.load(),

        # Turn detection - Multilingual model
        turn_detection=multilingual.MultilingualModel(),

        # Preemptive generation for faster responses
        preemptive_generation=True,

        # Handle false interruptions
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # Log metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # Log TTS sentence breaking
    @session.on("agent_speech_started")
    def _on_agent_speech_started(ev):
        logger.info(f"🎤 TTS STARTED")

    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(ev):
        logger.info(f"🎤 TTS COMMITTED - Audio playback started")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the agent session
    await session.start(agent=agent, room=ctx.room)

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and introduce yourself as Azure Assistant."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
