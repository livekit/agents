import logging
from typing import Annotated

from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm, metrics
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import google, silero

load_dotenv()
logger = logging.getLogger("voice-assistant")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# An example Voice Agent using Google STT, Gemini 2.0 Flash, and Google TTS.
# Prerequisites:
# 1. livekit-plugins-openai[vertex] package installed
# 2. save your service account credentials and set the following environments:
#    * GOOGLE_APPLICATION_CREDENTIALS to the path of the service account key file
#    * GOOGLE_CLOUD_PROJECT to your Google Cloud project ID
# 3. the following services are enabled on your Google Cloud project:
#    * Vertex AI
#    * Cloud Speech-to-Text API
#    * Cloud Text-to-Speech API

# Read more about authentication with Google: https://cloud.google.com/docs/authentication/application-default-credentials


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def get_weather(
        location: Annotated[str, llm.TypeInfo(description="The location to get the weather for")],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        return f"The weather in {location} is sunny."

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=google.STT(),
        llm=google.LLM(),
        tts=google.TTS(
            voice_name="en-US-Journey-D",
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    agent.start(ctx.room, participant)

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    await agent.say(
        "Hi there, this is Gemini, how can I help you today?", allow_interruptions=False
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
