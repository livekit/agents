import logging

from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm, metrics
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

load_dotenv()
logger = logging.getLogger("metrics-example")

# This example logs pipeline metrics and computes cost of the session

OPENAI_LLM_INPUT_PRICE = 2.50 / (10**6)  # $2.50 per million tokens
OPENAI_LLM_OUTPUT_PRICE = 10 / (10**6)  # $10 per million tokens
OPENAI_TTS_PRICE = 15 / (10**6)  # $15 per million characters
DEEPGRAM_STT_PRICE = 0.0043  # $0.0043 per minute


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_session_cost():
        summary = usage_collector.get_summary()
        llm_cost = (
            summary.llm_prompt_tokens * OPENAI_LLM_INPUT_PRICE
            + summary.llm_completion_tokens * OPENAI_LLM_OUTPUT_PRICE
        )
        tts_cost = summary.tts_characters_count * OPENAI_TTS_PRICE
        stt_cost = summary.stt_audio_duration * DEEPGRAM_STT_PRICE / 60

        total_cost = llm_cost + tts_cost + stt_cost

        logger.info(
            f"Total cost: ${total_cost:.4f} (LLM: ${llm_cost:.4f}, TTS: ${tts_cost:.4f}, STT: ${stt_cost:.4f})"
        )

    ctx.add_shutdown_callback(log_session_cost)

    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
