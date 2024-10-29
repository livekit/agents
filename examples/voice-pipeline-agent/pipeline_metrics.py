import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import PipelineMetrics, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

load_dotenv()
logger = logging.getLogger("metrics-example")


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

    @agent.on("metrics_collected")
    def on_metrics_collected(metrics: PipelineMetrics):
        if metrics["type"] == "vad_metrics":
            return  # don't log VAD metrics because it is noisy

        logger.info(f"metrics collected: {metrics}")

        # calculate the cost of a LLM request
        if metrics["type"] == "llm_metrics":
            prompt_tokens = metrics["prompt_tokens"]
            completion_tokens = metrics["completion_tokens"]

            # last gpt4o price
            input_price = 2.50 / (10**6)  # $2.50 per million tokens
            output_price = 10 / (10**6)  # $10 per million tokens

            logger.info(
                f"Calculated LLM cost: ${input_price * prompt_tokens + output_price * completion_tokens} for {prompt_tokens + completion_tokens} tokens"
            )

        # calculate the cost of a TTS request
        if metrics["type"] == "tts_metrics":
            num_characters = metrics["num_characters"]
            price = 15 / (10**6)  # $15 per million characters
            logger.info(
                f"Calculated TTS cost: ${price * num_characters} for {num_characters} characters"
            )

        # calculate the cost of a STT request
        if metrics["type"] == "stt_metrics":
            audio_duration = metrics["audio_duration"]
            price = 0.0043  # $0.0043 per minute

            logger.info(
                f"Calculated aggregated STT cost: ${price * audio_duration / 60} for {audio_duration} seconds"
            )

    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
