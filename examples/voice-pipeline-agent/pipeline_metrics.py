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

    total_llm_prompt_tokens = 0
    total_llm_completion_tokens = 0
    total_tts_characters_count = 0
    total_stt_audio_duration = 0

    @agent.on("metrics_collected")
    def on_metrics_collected(metrics: PipelineMetrics):
        nonlocal \
            total_llm_prompt_tokens, \
            total_llm_completion_tokens, \
            total_tts_characters_count, \
            total_stt_audio_duration

        if metrics["type"] == "vad_metrics":
            return  # don't log VAD metrics because it is noisy

        if metrics["type"] == "llm_metrics":
            total_llm_prompt_tokens += metrics["prompt_tokens"]
            total_llm_completion_tokens += metrics["completion_tokens"]

            sequence_id = metrics["sequence_id"]
            ttft = metrics["ttft"]
            tokens_per_second = metrics["tokens_per_second"]

            logger.info(
                f"LLM metrics: sequence_id={sequence_id}, ttft={ttft:.2f}, tokens_per_second={tokens_per_second:.2f}"
            )

        elif metrics["type"] == "tts_metrics":
            total_tts_characters_count += metrics["characters_count"]

            sequence_id = metrics["sequence_id"]
            ttfb = metrics["ttfb"]
            audio_duration = metrics["audio_duration"]

            logger.info(
                f"TTS metrics: sequence_id={sequence_id}, ttfb={ttfb}, audio_duration={audio_duration:.2f}"
            )

        elif metrics["type"] == "eou_metrics":
            sequence_id = metrics["sequence_id"]
            end_of_utterance_delay = metrics["end_of_utterance_delay"]
            transcription_delay = metrics["transcription_delay"]

            logger.info(
                f"EOU metrics: sequence_id={sequence_id}, end_of_utterance_delay={end_of_utterance_delay:.2f}, transcription_delay={transcription_delay:.2f}"
            )

        elif metrics["type"] == "stt_metrics":
            total_stt_audio_duration += metrics["audio_duration"]
            logger.info(f"STT metrics: audio_duration={metrics['audio_duration']:.2f}")

    async def log_session_cost():
        llm_cost = (
            total_llm_prompt_tokens * OPENAI_LLM_INPUT_PRICE
            + total_llm_completion_tokens * OPENAI_LLM_OUTPUT_PRICE
        )
        tts_cost = total_tts_characters_count * OPENAI_TTS_PRICE
        stt_cost = total_stt_audio_duration * DEEPGRAM_STT_PRICE / 60

        total_cost = llm_cost + tts_cost + stt_cost

        logger.info(
            f"Total cost: ${total_cost:.4f} (LLM: ${llm_cost:.4f}, TTS: ${tts_cost:.4f}, STT: ${stt_cost:.4f})"
        )

    ctx.add_shutdown_callback(log_session_cost)

    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
