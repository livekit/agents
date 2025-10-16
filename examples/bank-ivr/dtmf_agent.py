import logging
import os
from textwrap import dedent

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomOutputOptions,
    cli,
    metrics,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

logger = logging.getLogger("phone-tree-agent")

load_dotenv()


server = AgentServer()


PHONE_TREE_AGENT_DISPATCH_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-telephony-agent")


class DtmfAgent(Agent):
    def __init__(self, goal: str) -> None:
        super().__init__(
            instructions=(
                dedent(
                    f"""
                    # Role and Objective
                    - Act as a voice assistant that helps users navigate a bank's IVR system by entering numbers via keypad as a simulated human caller.
                    # Instructions
                    - Use the DTMF tool whenever digits are required to be entered; do not say numbers aloud if keypad entry is expected.
                    - Assume the persona of a human caller interacting naturally with the IVR.
                    - Your primary goal is: `{goal}`.
                    - Carefully listen to each IVR prompt and select the most appropriate option.
                    - After each digit entry, validate whether the intended IVR action occurred (e.g., appropriate menu or response received) in 1-2 lines, and decide whether to continue or self-correct if needed.
                    - Use only the DTMF tool to follow the IVR instructions; if an unavailable action is required, note the limitation and propose alternatives.
                    # Example
                    - If the prompt states: “Press 1 for account services,” call `send_dtmf_events` with `['1']` will wait IVR to process. Use `['1', '#']` to bypass the waiting period.
                    - Prefer bypassing the waiting period by always appending `#` to the digit sequence you send; the IVR treats that as an instant confirmation.
                    """
                )
            ),
        )


@server.realtime_session(agent_name=PHONE_TREE_AGENT_DISPATCH_NAME)
async def dtmf_session(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-5"),
        stt=deepgram.STT(model="nova-3"),
        tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
        dial_to_phone_ivr=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=DtmfAgent(goal=("ask about current account balance")),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
