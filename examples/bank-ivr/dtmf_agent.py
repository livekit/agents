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
    TelephonyOptions,
    cli,
    metrics,
)
from livekit.agents.llm.tool_context import function_tool
from livekit.plugins import deepgram, elevenlabs, openai, silero

logger = logging.getLogger("phone-tree-agent")

load_dotenv()


server = AgentServer()


PHONE_TREE_AGENT_DISPATCH_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-telephony-agent")


class DtmfAgent(Agent):
    def __init__(self, tasks: list[str]) -> None:
        super().__init__(
            instructions=(
                dedent(
                    f"""
                    # Role and Objective
                    - Act as a voice assistant that helps users navigate a bank's IVR system by entering numbers via keypad as a simulated human caller.
                    # Instructions
                    - Use the DTMF tool whenever digits are required to be entered; do not say numbers aloud if keypad entry is expected.
                    - Assume the persona of a human caller interacting naturally with the IVR.

                    # Tasks
                    Below are the tasks you will perform:

                    {"\n".join([f"{i + 1}. {task}" for i, task in enumerate(tasks)])}

                    - You will use account number 10000001 and PIN 0000 to authenticate and navigate the IVR.
                    - Carefully listen to each IVR prompt and select the most appropriate option.
                    - Use only the DTMF tool to follow the IVR instructions; if an unavailable action is required, note the limitation and propose alternatives.
                    - You should NEVER enter a single '#' or '*' key alone, always make sure the key is appended to the end of some non-empty number sequence.

                    # Example
                    - If the prompt states: “Press 1 for account services,” call `send_dtmf_events` with `['1']` will wait IVR to process. Use `['1', '#']` to bypass the waiting period.
                    - Prefer bypassing the waiting period by always appending `#` to the digit sequence you send; the IVR treats that as an instant confirmation.
                    """
                )
            ),
        )

    @function_tool
    async def record_task_result(self, content: str) -> None:
        """
        Record the IVR navigation task results.

        Args:
            content: The information gathered from completing a task, short validation, or any IVR interaction observation.
        """
        logger.info(f"==> {content}")


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
        telephony_options=TelephonyOptions(
            ivr_detection=True,
            max_ivr_silence_duration=15.0,
        ),
        min_endpointing_delay=3,
    )

    tasks = [
        "Retrieve the checking account balance and read the three most recent transactions.",
        "Confirm the high-yield savings balance and the posted interest rate.",
        "Report the Platinum Travel Rewards credit card statement balance, minimum payment, and due date.",
        "Summarize the mortgage outstanding balance, monthly payment amount, and next payment due date.",
        "Provide the customer's reward tier, total points, and available cashback.",
    ]

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
        agent=DtmfAgent(tasks=tasks),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
