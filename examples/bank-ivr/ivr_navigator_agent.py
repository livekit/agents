import logging
import os
from textwrap import dedent

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    inference,
    metrics,
)
from livekit.agents.llm.tool_context import function_tool
from livekit.plugins import silero

logger = logging.getLogger("phone-tree-agent")

load_dotenv()


server = AgentServer()


PHONE_TREE_AGENT_DISPATCH_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-telephony-agent")


class DtmfAgent(Agent):
    def __init__(self, user_request: str) -> None:
        super().__init__(
            instructions=(
                dedent(
                    f"""
                    # Role and Objective
                    - Act as a voice assistant that helps users navigate a bank's IVR system by entering numbers via keypad as a simulated human caller.
                    # Instructions
                    - Use the DTMF tool whenever digits are required to be entered; do not say numbers aloud if keypad entry is expected.
                    - Assume the persona of a human caller interacting naturally with the IVR.

                    # Task
                    - Your single task is:
                      {user_request}

                    Once complete with the task, call `record_task_result_and_hang_up` with the result.

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
    async def record_task_result_and_hang_up(self, context: RunContext, content: str) -> None:
        """
        Record the IVR navigation task results and hang up the call/session.

        ONLY call this tool once you have completed the task and the IVR has processed the result.

        DO not call this tool under any other circumstances, assume IVR system is always working correctly.

        Args:
            content: The information gathered from completing a task, short validation, or any IVR interaction observation.
        """
        logger.info(f"==> {content}")
        context.session.shutdown(drain=True)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=PHONE_TREE_AGENT_DISPATCH_NAME)
async def dtmf_session(ctx: JobContext) -> None:
    await ctx.connect()
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=inference.LLM("openai/gpt-4.1"),
        stt=inference.STT("deepgram/flux-general"),
        tts=inference.TTS("rime/arcana"),
        # This flag does two things:
        # 1. Helps agent avoid getting stuck listening to repeating IVR loops by actively responding when a loop is detected.
        # 2. Automatically gives the agent the `send_dtmf_events` tool to allow it to dial DTMF digits.
        ivr_detection=True,
        min_endpointing_delay=5,
    )

    # Get the single user request from the room metadata (set by the dispatcher)
    user_request = (
        "Summarize jordan carter checking account, including balance and recent transactions. "
        'Use account number "1 0 0 0 0 0 0 1" and PIN "0 0 0 0" to authenticate and navigate the IVR.'
    )
    logger.info(f"==> User request: {user_request}")

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
        agent=DtmfAgent(user_request=user_request),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
