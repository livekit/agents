import logging
import os
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
)
from livekit.agents.beta.workflows.dtmf_inputs import (
    GetDtmfTask,
)
from livekit.agents.llm.tool_context import ToolError, function_tool
from livekit.agents.voice.events import RunContext
from livekit.agents.worker import AgentServer
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("dtmf-agent")

load_dotenv()


DTMF_AGENT_DISPATCH_NAME = os.getenv("DTMF_AGENT_DISPATCH_NAME", "my-telephony-agent")

server = AgentServer()


class DtmfAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Horizon Wireless's automated support assistant speaking with a caller over the phone. "
                "Open with a brief, friendly greeting that mentions Horizon Wireless. "
                "Explain that you'll first confirm the caller's account phone number by using the `ask_for_phone_number` tool. "
                "After the phone number is confirmed, guide the caller through picking a support option with the `ask_for_service_options` tool. "
                "Keep the tone professional, concise, and smoothly transition between steps."
            ),
        )

        self.phone_number: Optional[str] = None

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the caller as Horizon Wireless's virtual assistant, briefly describe your role, "
                "and let them know you'll collect their 10-digit phone number now."
            )
        )

    @function_tool
    async def ask_for_service_options(self, context: RunContext) -> str:
        """Ask user to which telephone service they want to use."""

        if self.phone_number is None:
            raise ToolError(
                "Phone number not provided, you should ask for it via `ask_for_phone_number` tool"
            )

        while True:
            try:
                result = await GetDtmfTask(
                    num_digits=1,
                    chat_ctx=self.chat_ctx.copy(
                        exclude_instructions=True,
                        exclude_function_call=True,
                        exclude_handoff=True,
                    ),
                    extra_instructions=(
                        "Let the caller know they can choose one of three Horizon Wireless services: "
                        "press 1 to hear details about their current plan, press 2 to enable international data roaming, "
                        "or press 3 to explore upgrade options. Prompt them for a single digit and give them a moment to respond."
                    ),
                    repeat_instructions=2,
                )
            except ToolError as e:
                await self.session.generate_reply(instructions=e.message, allow_interruptions=False)
                continue

            if result.user_input == "1":
                return "Your current plan is $100 per month"
            elif result.user_input == "2":
                return "International data roaming is enabled"
            elif result.user_input == "3":
                return "Your new plan is $150 per month"

            await self.session.generate_reply(
                instructions=(
                    "Apologize that the selection wasn't recognized, then remind the caller to enter 1 for plan details, "
                    "2 for international roaming, or 3 for upgrades."
                ),
                allow_interruptions=False,
            )

    @function_tool
    async def ask_for_phone_number(self, context: RunContext) -> str:
        """Ask user to provide a phone number."""
        while True:
            try:
                result = await GetDtmfTask(
                    num_digits=10,
                    chat_ctx=self.chat_ctx.copy(
                        exclude_instructions=True,
                        exclude_function_call=True,
                        exclude_handoff=True,
                    ),
                    ask_for_confirmation=True,
                    extra_instructions=(
                        "Let the caller know you'll record their 10-digit account number and that they can speak or dial it. "
                        "Provide an example such as 415 555 0199, remind them you're keeping their information secure, "
                        "then capture the digits. Read the number back in grouped segments for confirmation and invite them to confirm or re-enter."
                    ),
                    repeat_instructions=2,
                )
            except ToolError as e:
                await self.session.generate_reply(instructions=e.message, allow_interruptions=False)
                continue

            break

        self.phone_number = result.user_input
        return f"User's phone number is {result.user_input}"


@server.rtc_session(agent_name=DTMF_AGENT_DISPATCH_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        vad=silero.VAD.load(),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        stt=inference.STT("deepgram/flux-general"),
        tts=inference.TTS("inworld/inworld-tts-1"),
        turn_detection=MultilingualModel(),
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
        agent=DtmfAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
