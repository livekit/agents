import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.llm import function_tool
from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
from livekit.plugins import silero

logger = logging.getLogger("async-tool-report")
logger.setLevel(logging.INFO)

load_dotenv()


# Demonstrates AsyncToolset with a mix of regular and async tools.
# The report generation tool runs in the background — the agent speaks to the user
# while the report is being generated, and narrates the result when it's done.


class ReportToolset(AsyncToolset):
    @function_tool
    async def generate_report(self, ctx: AsyncRunContext, topic: str) -> str:
        """Generate a detailed report on a topic and email it to the user.

        Args:
            topic: The topic to generate a report about.
        """
        logger.info(f"Generating report for {topic}")

        email_result = await GetEmailTask(chat_ctx=ctx.session.history.copy())

        # Set the pending message — returned to the LLM as tool output immediately
        speech_handle = ctx.pending(
            f"Starting report generation on '{topic}'. "
            "This will take a moment, I'll keep you updated and email to you when it's done."
        )

        # Simulate research phase
        await asyncio.sleep(30)
        await speech_handle  # optionally wait for the speech handle to be completed

        # Push a progress update — triggers a new LLM turn
        logger.info("Pushing progress update")
        ctx.update(f"Found 12 documents on '{topic}', compiling the report...", role="system")

        # Simulate compilation and sending
        await asyncio.sleep(30)

        logger.info("Pushing final output")
        final_message = (
            f"Report on '{topic}' has been generated and sent to {email_result.email_address}."
        )

        # Return the final result will create another function call and output pair to the LLM
        # optionally, you can call ctx.update(final_message, role="assistant") to push the final message to the LLM
        # and return None to avoid creating another function call and output to the LLM
        return final_message


class ReportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful research assistant that generates reports. "
                "You would interact with users via voice. "
                "with that in mind keep your responses concise and to the point. "
                "do not use emojis, asterisks, markdown, or other special characters in your responses. "
                "You are curious and friendly, and have a sense of humor. "
                "you will speak english to the user"
            ),
            tools=[ReportToolset(id="reports")],
        )

    async def on_enter(self):
        self.session.generate_reply(
            # instructions="Greet the user and ask what topic they'd like a report on."
            user_input="Generate a report on the topic of AI"
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=ReportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
