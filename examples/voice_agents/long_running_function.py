import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero

logger = logging.getLogger("long-running-function")
logger.setLevel(logging.INFO)

load_dotenv()

# This example shows how to interrupt a long running function call.
# A tool execution won't be canceled after the associated agent speech is finished,
# it will continue in the background and send the result to the LLM when it's done.


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=("You are a voice agent."))

    @function_tool
    async def search_web(self, query: str, run_ctx: RunContext) -> str | None:
        """Called when user asks to search the web.
        Args:
            query: The query to search the web for.
        """
        logger.info(f"Searching the web for {query}")

        # wait for the task to finish or the agent speech to be interrupted
        # alternatively, you can disallow interruptions for this function call with
        # run_ctx.disallow_interruptions()

        wait_for_result = asyncio.ensure_future(self._a_long_running_task(query))
        await run_ctx.speech_handle.wait_if_not_interrupted([wait_for_result])

        if run_ctx.speech_handle.interrupted:
            logger.info(f"Interrupted searching the web for {query}")
            # return None to skip the tool reply
            wait_for_result.cancel()
            return None

        output = wait_for_result.result()
        logger.info(f"Done searching the web for {query}, result: {output}")
        return output

    async def _a_long_running_task(self, query: str) -> str:
        """Simulate a long running task."""
        await asyncio.sleep(5)
        return f"I got some results for {query}"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
