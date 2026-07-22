import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.beta.workflows import GetCreditCardTask

logger = logging.getLogger("credit-card-collection")

load_dotenv()


class CreditCardCollectionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a payment assistant. Never repeat, summarize, or expose payment "
                "details supplied by the user."
            )
        )

    async def on_enter(self) -> None:
        await GetCreditCardTask(require_confirmation=True)

        # Keep PCI-sensitive result fields out of logs and follow-up model context.
        logger.info("credit-card collection completed")
        await self.session.generate_reply(
            instructions=(
                "Tell the user their payment details were collected successfully. "
                "Do not repeat any payment details."
            )
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-5.4"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    await session.start(agent=CreditCardCollectionAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
