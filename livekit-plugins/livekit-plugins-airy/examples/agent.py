from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import airy


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a concise, helpful Korean voice assistant.")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user briefly in Korean.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="ko"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=airy.TTS(language="ko"),
    )
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
