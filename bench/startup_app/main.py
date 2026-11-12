"""The smallest voice agent: the framework's own startup cost, without an app's."""

from livekit.agents import Agent, AgentHandler, AgentServer, JobContext, VoiceSession, cli

assistant = Agent(instructions="You are a helpful assistant.")

handler = AgentHandler()


@handler.rtc_session()
async def call(ctx: JobContext) -> None:
    session = VoiceSession(stt="deepgram/nova-3", llm="openai/gpt-4.1-mini", tts="cartesia/sonic-2")
    await session.start(agent=assistant, room=ctx.room)


server = AgentServer(agent_name="startup-bench")
server.add(handler)

if __name__ == "__main__":
    cli.run_app(server)
