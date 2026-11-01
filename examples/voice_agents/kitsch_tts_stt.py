"""Run with LIVEKIT_URL/API_KEY/API_SECRET, KITSCH_API_KEY and LIVEKIT_LLM set."""

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import kitsch

server = AgentServer()


@server.rtc_session(agent_name="kitsch")
async def entrypoint(ctx: JobContext) -> None:
    tts = kitsch.TTS(voice_id="minjun", language_code="ko")
    stt = kitsch.STT(language="ko")
    session = AgentSession(stt=stt, tts=tts, llm=os.environ["LIVEKIT_LLM"])

    async def close() -> None:
        await session.aclose()
        await stt.aclose()
        await tts.aclose()

    ctx.add_shutdown_callback(close)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="한국어로 짧고 명확하게 답하세요."),
    )
    await ctx.connect()
    await session.say("안녕하세요. 키치랩스입니다.")


if __name__ == "__main__":
    cli.run_app(server)
