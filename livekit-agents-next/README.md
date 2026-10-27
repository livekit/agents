# LiveKit Agents for Python

Realtime framework for production-grade multimodal and voice AI agents.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for quickstarts, documentation, and examples.

```python
from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, JobContext, VoiceSession, cli
from livekit.plugins import openai

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = VoiceSession(llm=openai.realtime.RealtimeModel(voice="coral"))
    await session.start(
        room=ctx.room, agent=Agent(instructions="You are a helpful voice AI assistant.")
    )
    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    cli.run_app(server)
```
