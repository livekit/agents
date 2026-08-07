# LiveKit Agents for Python

Realtime framework for production-grade multimodal and voice AI agents.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for quickstarts, documentation, and examples.

```python
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai

load_dotenv()

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice AI assistant.")
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```
