"""
Basic LiveKit voice agent using Lokutor TTS.

Run:
  export LOKUTOR_API_KEY="your-api-key"
  export OPENAI_API_KEY="your-openai-key"
  export DEEPGRAM_API_KEY="your-deepgram-key"
  python examples/basic-agent.py
"""

from livekit.agents import Agent, AgentSession
from livekit.plugins import deepgram, lokutor, openai, silero


async def main():
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=lokutor.TTS(
            voice="F1",
            language="en",
            speed=1.05,
            steps=5,
        ),
        vad=silero.VAD.load(),
    )

    agent = Agent(
        instructions="You are a helpful voice assistant.",
    )

    await session.start(agent=agent)
    await session.wait_for_end()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
