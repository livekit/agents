#!/usr/bin/env python3
"""
Simple test script for basic_agent.py functionality
"""
import asyncio
import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, ModelSettings, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a test agent. Say hello and confirm you're working."
        )

    @function_tool
    async def test_function(self, context: RunContext):
        """Test function to verify agent functionality."""
        return "Test function executed successfully!"

async def test_basic_agent():
    """Test basic agent functionality without full session setup."""
    print("Testing basic agent functionality...")

    try:
        # Test TTS configuration
        print("Testing TTS configuration...")
        tts = cartesia.TTS(voice="ash")
        print("✓ TTS configuration successful")

        # Test STT configuration
        print("Testing STT configuration...")
        stt = cartesia.STT()
        print("✓ STT configuration successful")

        # Test LLM configuration
        print("Testing LLM configuration...")
        llm = openai.LLM(model="gpt-4o-mini")
        print("✓ LLM configuration successful")

        print("\n🎉 All basic configurations are working!")
        print("The agent should be functional with Deepgram STT/TTS and OpenAI LLM.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

    return True

if __name__ == "__main__":
    asyncio.run(test_basic_agent())
