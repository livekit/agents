import asyncio
import logging

from livekit.agents import AutoSubscribe, Worker
from livekit.agents.llm import OpenAIChat
from livekit.agents.stt import DeepGramSTT
from livekit.agents.tts import ElevenLabsTTS

from filler_filter import FilteredVoiceAssistant
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")


class IntelligentVoiceAgent(Worker):
    """
    Main agent worker with intelligent interruption handling
    """

    async def __start(self, ctx):
        logger.info("Starting IntelligentVoiceAgent...")

        # Initialize components
        llm = OpenAIChat(
            model="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant. Be concise and natural in conversation."
        )

        stt = DeepGramSTT()
        tts = ElevenLabsTTS()

        # Create filtered voice assistant
        assistant = FilteredVoiceAssistant(
            llm=llm,
            stt=stt,
            tts=tts,
            enable_habits=False
        )

        # Start the assistant
        await assistant.start(ctx.room, AutoSubscribe.SUBSCRIBE_ALL)

        logger.info("IntelligentVoiceAgent started successfully")
        logger.info(f"Configured with {len(config.ignored_words)} ignored words")
        logger.info(f"Confidence threshold: {config.confidence_threshold}")

        # Keep the agent running
        while True:
            await asyncio.sleep(1)

    async def run(self, ctx):
        try:
            await self.__start(ctx)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            raise


if __name__ == "__main__":
    # Create and run the worker
    worker = IntelligentVoiceAgent()
    worker.run()