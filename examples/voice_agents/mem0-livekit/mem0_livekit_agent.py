import logging
from dotenv import load_dotenv

from mem0 import AsyncMemoryClient

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    ChatContext,
    ChatMessage,
    RoomInputOptions,
    Agent,
    AgentSession,
)
from livekit.plugins import openai, silero, deepgram, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# User ID for RAG data in Mem0
RAG_USER_ID = "livekit-mem0"
mem0_client = AsyncMemoryClient()

class MemoryEnabledAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation) with Mem0.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant.
                You are a travel guide named George and will help the user to plan a travel trip of their dreams.
                You should help the user plan for various adventures like work retreats, family vacations or solo backpacking trips.
                You should be careful to not suggest anything that would be dangerous, illegal or inappropriate.
                You can remember past interactions and use them to inform your answers.
                Use semantic memory retrieval to provide contextually relevant responses.
            """,
        )
        self._seen_results = set()  # Track previously seen result IDs
        logger.info(f"Mem0 Agent initialized. Using user_id: {RAG_USER_ID}")

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance."
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Persist the user message in Mem0
        try:
            logger.info(f"Adding user message to Mem0: {new_message.text_content}")
            add_result = await mem0_client.add(
                [{"role": "user", "content": new_message.text_content}],
                user_id=RAG_USER_ID
            )
            logger.info(f"Mem0 add result (user): {add_result}")
        except Exception as e:
            logger.warning(f"Failed to store user message in Mem0: {e}")

        # RAG: Retrieve relevant context from Mem0 and inject as assistant message
        try:
            logger.info("About to await mem0_client.search for RAG context")
            search_results = await mem0_client.search(
                new_message.text_content,
                user_id=RAG_USER_ID,
            )
            logger.info(f"mem0_client.search returned: {search_results}")
            if search_results and isinstance(search_results, list):
                context_parts = []
                for result in search_results:
                    paragraph = result.get("memory") or result.get("text")
                    if paragraph:
                        source = "mem0 Memories"
                        if "from [" in paragraph:
                            source = paragraph.split("from [")[1].split("]")[0]
                            paragraph = paragraph.split("]")[1].strip()
                        context_parts.append(f"Source: {source}\nContent: {paragraph}\n")
                if context_parts:
                    full_context = "\n\n".join(context_parts)
                    logger.info(f"Injecting RAG context: {full_context}")
                    turn_ctx.add_message(role="assistant", content=full_context)
                    await self.update_chat_ctx(turn_ctx)
        except Exception as e:
            logger.warning(f"Failed to inject RAG context from Mem0: {e}")

        await super().on_user_turn_completed(turn_ctx, new_message)

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash",),
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=MemoryEnabledAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly as George the travel guide and ask how you can help them plan their next adventure.",
        allow_interruptions=True
    )

# Run the application
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
