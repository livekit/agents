"""
Fullstack RAG Video Platform - Main Agent
A masterpiece AI agent with RAG memory, video streaming, and persistent context.
"""

import asyncio
import logging
from typing import Annotated
from datetime import datetime

from livekit import agents
from livekit.agents import (
    Agent,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice import VoiceSession
from livekit.plugins import deepgram, openai, silero, elevenlabs

from config import Config
from rag_engine import RAGEngine
from memory_manager import MemoryManager
from tools import create_rag_tools, create_memory_tools
from video_handler import VideoHandler

logger = logging.getLogger("rag-video-agent")
logger.setLevel(logging.INFO)


class RAGVideoAgent:
    """Advanced AI agent with RAG capabilities and video streaming."""

    def __init__(self, config: Config):
        self.config = config
        self.rag_engine: RAGEngine | None = None
        self.memory_manager: MemoryManager | None = None
        self.video_handler: VideoHandler | None = None

    async def initialize(self):
        """Initialize RAG engine, memory manager, and video handler."""
        logger.info("Initializing RAG Video Agent...")

        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            vector_db_url=self.config.vector_db_url,
            embedding_model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        await self.rag_engine.initialize()
        logger.info("✓ RAG Engine initialized")

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            db_path=self.config.memory_db_path,
            window_size=self.config.memory_window,
        )
        await self.memory_manager.initialize()
        logger.info("✓ Memory Manager initialized")

        # Initialize video handler
        self.video_handler = VideoHandler(
            avatar_provider=self.config.avatar_provider,
            video_fps=self.config.video_fps,
        )
        logger.info("✓ Video Handler initialized")

        logger.info("🚀 RAG Video Agent ready!")

    async def process_rag_query(
        self, query: str, user_id: str
    ) -> Annotated[str, "The answer based on retrieved documents"]:
        """
        Query the RAG system to retrieve relevant information.

        Args:
            query: The user's question
            user_id: The user identifier for context

        Returns:
            Relevant information from the knowledge base
        """
        if not self.rag_engine:
            return "RAG system not initialized"

        try:
            # Get conversation history for context
            history = await self.memory_manager.get_recent_history(user_id, limit=5)

            # Perform RAG retrieval
            results = await self.rag_engine.query(
                query=query,
                top_k=self.config.top_k,
                conversation_context=history,
            )

            return results

        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return f"I encountered an error retrieving information: {str(e)}"

    async def save_memory(
        self, user_id: str, message: str, role: str = "user"
    ) -> Annotated[str, "Confirmation of memory saved"]:
        """
        Save important information to long-term memory.

        Args:
            user_id: The user identifier
            message: The information to remember
            role: The role (user/assistant)

        Returns:
            Confirmation message
        """
        if not self.memory_manager:
            return "Memory system not initialized"

        try:
            await self.memory_manager.save_message(
                user_id=user_id,
                message=message,
                role=role,
                timestamp=datetime.utcnow(),
            )
            return "Memory saved successfully"

        except Exception as e:
            logger.error(f"Memory save error: {e}")
            return f"Error saving memory: {str(e)}"

    async def get_conversation_summary(
        self, user_id: str
    ) -> Annotated[str, "Summary of conversation history"]:
        """
        Get a summary of the conversation history with the user.

        Args:
            user_id: The user identifier

        Returns:
            Conversation summary
        """
        if not self.memory_manager:
            return "Memory system not initialized"

        try:
            summary = await self.memory_manager.get_summary(user_id)
            return summary or "No previous conversation history found"

        except Exception as e:
            logger.error(f"Summary retrieval error: {e}")
            return f"Error retrieving summary: {str(e)}"


async def entrypoint(ctx: JobContext):
    """Main entry point for the RAG video agent."""
    logger.info(f"Starting RAG Video Agent for room: {ctx.room.name}")

    # Load configuration
    config = Config()

    # Initialize agent
    rag_agent = RAGVideoAgent(config)
    await rag_agent.initialize()

    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Get user ID from room metadata
    user_id = ctx.room.name  # Use room name as user ID for now

    # Load conversation history
    history = await rag_agent.memory_manager.get_recent_history(
        user_id, limit=config.memory_window
    )

    # Build system instructions with context
    system_instructions = f"""You are an advanced AI assistant with access to a comprehensive knowledge base and conversation memory.

Your capabilities:
1. **RAG Knowledge Retrieval**: You can search through uploaded documents to answer questions accurately
2. **Persistent Memory**: You remember previous conversations and context across sessions
3. **Video Interaction**: You can see and respond to visual input from the user
4. **Multi-modal Communication**: You support text, voice, and video interactions

Guidelines:
- Always use the RAG system when users ask about specific topics in the knowledge base
- Reference previous conversations when relevant
- Be conversational, helpful, and concise
- If you don't know something, admit it and offer to search the knowledge base
- Save important user preferences and information to long-term memory
- Provide outstanding quality responses with accurate, well-structured information

Current user: {user_id}
Conversation history: {"Available - " + str(len(history)) + " previous messages" if history else "New conversation"}
"""

    # Configure LLM based on provider
    if config.llm_provider == "openai":
        llm_instance = openai.LLM(
            model=config.llm_model,
            temperature=0.7,
        )
    elif config.llm_provider == "anthropic":
        from livekit.plugins import anthropic

        llm_instance = anthropic.LLM(
            model=config.llm_model,
            temperature=0.7,
        )
    else:
        # Default to OpenAI
        llm_instance = openai.LLM(model="gpt-4-turbo")

    # Configure STT
    if config.stt_provider == "deepgram":
        stt_instance = deepgram.STT(
            model="nova-2-general",
            language="en",
        )
    else:
        stt_instance = deepgram.STT()

    # Configure TTS
    if config.tts_provider == "elevenlabs":
        tts_instance = elevenlabs.TTS(
            model="eleven_turbo_v2_5",
            voice="Rachel",
        )
    elif config.tts_provider == "openai":
        tts_instance = openai.TTS(voice="nova")
    else:
        tts_instance = elevenlabs.TTS()

    # Create AI agent with RAG and memory tools
    agent = Agent(
        instructions=system_instructions,
        llm=llm_instance,
        tools=[
            llm.FunctionTool(
                name="query_knowledge_base",
                description="Search the knowledge base for relevant information. Use this when users ask about specific topics covered in uploaded documents.",
                callable=lambda query: rag_agent.process_rag_query(query, user_id),
            ),
            llm.FunctionTool(
                name="save_to_memory",
                description="Save important information to long-term memory. Use this for user preferences, important facts, or context to remember.",
                callable=lambda message: rag_agent.save_memory(
                    user_id, message, "assistant"
                ),
            ),
            llm.FunctionTool(
                name="get_conversation_history",
                description="Retrieve a summary of previous conversations with this user.",
                callable=lambda: rag_agent.get_conversation_summary(user_id),
            ),
        ],
    )

    # Create voice session
    voice_session = VoiceSession(
        vad=silero.VAD.load(),
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        turn_detector=None,  # Use VAD for turn detection
    )

    # Start the voice session
    await voice_session.start(agent=agent, room=ctx.room)

    # Handle video if enabled
    if config.enable_video and rag_agent.video_handler:
        await rag_agent.video_handler.start(ctx.room)

    # Log conversation messages for memory
    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(
            f"Track published: {publication.sid} by {participant.identity}"
        )

    # Save messages to memory
    async def save_conversation_turn(message: str, role: str):
        try:
            await rag_agent.memory_manager.save_message(
                user_id=user_id,
                message=message,
                role=role,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    # Hook into LLM events to save conversations
    @voice_session.on("user_speech_committed")
    def on_user_speech(msg: str):
        asyncio.create_task(save_conversation_turn(msg, "user"))

    @voice_session.on("agent_speech_committed")
    def on_agent_speech(msg: str):
        asyncio.create_task(save_conversation_turn(msg, "assistant"))

    logger.info("✓ Voice session started successfully")

    # Keep the agent running
    await asyncio.Event().wait()


if __name__ == "__main__":
    # Run the agent with CLI
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=agents.WorkerType.ROOM,
        )
    )
