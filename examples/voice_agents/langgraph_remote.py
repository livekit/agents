#!/usr/bin/env python3

"""
Enhanced LiveKit-LangGraph RemoteGraph Voice Example

This example demonstrates how to use the enhanced LiveKit plugin for LangGraph
RemoteGraph workflows with universal filtering. The plugin ensures only
user-facing responses are spoken while filtering out tool calls, SQL queries,
and intermediate agent outputs.

Features demonstrated:
- Integration with LangGraph RemoteGraph
- Universal filtering of tool calls and intermediate outputs
- Node-based filtering (only supervisor responses in this example)
- Real-time voice interaction with RemoteGraph AI systems

Prerequisites:
1. A deployed LangGraph, either running on localhost or on Langgraph platform
2. OPENAI_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY in .env file
3. LiveKit room credentials (if connecting to specific room)

Usage:
    python examples/langgraph_remote.py dev
"""

import logging
from dotenv import load_dotenv
from langgraph.pregel.remote import RemoteGraph

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.langchain import langgraph_plugin

# Configure logging
logger = logging.getLogger("langgraph-voice-agent")

# Load environment variables from .env file
load_dotenv()

# LangGraph server configuration
LANGGRAPH_SERVER_URL = "http://localhost:2024"  # could be the URL of any deployed LangGraph (localhost or on Langgraph platform)
GRAPH_ID = "multi_agent_workflow"  # Replace with your actual graph ID


def prewarm(proc: JobProcess):
    """Pre-warm function to load models before agent starts.

    This improves response times by loading the Voice Activity Detection (VAD)
    model during worker initialization rather than on first use.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main agent entrypoint that sets up the voice-enabled RemoteGraph system.

    This function:
    1. Connects to the LangGraph RemoteGraph workflow
    2. Configures filtering to only allow supervisor responses
    3. Sets up voice components (STT, TTS, turn detection)
    4. Starts the agent session for voice interaction
    """

    # Connect to LangGraph RemoteGraph
    # This connects to your deployed RemoteGraph workflow
    graph = RemoteGraph(GRAPH_ID, api_key=LANGCHAIN_API_KEY, url=LANGGRAPH_SERVER_URL)

    # Generate a unique thread ID for each session to avoid conflicts
    # Each voice session gets its own conversation thread
    import uuid

    thread_id = str(uuid.uuid4())
    logger.info(f"Starting new session with thread ID: {thread_id}")

    # Configuration for LangGraph execution
    config = {"configurable": {"thread_id": thread_id}}

    # Create the enhanced LLM adapter with filtering
    # langgraph_node="supervisor" means only supervisor responses will be spoken
    # Note: "supervisor" is just an example - replace with your actual node name
    # You can also use:
    # - langgraph_node=["node_a", "node_b"] for multiple nodes
    # - langgraph_node=None for no node filtering (just tool filtering)
    llm_adapter = langgraph_plugin.LLMAdapter(
        graph,
        config=config,
        # filter_messages=False, # Optional, default is True
        langgraph_node=[
            "supervisor",
            "sql_agent",
        ],  # Replace with your actual node name
    )

    # Create the voice agent with comprehensive instructions
    agent = Agent(
        instructions="""
        You are a helpful AI assistant.
        """,
        llm=llm_adapter,
    )

    # Configure the agent session with voice components
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],  # Voice Activity Detection
        # Speech-to-Text: Convert user speech to text
        stt=deepgram.STT(
            model="nova-2",
            language="en",  # High-quality model  # English language
        ),
        # Text-to-Speech: Convert agent responses to speech
        tts=deepgram.TTS(model="aura-asteria-en"),  # Natural-sounding voice
        # Turn Detection: Determine when user has finished speaking
        # MultilingualModel supports multiple languages for end-of-utterance detection
        turn_detection=MultilingualModel(),
    )

    # Start the agent session
    # This begins listening for voice input and responding with voice output
    logger.info("Starting voice agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,  # LiveKit room for voice communication
        room_input_options=RoomInputOptions(
            # to use Krisp background voice cancellation, install livekit-plugins-noise-cancellation
            # and `from livekit.plugins import noise_cancellation`
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room
    await ctx.connect()

    # Start the conversation with a greeting
    # This makes the agent proactively initiate the conversation
    logger.info("Generating initial greeting...")
    await session.generate_reply(instructions="ask the user how they are doing?")


if __name__ == "__main__":
    # CLI entry point for the agent
    # Usage: python examples/multi_agent_voice.py dev
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
