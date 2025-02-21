import logging
import random

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simplified log format for Rich
    datefmt="[%X]",
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize the RAG system
livekit_expert = RAGSystem(collection_name='about-livekit', data_source_path='data', create_if_not_exists=True)

# Prepare the initial Chat Context for the assistant
initial_ctx = llm.ChatContext().append(
    role="system",
    text=(  # System message for initial setup
        "You are a voice assistant powered by LiveKit. "
        "You can assist users with information from a specific dataset. "
        "Please provide concise responses, and avoid any unpronounceable symbols or punctuation."
    ),
)

# Define the query function using RAG methods
async def query_rag_system(query: str) -> str:
    try:
        logging.info(f"Received query: {query}")
        result = livekit_expert.process(query=query)  # Use the 'process' method of RAGSystem for querying
        if result:
            logging.info(f"Query result: {result}")
            return str(result)
        else:
            return "Sorry, I couldn't find any relevant information."
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "Sorry, something went wrong while processing your query."

# Define the entrypoint for the LiveKit voice assistant
async def entrypoint(ctx: JobContext):
    # Connect to the audio stream and initialize the assistant
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Initialize the function context for RAG queries
    fnc_ctx = llm.FunctionContext()

    # Register the query function
    @fnc_ctx.ai_callable(description="When user asks for specific or external information")
    async def query_info(query: str) -> str:
        agent = AgentCallContext.get_current().agent

        if not agent.chat_ctx.messages or agent.chat_ctx.messages[-1].role != "assistant":
            # Skip if the assistant already said something
            filler_messages = [
                "Let me look that up for you.",
                "Let me search the database for that.",
            ]
            message = random.choice(filler_messages)
            logging.info(f"saying filler message: {message}")

            # Add filler message to the chat context
            await agent.say(message, add_to_chat_ctx=True)

        # Now perform the query
        result = await query_rag_system(query)
        return result

    # Create the voice assistant with LiveKit
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),  # Voice activity detection
        stt=deepgram.STT(),  # Speech-to-text using Deepgram
        llm=openai.LLM.with_groq(model='llama-3.3-70b-versatile'),
        tts=deepgram.TTS(model="aura-hera-en"),
        chat_ctx=initial_ctx,  # Initial chat context
        fnc_ctx=fnc_ctx,  # Function context for RAG queries
    )

    # Start the assistant and prompt for interaction
    assistant.start(ctx.room)
    await assistant.say("Hello, how can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    # Run the LiveKit assistant app
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


