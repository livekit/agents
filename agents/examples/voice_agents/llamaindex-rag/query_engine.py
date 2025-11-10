from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, openai, silero

load_dotenv()

# check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"
if not PERSIST_DIR.exists():
    # load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


@llm.function_tool
async def query_info(query: str) -> str:
    """Get more information about a specific topic"""
    query_engine = index.as_query_engine(use_async=True)
    res = await query_engine.aquery(query)
    print("Query result:", res)
    return str(res)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = Agent(
        instructions=(
            "You are a voice assistant created by LiveKit. Your interface "
            "with users will be voice. You should use short and concise "
            "responses, and avoiding usage of unpronouncable punctuation."
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        tools=[query_info],
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
