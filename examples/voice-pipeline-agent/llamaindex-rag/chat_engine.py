import os

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, llama_index, openai, silero
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode

load_dotenv()

# check if storage already exists
PERSIST_DIR = "./chat-engine-storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=llama_index.LLM(chat_engine=chat_engine),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
