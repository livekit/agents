import os

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import MetadataMode

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

load_dotenv()

# check if storage already exists
PERSIST_DIR = "./retrieval-engine-storage"
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
    system_msg = llm.ChatMessage(
        role="system",
        content=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "  # noqa: E501
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."  # noqa: E501
        ),
    )
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(system_msg)

    async def _will_synthesize_assistant_reply(
        assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext
    ):
        ctx_msg = system_msg.copy()
        user_msg = chat_ctx.messages[-1]
        retriever = index.as_retriever()
        nodes = await retriever.aretrieve(user_msg.content)

        ctx_msg.content = "Context that might help answer the user's question:"
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            ctx_msg.content += f"\n\n{node_content}"

        chat_ctx.messages[0] = ctx_msg  # the first message is the system message
        return assistant.llm.chat(chat_ctx=chat_ctx)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        will_synthesize_assistant_reply=_will_synthesize_assistant_reply,
    )
    assistant.start(ctx.room)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
