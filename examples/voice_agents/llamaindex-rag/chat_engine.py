from collections.abc import AsyncIterable
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage, MessageRole

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    ModelSettings,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, openai, silero

load_dotenv()

# check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "chat-engine-storage"
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


class DummyLLM(llm.LLM):
    async def chat(self, *args, **kwargs):
        raise NotImplementedError("DummyLLM does not support chat")


class ChatEngineAgent(Agent):
    def __init__(self, index: VectorStoreIndex):
        super().__init__(
            instructions=(
                "You are a voice assistant created by LiveKit. Your interface "
                "with users will be voice. You should use short and concise "
                "responses, and avoiding usage of unpronouncable punctuation."
            ),
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=DummyLLM(),  # use a dummy LLM to enable the pipeline reply
            tts=openai.TTS(),
        )
        self.index = index
        self.chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, llm="default")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        user_msg = chat_ctx.items.pop()
        assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
        user_query = user_msg.text_content
        assert user_query is not None

        llama_chat_messages = [
            ChatMessage(content=msg.text_content, role=MessageRole(msg.role))
            for msg in chat_ctx.items
            if isinstance(msg, llm.ChatMessage)
        ]

        stream = await self.chat_engine.astream_chat(user_query, chat_history=llama_chat_messages)
        async for delta in stream.async_response_gen():
            yield delta


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = ChatEngineAgent(index)
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
