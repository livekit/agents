import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm import ChatMessage
from livekit.plugins import deepgram
from openai import OpenAI
from pinecone import Pinecone
import os

load_dotenv()

system_prompt = "" #write the system prompt you need

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")


# TTS: also you can use othe tts vendor
deepgram_tts = deepgram.tts.TTS(
  model="aura-asteria-en",
  api_key=DEEPGRAM_API_KEY,
)

# Configuración de Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "" # use your pinecone index name
name_spaces = "" # use your pinecone namespace
index = pc.Index(name=index_name)


async def get_embedding(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

async def retrieve_relevant_info(user_input: str):
    vector = await get_embedding(user_input)
    response = index.query(vector=vector, namespace=name_spaces, top_k=4, include_metadata=True)
    matches = response.get("matches", [])

    if not matches:
        return ""
    
    retrieved_texts = [match.metadata.get("Description", "") for match in matches]
    print("retrieved_texts: ", retrieved_texts)
    return "\n\n".join(retrieved_texts)

async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    formatted_messages = []
    for msg in chat_ctx.messages:
        if msg.role in ["assistant", "user"]:
            formatted_messages.append(f"- {msg.role}: {msg.content}")
    
    last_messages = "\n".join(formatted_messages)
    retrieved_info = await retrieve_relevant_info(last_messages)
    chat_ctx.messages.append(ChatMessage(role="user", content=retrieved_info))
    

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt)
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini", # use the model you want
                       temperature=0.1),
        tts=deepgram_tts,
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb,
        interrupt_speech_duration = 0.05,
        max_endpointing_delay = 0.15 
    )
    assistant.start(ctx.room)

    await asyncio.sleep(0.5)
    await assistant.say(
        "Hi! I'm Ana.How can i help you today?", 
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))