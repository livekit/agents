import asyncio
import logging
import pickle

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, cartesia, rag

annoy_index = rag.annoy.AnnoyIndex.load("vdb_data") # see build_data.py

# open my_data.pkl
with open("my_data.pkl", "rb") as f:
    indexed_paragraphs = pickle.load(f)

async def entrypoint(ctx: JobContext):

    async def _will_create_llm_stream(
        assistant: VoiceAssistant, chat_ctx: llm.ChatContext
    ) -> llm.LLMStream:
        user_msg = chat_ctx.messages[-1].content
        user_embedding = await openai.create_embeddings(
            input=user_msg,
            model="text-embedding-3-small",
            dimensions=512,
        )

        uuids = annoy_index.query(user_embedding[0].embedding, n=1)
        uuid = uuids[0]
        
        entire_paragraph = indexed_paragraphs[uuid[0]]
        print(entire_paragraph["text"])


        return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)


    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        will_create_llm_stream=_will_create_llm_stream,
    )
    assistant.start(ctx.room)


    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
