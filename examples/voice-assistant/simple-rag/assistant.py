import asyncio
import logging
import pickle

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import cartesia, deepgram, openai, rag, silero

annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py

embeddings_dimension = 512
with open("my_data.pkl", "rb") as f:
    paragraphs_by_uuid = pickle.load(f)


async def entrypoint(ctx: JobContext):
    async def _will_synthesize_assistant_answer(
        assistant: VoiceAssistant, chat_ctx: llm.ChatContext
    ):
        # copy so we don't edit the original chat_ctx inside the assistant
        chat_ctx = chat_ctx.copy()
        user_msg = chat_ctx.messages[-1]
        user_embedding = await openai.create_embeddings(
            input=user_msg.content,  # type: ignore
            model="text-embedding-3-small",
            dimensions=embeddings_dimension,
        )

        uuid = annoy_index.query(user_embedding[0].embedding, n=1)[0]
        paragraph = paragraphs_by_uuid[uuid]

        user_msg.content = paragraph + "\n\n" + user_msg.content
        return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        will_synthesize_assistant_reply=_will_synthesize_assistant_answer,
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
