import asyncio
import logging
import pickle

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, rag, silero, cartesia

annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py

embeddings_dimension = 1536
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

        result = annoy_index.query(user_embedding[0].embedding, n=1)[0]

        print(result.distance)
        paragraph = paragraphs_by_uuid[result.userdata]
        user_msg.content = (
            "Context:\n" + paragraph + "\n\nUser question: " + user_msg.content
        )
        print(user_msg)
        return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    assistant = VoiceAssistant(
        chat_ctx=initial_ctx,
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        will_synthesize_assistant_reply=_will_synthesize_assistant_answer,
        plotting=True,
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
