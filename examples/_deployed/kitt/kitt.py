import asyncio
import copy
import logging
from collections import deque

from livekit import agents, rtc
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatRole,
)
from livekit.agents.voice_assistant import AssistantContext, VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero

MAX_IMAGES = 3


class AssistantFnc(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        desc="Called when asked to look at something or see something or evaulate a visual using your vision capabilities"
    )
    async def image(self):
        pass


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                text=(
                    "You are a funny bot created by LiveKit. Your interface with users will be voice. "
                    "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
                ),
            )
        ]
    )

    gpt = openai.LLM()
    latest_image: rtc.VideoFrame | None = None
    img_msg_queue: deque[agents.llm.ChatMessage] = deque()
    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(encoding="pcm_44100"),
        chat_ctx=initial_ctx,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer_from_text(text: str):
        chat_ctx = copy.deepcopy(assistant.chat_context)
        chat_ctx.messages.append(ChatMessage(role=ChatRole.USER, text=text))

        stream = await gpt.chat(chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if not msg.message:
            return

        asyncio.create_task(_answer_from_text(msg.message))

    async def respond_to_image():
        nonlocal latest_image, img_msg_queue
        if not latest_image:
            return
        initial_ctx.messages[-1].images.append(
            agents.llm.ChatImage(image=latest_image, dimensions=(128, 128))
        )
        img_msg_queue.append(initial_ctx.messages[-1])
        if len(img_msg_queue) >= MAX_IMAGES:
            img_msg_queue.popleft()

        stream = await gpt.chat(initial_ctx)
        await assistant.say(stream)

    async def video_track_process():
        nonlocal latest_image
        while True:
            await asyncio.sleep(0.1)
            participants = [ctx.room.participants[p] for p in ctx.room.participants]
            if len(participants) == 0:
                continue
            p = participants[0]
            tracks = [
                p.tracks[t]
                for t in p.tracks
                if p.tracks[t].kind == rtc.TrackKind.KIND_VIDEO
            ]
            if len(tracks) == 0 or not tracks[0].track:
                continue
            async for event in rtc.VideoStream(tracks[0].track):
                latest_image = event.frame

    @assistant.on("function_calls_finished")
    def _function_calls_done(ctx: AssistantContext):
        asyncio.ensure_future(respond_to_image())

    assistant.start(ctx.room)

    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)
    await video_track_process()


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
