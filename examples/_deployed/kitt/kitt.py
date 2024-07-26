import asyncio
from collections import deque
from typing import Annotated, List

from livekit import agents, rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero

MAX_IMAGES = 3
NO_IMAGE_MESSAGE_GENERIC = (
    "I'm sorry, I don't have an image to process. Are you publishing your video?"
)


def prewarm(p: JobProcess):
    vad = silero.VAD.load()
    p.userdata["vad"] = vad


class AssistantFnc(agents.llm.FunctionContext):
    @agents.llm.ai_callable(description=agents.llm.USE_DOCSTRING)
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        """
        Called when asked to evaluate something that would require vision capabilities.
        """
        return user_msg


async def entrypoint(ctx: JobContext):
    video_track_future = asyncio.Future[rtc.RemoteVideoTrack]()

    def on_sub(track: rtc.Track, *_):
        if isinstance(track, rtc.RemoteVideoTrack):
            video_track_future.set_result(track)

    ctx.room.on("track_subscribed", on_sub)

    await ctx.connect()

    sip = ctx.room.name.startswith("sip")
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a funny bot created by LiveKit. Your interface with users will be voice. "
                    "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")
    latest_image: rtc.VideoFrame | None = None
    img_msg_queue: deque[agents.llm.ChatMessage] = deque()
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(encoding="pcm_44100"),
        fnc_ctx=None if sip else AssistantFnc(),
        chat_ctx=initial_ctx,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer_from_text(text: str):
        chat_ctx = assistant.chat_ctx.copy()
        chat_ctx.messages.append(ChatMessage(role="user", content=text))

        stream = gpt.chat(chat_ctx=chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if not msg.message:
            return

        asyncio.create_task(_answer_from_text(msg.message))

    async def respond_to_image(user_msg: str):
        nonlocal latest_image, img_msg_queue, initial_ctx
        if not latest_image:
            await assistant.say(NO_IMAGE_MESSAGE_GENERIC)
            return

        initial_ctx.messages.append(
            agents.llm.ChatMessage(
                role="user",
                content=[user_msg, agents.llm.ChatImage(image=latest_image)],
            )
        )
        img_msg_queue.append(initial_ctx.messages[-1])
        if len(img_msg_queue) >= MAX_IMAGES:
            msg = img_msg_queue.popleft()
            if isinstance(msg.content, list):
                msg.content = [
                    c for c in msg.content if not isinstance(c, agents.llm.ChatImage)
                ]

        stream = gpt.chat(chat_ctx=initial_ctx)
        await assistant.say(stream, allow_interruptions=True)

    @assistant.on("function_calls_finished")
    def _function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        user_msg = called_functions[0].call_info.arguments["user_msg"]
        asyncio.ensure_future(respond_to_image(user_msg))

    assistant.start(ctx.room)

    await asyncio.sleep(0.5)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await video_track_future
        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, num_idle_processes=1
        )
    )
