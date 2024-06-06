import asyncio
import copy
import logging
from collections import deque
from typing import Annotated

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

NO_IMAGE_MESSAGE_SIP = "I'm sorry, I can't see anything when you're calling from a phone. Try the web demo."
NO_IMAGE_MESSAGE_GENERIC = (
    "I'm sorry, I don't have an image to process. Are you publishing your video?"
)


class AssistantFnc(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        desc="Called when asked to evaluate something that would require vision capabilities."
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(desc="The user message that triggered this function"),
        ],
    ):
        ctx = AssistantContext.get_current()
        ctx.store_metadata("user_msg", user_msg)


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

    gpt = openai.LLM(
        model="gpt-4o",
    )
    latest_image: rtc.VideoFrame | None = None
    img_msg_queue: deque[agents.llm.ChatMessage] = deque()
    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(encoding="pcm_44100"),
        fnc_ctx=AssistantFnc(),
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

    async def respond_to_image(user_msg: str):
        nonlocal latest_image, img_msg_queue, initial_ctx
        if not latest_image:
            # We've setup sip-created rooms to have the 'sip' prefix
            sip = ctx.room.name.startswith("sip")
            msg = NO_IMAGE_MESSAGE_SIP if sip else NO_IMAGE_MESSAGE_GENERIC
            await assistant.say(msg)
            return

        initial_ctx.messages.append(
            agents.llm.ChatMessage(
                role=agents.llm.ChatRole.USER,
                text=user_msg,
                images=[agents.llm.ChatImage(image=latest_image)],
            )
        )
        img_msg_queue.append(initial_ctx.messages[-1])
        if len(img_msg_queue) >= MAX_IMAGES:
            msg = img_msg_queue.popleft()
            msg.images = []

        stream = await gpt.chat(initial_ctx)
        await assistant.say(stream, allow_interruptions=True)

    async def get_human_video_track():
        remote_video_tracks = [
            t_pub.track
            for _, p in ctx.room.participants.items()
            for _, t_pub in p.tracks.items()
            if t_pub.track is not None
            and t_pub.kind == rtc.TrackKind.KIND_VIDEO
            and isinstance(t_pub.track, rtc.RemoteVideoTrack)
        ]

        if len(remote_video_tracks) > 0:
            return remote_video_tracks[0]

        track_future = asyncio.Future[rtc.RemoteVideoTrack]()

        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if isinstance(track, rtc.RemoteVideoTrack):
                track_future.set_result(track)

        ctx.room.on("track_subscribed", on_track_subscribed)

        video_track = await track_future
        ctx.room.off("track_subscribed", on_track_subscribed)
        return video_track

    async def process_video_track():
        nonlocal latest_image
        while True:
            video_track = await get_human_video_track()
            async for event in rtc.VideoStream(video_track):
                latest_image = event.frame

    @assistant.on("function_calls_finished")
    def _function_calls_done(ctx: AssistantContext):
        user_msg = ctx.get_metadata("user_msg")
        if not user_msg:
            return
        asyncio.ensure_future(respond_to_image(user_msg))

    assistant.start(ctx.room)

    await asyncio.sleep(0.5)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)
    await process_video_track()


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
