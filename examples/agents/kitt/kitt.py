import asyncio
import livekit.rtc as rtc
from livekit import agents
from livekit.plugins.vad import VADPlugin, VAD
from livekit.plugins.google import SpeechRecognitionPlugin
from livekit.plugins.openai import WhisperOpenSourceTranscriberPlugin, ChatGPTPlugin, ChatGPTMessage, ChatGPTMessageRole
from typing import AsyncIterator

PROMPT = "You are KITT, a voice assistant in a meeting created by LiveKit. \
          Keep your responses concise while still being friendly and personable. \
          If your response is a question, please append a question mark symbol to the end of it."


async def kitt_agent(ctx: agents.JobContext):

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_plugin = VADPlugin(
            left_padding_ms=250, silence_threshold_ms=500)
        stt_plugin = WhisperOpenSourceTranscriberPlugin()
        chatgpt_plugin = ChatGPTPlugin(message_capacity=5, prompt=PROMPT)

        async def vad_result_loop(queue: AsyncIterator[VAD.Event]):
            async for event in queue:
                if event.type == "voice_finished":
                    frames = event.frames
                    stt_plugin.push(frames)

        async def stt_result_loop(queue: AsyncIterator[AsyncIterator[agents.STTPlugin.Event]]):
            async for event_iterator in queue:
                async for event in event_iterator:
                    chatgpt_plugin.push(ChatGPTMessage(
                        content=event.text, role=ChatGPTMessageRole.user))

        async def chatgpt_result_loop(queue: AsyncIterator[AsyncIterator[str]]):
            async for event_iterator in queue:
                async for event in event_iterator:
                    asyncio.create_task(
                        ctx.room.local_participant.publish_data(event))

        async def tts_result_loop(queue: AsyncIterator[AsyncIterator[rtc.AudioFrame]]):
            async for event_iterator in queue:
                async for event in event_iterator:
                    pass

        asyncio.create_task(stt_result_loop(queue=stt_plugin.stream()))
        asyncio.create_task(vad_result_loop(queue=vad_plugin.stream()))
        asyncio.create_task(chatgpt_result_loop(
            queue=chatgpt_plugin.stream()))

        async for frame in audio_stream:
            vad_plugin.push(frame)

    @ctx.room.on("data_received")
    def on_data_received(data: str, participant: rtc.RemoteParticipant):
        print(f"Data Received: {data}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))
