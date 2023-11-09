import os
import asyncio
import logging
from livekit import agents, protocol, rtc
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin, VAD
from livekit.plugins.openai import WhisperOpenSourceTranscriberPlugin, ChatGPTPlugin, ChatGPTMessage, ChatGPTMessageRole
from livekit.plugins.elevenlabs import ElevenLabsTTSPlugin
from typing import AsyncIterator

PROMPT = "You are KITT, a voice assistant in a meeting created by LiveKit. \
          Keep your responses concise while still being friendly and personable. \
          If your response is a question, please append a question mark symbol to the end of it."


async def kitt_agent(ctx: agents.JobContext):

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)

        vad_plugin = VADPlugin(
            left_padding_ms=1000, silence_threshold_ms=500)
        stt_plugin = WhisperOpenSourceTranscriberPlugin()
        chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
        tts_plugin = ElevenLabsTTSPlugin()

        vad_results = vad_plugin\
            .start(audio_stream)\
            .filter(lambda data: data.type == core.VADPluginResultType.FINISHED)\
            .map(lambda data: data.frames)

        stt_results = stt_plugin.start(vad_results)

        async for stt_iterator in stt_results:
            print("STT: ", stt_iterator)
            complete_stt_result = ""
            async for stt_r in stt_iterator:
                complete_stt_result += stt_r.text
                chatgpt_results = chatgpt_plugin.start(ChatGPTMessage(
                    role=ChatGPTMessageRole.user, content=complete_stt_result))

                async for chatgpt_r in chatgpt_results:
                    logging.info(f"ChatGPT: {chatgpt_r}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))

    @ctx.room.on('track_published')
    def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        publication.set_subscribed(True)

    for participant in ctx.room.participants.values():
        for publication in participant.tracks.values():
            if publication.kind != rtc.TrackKind.KIND_AUDIO:
                continue

            publication.set_subscribed(True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        await job_request.accept(kitt_agent)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=protocol.agent.JobType.JT_ROOM)
    agents.run_app(worker)
