import os
import asyncio
import livekit.rtc as rtc
from livekit import agents
from livekit import plugins
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

        vad_results = vad_plugin\
            .start(audio_stream)\
            .filter(lambda data: data.type == plugins.VADPluginEventType.FINISHED)\
            .map(lambda data: data.frames)\
            .unwrap()
        stt_results = stt_plugin.start(vad_results)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))

if __name__ == "__main__":
    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        job_request.accept(kitt_agent)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    asyncio.run(worker.start())
