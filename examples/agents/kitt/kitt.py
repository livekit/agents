import os
import asyncio
import logging
from livekit import agents, protocol, rtc
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin, VAD
from livekit.plugins.openai import (WhisperOpenSourceTranscriberPlugin,
                                    ChatGPTPlugin,
                                    ChatGPTMessage,
                                    ChatGPTMessageRole,
                                    TTSPlugin)
from typing import AsyncIterator
import audioread

PROMPT = "You are KITT, a voice assistant in a meeting created by LiveKit. \
          Keep your responses concise while still being friendly and personable. \
          If your response is a question, please append a question mark symbol to the end of it."

OAI_TTS_SAMPLE_RATE = 24000
OAI_TTS_CHANNELS = 1


async def kitt_agent(ctx: agents.JobContext):
    source = rtc.AudioSource(OAI_TTS_SAMPLE_RATE, OAI_TTS_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await ctx.room.local_participant.publish_track(track, options)

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)

        vad_plugin = VADPlugin(
            left_padding_ms=1000, silence_threshold_ms=500)
        stt_plugin = WhisperOpenSourceTranscriberPlugin()
        chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
        tts_plugin = TTSPlugin()

        vad_results = vad_plugin\
            .start(audio_stream)\
            .filter(lambda data: data.type == core.VADPluginResultType.FINISHED)\
            .map(lambda data: data.frames)

        chat_gpt_input_iterator = core.AsyncQueueIterator(
            asyncio.Queue[ChatGPTMessage]())
        stt_results = stt_plugin.start(vad_results)
        chatgpt_result = chatgpt_plugin.start(chat_gpt_input_iterator)
        tts_result = tts_plugin.start(chatgpt_result)

        async def process_stt():
            async for stt_iterator in stt_results:
                complete_stt_result = ""
                async for stt_r in stt_iterator:
                    complete_stt_result += stt_r.text
                    print("STT: ", complete_stt_result)
                    if complete_stt_result.strip() == "":
                        continue
                    msg = ChatGPTMessage(
                        role=ChatGPTMessageRole.user, content=stt_r.text)
                    await chat_gpt_input_iterator.put(msg)

        async def send_audio():
            async for frame_iter in tts_result:
                async for frame in frame_iter:
                    await source.capture_frame(frame)

        asyncio.create_task(process_stt())
        asyncio.create_task(send_audio())

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
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        await job_request.accept(kitt_agent)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=protocol.agent.JobType.JT_ROOM)
    agents.run_app(worker)
