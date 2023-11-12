import os
import asyncio
import logging
from openai import AsyncOpenAI
from livekit import agents, protocol, rtc
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin
from livekit.plugins.openai import (WhisperAPITranscriber,
                                    ChatGPTPlugin,
                                    ChatGPTMessage,
                                    ChatGPTMessageRole,
                                    TTSPlugin)
from typing import AsyncIterator
from enum import Enum

PROMPT = "You are KITT, a voice assistant in a meeting created by LiveKit. \
          Keep your responses concise while still being friendly and personable. \
          If your response is a question, please append a question mark symbol to the end of it."

OAI_TTS_SAMPLE_RATE = 24000
OAI_TTS_CHANNELS = 1


AgentState = Enum("AgentState", "LISTENING, SPEAKING")


async def kitt_agent(ctx: agents.JobContext):
    source = rtc.AudioSource(OAI_TTS_SAMPLE_RATE, OAI_TTS_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await ctx.room.local_participant.publish_track(track, options)

    state = [AgentState.LISTENING, 0]  # state, sequence number of interest

    async def process_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        input_iterator = core.PluginIterator.create(audio_stream)

        vad_plugin = VADPlugin(
            left_padding_ms=1000, silence_threshold_ms=500)
        stt_plugin = WhisperAPITranscriber()
        chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
        complete_sentence_plugin = core.CompleteSentencesPlugin()
        tts_plugin = TTSPlugin()

        def vad_state_changer(vad_result: core.VADPluginResult, metadata: core.PluginIterator.ResultMetadata):
            if vad_result.type == core.VADPluginResultType.STARTED:
                state[0] = AgentState.LISTENING
                state[1] = metadata.sequence_number
                chatgpt_plugin.interrupt()
            else:
                state[0] = AgentState.SPEAKING
                state[1] = metadata.sequence_number

        async def process_stt(text_stream: AsyncIterator[str], metadata: core.PluginIterator.ResultMetadata):
            complete_stt_result = ""
            async for stt_r in text_stream:
                complete_stt_result += stt_r.text
            msg = ChatGPTMessage(
                role=ChatGPTMessageRole.user, content=complete_stt_result)
            return msg

        async def send_audio(frame_stream: AsyncIterator[rtc.AudioFrame], metadata: core.PluginIterator.ResultMetadata):
            async for frame in frame_stream:
                if (should_skip(frame, metadata)):
                    continue
                await source.capture_frame(frame)

        def should_skip(_: any, metadata: core.PluginIterator.ResultMetadata):
            return state[0] == AgentState.LISTENING or metadata.sequence_number < state[1]

        await vad_plugin\
            .set_input(input_iterator)\
            .do(vad_state_changer)\
            .filter(lambda data, _: data.type == core.VADPluginResultType.FINISHED)\
            .map(lambda data, _: data.frames)\
            .pipe(stt_plugin)\
            .map_async(process_stt)\
            .pipe(chatgpt_plugin)\
            .pipe(complete_sentence_plugin)\
            .skip_while(should_skip)\
            .pipe(tts_plugin)\
            .do_async(send_audio)\
            .run()

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
