import os
import json
import asyncio
import logging
from livekit import agents, rtc
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


def create_message(**kwargs):
    return json.dumps(kwargs)


async def process_track(ctx: agents.JobContext, track: rtc.Track, source: rtc.AudioSource, state: [AgentState, int]):
    audio_stream = rtc.AudioStream(track)
    input_iterator = core.PluginIterator.create(audio_stream)

    vad_plugin = VADPlugin(
        left_padding_ms=1000, silence_threshold_ms=500)
    stt_plugin = WhisperAPITranscriber()
    chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
    complete_sentence_plugin = core.utils.CompleteSentencesPlugin()
    tts_plugin = TTSPlugin()

    async def vad_state_changer(
            vad_result: core.VADPluginResult,
            metadata: core.PluginIterator.ResultMetadata):
        if vad_result.type == core.VADPluginResultType.STARTED:
            state[0] = AgentState.LISTENING
            state[1] = metadata.sequence_number
            chatgpt_plugin.interrupt()
            await ctx.room.local_participant.update_metadata(
                create_message(state="listening"))
        else:
            state[0] = AgentState.SPEAKING
            state[1] = metadata.sequence_number
            await ctx.room.local_participant.update_metadata(
                create_message(state="speaking"))

    async def process_stt(text_stream: AsyncIterator[str], metadata: core.PluginIterator.ResultMetadata):
        complete_stt_result = ""
        async for stt_r in text_stream:
            complete_stt_result += stt_r.text
            asyncio.create_task(ctx.room.local_participant.publish_data(
                create_message(type="transcription", text=stt_r.text)))
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
        .do_async(vad_state_changer)\
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


async def kitt_agent(ctx: agents.JobContext):

    source = rtc.AudioSource(OAI_TTS_SAMPLE_RATE, OAI_TTS_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    state = [AgentState.LISTENING, 0]  # state, sequence number of interest

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        logging.info("Subscribed to track")
        asyncio.create_task(process_track(ctx, track, source, state))

    await ctx.room.local_participant.publish_track(track, options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        await job_request.accept(kitt_agent, should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
