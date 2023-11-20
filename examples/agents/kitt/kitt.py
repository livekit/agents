import os
import json
import asyncio
import logging
from dataclasses import dataclass
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


AgentState = Enum("AgentState", "LISTENING, THINKING, SPEAKING")
UserState = Enum("UserState", "SPEAKING, SILENT")


@dataclass
class State:
    agent_sending_audio: bool = False
    chat_gpt_working: bool = False
    user_state: UserState = UserState.SILENT
    current_sequence_number: int = 0

    def get_agent_state(self):
        if self.agent_sending_audio:
            return AgentState.SPEAKING

        if self.chat_gpt_working:
            return AgentState.THINKING

        return AgentState.LISTENING

    def to_metadata(self):
        return create_message(
            user_state=self.user_state.name.lower(),
            agent_state=self.get_agent_state().name.lower())


def create_message(**kwargs):
    return json.dumps(kwargs)


async def process_track(ctx: agents.JobContext, track: rtc.Track, source: rtc.AudioSource, state: State):
    logging.info("Processing Track: %s", track.sid)
    audio_stream = rtc.AudioStream(track)
    input_iterator = core.PluginIterator.create(audio_stream)

    vad_plugin = VADPlugin(
        left_padding_ms=1000, silence_threshold_ms=500)
    stt_plugin = WhisperAPITranscriber()
    chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
    complete_sentence_plugin = core.utils.CompleteSentencesPlugin()
    tts_plugin = TTSPlugin()

    async def set_metadata():
        await ctx.room.local_participant.publish_data(state.to_metadata())

    async def vad_state_changer(
            vad_result: core.VADPluginResult,
            metadata: core.PluginIterator.ResultMetadata):
        if vad_result.type == core.VADPluginResultType.STARTED:
            state.user_state = UserState.SPEAKING
            state.current_sequence_number = metadata.sequence_number
            chatgpt_plugin.interrupt()
            await set_metadata()
        else:
            state.user_state = UserState.SILENT
            state.current_sequence_number = metadata.sequence_number
            await set_metadata()

    async def process_stt(text_stream: AsyncIterator[str], metadata: core.PluginIterator.ResultMetadata):
        complete_stt_result = ""
        async for stt_r in text_stream:
            complete_stt_result += stt_r.text
            asyncio.create_task(ctx.room.local_participant.publish_data(
                create_message(type="transcription", text=stt_r.text)))
        msg = ChatGPTMessage(
            role=ChatGPTMessageRole.user, content=complete_stt_result)
        return msg

    async def process_chatgpt(chatgpt_stream: AsyncIterator[str], metadata: core.PluginIterator.ResultMetadata):
        async def iterator():
            state.chat_gpt_working = True
            await set_metadata()
            async for chatgpt_r in chatgpt_stream:
                yield chatgpt_r

            state.chat_gpt_working = False
            await set_metadata()

        return iterator()

    async def send_audio(frame_stream: AsyncIterator[rtc.AudioFrame], metadata: core.PluginIterator.ResultMetadata):
        state.agent_sending_audio = True
        await set_metadata()
        async for frame in frame_stream:
            if (should_skip(frame, metadata)):
                continue
            await source.capture_frame(frame)
        state.agent_sending_audio = False
        await set_metadata()

    def should_skip(_: any, metadata: core.PluginIterator.ResultMetadata):
        return state.user_state == UserState.SPEAKING or metadata.sequence_number < state.current_sequence_number

    await vad_plugin\
        .set_input(input_iterator)\
        .do_async(vad_state_changer)\
        .filter(lambda data, _: data.type == core.VADPluginResultType.FINISHED)\
        .map(lambda data, _: data.frames)\
        .pipe(stt_plugin)\
        .map_async(process_stt)\
        .do(lambda *a: logging.info("After STT"))\
        .pipe(chatgpt_plugin)\
        .map_async(process_chatgpt)\
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
    state = State()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        logging.info("Subscribed to track")
        asyncio.create_task(process_track(ctx, track, source, state))

    await ctx.room.local_participant.publish_track(track, options)
    logging.info("Published track")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        await job_request.accept(kitt_agent, should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
