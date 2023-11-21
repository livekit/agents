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
from typing import List
from enum import Enum


PROMPT = "You are KITT, a voice assistant in a meeting created by LiveKit. \
          Keep your responses concise while still being friendly and personable. \
          If your response is a question, please append a question mark symbol to the end of it."

OAI_TTS_SAMPLE_RATE = 24000
OAI_TTS_CHANNELS = 1


AgentState = Enum("AgentState", "LISTENING, THINKING, SPEAKING")
UserState = Enum("UserState", "SPEAKING, SILENT")


class KITT():
    def __init__(self):
        self.agent_sending_audio = False
        self.chat_gpt_working = False
        self.user_state = UserState.SILENT
        self.current_sequence_number = 0
        self.vad_plugin = VADPlugin(left_padding_ms=1000, silence_threshold_ms=500)
        self.chatgpt_plugin = ChatGPTPlugin(prompt=PROMPT, message_capacity=20)
        self.complete_sentences_plugin = core.utils.CompleteSentencesPlugin()
        self.stt_plugin = WhisperAPITranscriber()
        self.tts_plugin = TTSPlugin()
        self.ctx = None
        self.source = None

    async def start(self, ctx: agents.JobContext):
        self.ctx = ctx
        ctx.room.on("track_subscribed", self.on_track_subscribed)
        await self.publish_audio()

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        asyncio.create_task(self.process_track(track))

    def get_agent_state(self):
        if self.agent_sending_audio:
            return AgentState.SPEAKING

        if self.chat_gpt_working:
            return AgentState.THINKING

        return AgentState.LISTENING

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_output_stream = await self.vad_plugin.process(audio_stream)
        asyncio.create_task(self.process_vad_result(vad_output_stream))

    async def process_vad_result(self, vad_output_stream):
        async for vad_result in vad_output_stream:
            if vad_result.type == core.VADPluginResultType.STARTED:
                self.user_state = UserState.SPEAKING
                self.chatgpt_plugin.interrupt()
                await self.send_datachannel_state()
            else:
                self.user_state = UserState.SILENT
                await self.send_datachannel_state()
                stt_output_stream = await self.stt_plugin.process(vad_result.frames)
                asyncio.create_task(self.process_stt_result(stt_output_stream))

    async def process_stt_result(self, text_stream):
        complete_stt_result = ""
        async for stt_r in text_stream:
            complete_stt_result += stt_r.text
        asyncio.create_task(self.ctx.room.local_participant.publish_data(json.dumps({"type": "transcription", "text": complete_stt_result})))
        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=complete_stt_result)
        chatgpt_result = await self.chatgpt_plugin.process(msg)
        asyncio.create_task(self.process_chatgpt_result(chatgpt_result))

    async def process_chatgpt_result(self, text_stream):
        async def iterator():
            self.chat_gpt_working = True
            await self.send_datachannel_state()
            async for text in text_stream:
                yield text
            self.chat_gpt_working = False
            await self.send_datachannel_state()

        complete_sentence_result = await self.complete_sentences_plugin.process(iterator())
        asyncio.create_task(self.process_complete_sentence_result(complete_sentence_result))

    async def process_complete_sentence_result(self, complete_sentences_stream):
        async for sentence in complete_sentences_stream:

            async def iterator(s):
                yield s

            tts_result = await self.tts_plugin.process(iterator(sentence))
            await self.process_tts(tts_result)

    async def process_tts(self, audio_stream):
        self.agent_sending_audio = True
        await self.send_datachannel_state()
        async for frame in audio_stream:
            await self.source.capture_frame(frame)
        self.agent_sending_audio = False
        await self.send_datachannel_state()

    async def send_datachannel_state(self):
        msg = json.dumps({"type": "state", "user_state": self.user_state.name.lower(), "agent_state": self.get_agent_state().name.lower()})
        await self.ctx.room.local_participant.publish_data(msg)

    async def publish_audio(self):
        self.source = rtc.AudioSource(OAI_TTS_SAMPLE_RATE, OAI_TTS_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.ctx.room.local_participant.publish_track(track, options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        kitt = KITT()
        await job_request.accept(kitt.start, should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
