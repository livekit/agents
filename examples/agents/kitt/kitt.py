# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import asyncio
import logging
from typing import Optional, Set
from livekit import agents, rtc
from livekit.plugins.vad import VADPlugin, VADEventType
from livekit.plugins.openai import (WhisperAPITranscriber,
                                    ChatGPTPlugin,
                                    ChatGPTMessage,
                                    ChatGPTMessageRole,
                                    TTSPlugin)
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
        # state
        self.agent_sending_audio: bool = False
        self.chat_gpt_working: bool = False
        self.user_state: UserState = UserState.SILENT

        # plugins
        self.vad_plugin = VADPlugin(
            left_padding_ms=1000,
            silence_threshold_ms=500)
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=PROMPT, message_capacity=20, model="gpt-3.5-turbo")
        self.stt_plugin = WhisperAPITranscriber()
        self.tts_plugin = TTSPlugin()

        self.ctx: Optional[agents.JobContext] = None
        self.track_tasks: Set[asyncio.Task] = set()
        self.stt_tasks: Set[asyncio.Task] = set()

        self.ctx = None
        self.line_out: Optional[rtc.AudioSource] = None
        self.track_tasks = set()
        self.stt_tasks = set()

    async def start(self, ctx: agents.JobContext):
        self.ctx = ctx
        ctx.room.on("track_subscribed", self.on_track_subscribed)
        ctx.room.on("disconnected", self.cleanup)
        await self.publish_audio()

    async def publish_audio(self):
        self.line_out = rtc.AudioSource(OAI_TTS_SAMPLE_RATE, OAI_TTS_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(
            "agent-mic", self.line_out)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.ctx.room.local_participant.publish_track(track, options)

    def on_track_subscribed(
            self,
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        t = asyncio.create_task(self.process_track(track))
        self.track_tasks.add(t)
        t.add_done_callback(
            lambda t: t in self.track_tasks and self.track_tasks.remove(t))

    def cleanup(self):
        pass

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for vad_result in self.vad_plugin.start(audio_stream):
            if vad_result.type == VADEventType.STARTED:
                self.user_state = UserState.SPEAKING
                self.chatgpt_plugin.interrupt()
                for task in self.stt_tasks:
                    task.cancel()
                self.stt_tasks.clear()
                await self.send_state_update()
            elif vad_result.type == VADEventType.FINISHED:
                self.user_state = UserState.SILENT
                await self.send_state_update()
                stt_output = await self.stt_plugin.transcribe_frames(vad_result.frames)
                if len(stt_output) == 0:
                    continue
                t = asyncio.create_task(self.process_stt_result(stt_output))
                self.stt_tasks.add(t)
                t.add_done_callback(
                    lambda t: t in self.stt_tasks and self.stt_tasks.remove(t))

    async def process_stt_result(self, text):
        await self.ctx.room.local_participant.publish_data(json.dumps({"type": "transcription", "text": text}))
        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=text)
        chatgpt_result = self.chatgpt_plugin.add_message(msg)
        await self.process_chatgpt_result(chatgpt_result)

    async def process_chatgpt_result(self, text_stream):
        self.chat_gpt_working = True
        await self.send_state_update()

        sentence = ""
        async for text in text_stream:
            sentence += text

            if text.endswith("\n") or text.endswith(
                    "?") or text.endswith("!") or text.endswith("."):
                audio_stream = await self.tts_plugin.generate_speech_from_text(sentence)
                await self.send_audio_stream(audio_stream)
                sentence = ""

        if len(sentence) > 0:
            audio_stream = await self.tts_plugin.generate_speech_from_text(sentence)
            await self.send_audio_stream(audio_stream)

        self.chat_gpt_working = False
        await self.send_state_update()

    async def send_audio_stream(self, audio_stream):
        self.agent_sending_audio = True
        await self.send_state_update()
        async for frame in audio_stream:
            await self.line_out.capture_frame(frame)
        self.agent_sending_audio = False
        await self.send_state_update()

    def get_agent_state(self):
        if self.agent_sending_audio:
            return AgentState.SPEAKING

        if self.chat_gpt_working:
            return AgentState.THINKING

        return AgentState.LISTENING

    async def send_state_update(self):
        msg = json.dumps({"type": "state",
                          "user_state": self.user_state.name.lower(),
                          "agent_state": self.get_agent_state().name.lower()})
        await self.ctx.room.local_participant.publish_data(msg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        print("Accepting job for KITT")
        kitt = KITT()
        await job_request.accept(kitt.start, should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO)

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
