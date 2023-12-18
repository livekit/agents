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
import logging
from state_manager import StateManager, AgentState, UserState

from livekit import rtc, agents
from livekit.plugins.openai import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
    WhisperAPITranscriber,
)
from livekit.plugins.elevenlabs import TTSPlugin
from livekit.plugins.vad import VADEventType, VADPlugin

PROMPT = "You are KITT, a friendly voice assistant powered by LiveKit.  \
          Conversation should be personable, and be sure to ask follow up questions. \
          If your response is a question, please append a question mark symbol to the end of it.\
          Don't respond with more than a few sentences."
INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit, ChatGPT, and Eleven Labs. \
        You can find my source code in the top right of this screen if you're curious how I work. \
        Feel free to ask me anything — I'm here to help! Just start talking or type in the chat."

ELEVEN_TTS_SAMPLE_RATE = 44100
ELEVEN_TTS_CHANNELS = 1


class KITT:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        kitt = KITT(ctx)
        await kitt.start()

    def __init__(self, ctx: agents.JobContext):
        # state
        self.state = StateManager(ctx)

        # plugins
        self.vad_plugin = VADPlugin(left_padding_ms=100, silence_threshold_ms=250)
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=PROMPT, message_capacity=20, model="gpt-4-1106-preview"
        )
        self.stt_plugin = WhisperAPITranscriber()
        self.tts_plugin = TTSPlugin()

        self.ctx: agents.JobContext = ctx
        self.line_out = rtc.AudioSource(ELEVEN_TTS_SAMPLE_RATE, ELEVEN_TTS_CHANNELS)

    async def start(self):
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("data_received", self.on_data_received)
        self.ctx.room.on("disconnected", self.cleanup)
        await self.publish_audio()
        async def intro_text_stream():
            yield INTRO

        await self.process_chatgpt_result(intro_text_stream())

    def on_data_received(self, data: bytes, participant: rtc.RemoteParticipant, kind):
        payload = json.loads(data.decode("utf-8"))

        if payload["type"] == "user_chat_message":
            text = payload["text"]
            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=text)
            chatgpt_result = self.chatgpt_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_result))

    async def publish_audio(self):
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.line_out)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.ctx.room.local_participant.publish_track(track, options)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    def cleanup(self):
        logging.info("KITT agent clean up")

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for vad_result in self.vad_plugin.start(audio_stream):
            if vad_result.type == VADEventType.STARTED:
                self.user_state = UserState.SPEAKING
            elif vad_result.type == VADEventType.FINISHED:
                self.user_state = UserState.SILENT
                if self.state.agent_state in [AgentState.SPEAKING, AgentState.THINKING]:
                    continue
                stt_output = await self.stt_plugin.transcribe_frames(vad_result.frames)
                if len(stt_output) == 0:
                    continue
                self.ctx.create_task(self.process_stt_result(stt_output))

    async def process_stt_result(self, text):
        await self.ctx.room.local_participant.publish_data(
            json.dumps({"type": "transcription", "text": text})
        )
        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=text)
        chatgpt_result = self.chatgpt_plugin.add_message(msg)
        await self.process_chatgpt_result(chatgpt_result)

    async def process_chatgpt_result(self, text_stream):
        self.state.chat_gpt_working = True
        all_text = ""

        async for text in text_stream:
            all_text += text

        self.state.chat_gpt_working = False

        async def text_iterator():
            yield all_text

        audio_stream = await self.tts_plugin.generate_speech(text_iterator())
        await self.send_message_from_agent(all_text)
        await self.send_audio_stream(audio_stream)

    async def send_audio_stream(self, audio_stream):
        self.state.agent_sending_audio = True
        async for frame in audio_stream:
            await self.line_out.capture_frame(frame)
        self.state.agent_sending_audio = False

    async def send_message_from_agent(self, text):
        await self.ctx.room.local_participant.publish_data(
            json.dumps({"type": "agent_chat_message", "text": text})
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            KITT.create,
            identity="kitt_agent",
            subscribe_cb=agents.SubscribeCallbacks.AUDIO_ONLY,
            auto_disconnect_cb=agents.AutoDisconnectCallbacks.DEFAULT,
        )

    worker = agents.Worker(
        job_request_cb=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
