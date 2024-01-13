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
from typing import AsyncIterable

from livekit import rtc, agents
from livekit.agents import AgentStatePreset, AgentState
from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)
from livekit.plugins.deepgram import STT
from livekit.plugins.elevenlabs import TTS

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
        # plugins
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=PROMPT, message_capacity=20, model="gpt-4-1106-preview"
        )
        self.stt_plugin = STT()
        self.tts_plugin = TTS()

        self.ctx: agents.JobContext = ctx
        self.data_transport = agents.DataHelper(ctx)
        self.data_transport.on_chat_message(self.on_chat_received)
        self.line_out = rtc.AudioSource(ELEVEN_TTS_SAMPLE_RATE, ELEVEN_TTS_CHANNELS)

        self._sending_audio = False
        self._processing = False
        self._agent_state: AgentStatePreset = AgentStatePreset.IDLE

    async def start(self):
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("disconnected", self.cleanup)
        await self.publish_audio()

        async def intro_text_stream():
            yield INTRO

        await self.process_chatgpt_result(intro_text_stream())
        self.update_state()

    def on_chat_received(self, message: agents.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return

        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=message.message)
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
        stream = self.stt_plugin.stream(sample_rate=44100)
        self.ctx.create_task(self.process_stt_stream(stream))
        async for audio_frame in audio_stream:
            if self._agent_state != AgentStatePreset.LISTENING:
                continue
            stream.push_frame(audio_frame)

    async def process_stt_stream(self, stream):
        async for event in stream:
            if not event.is_final or self._agent_state != AgentStatePreset.LISTENING:
                continue

            alt = event.alternatives[0]
            text = alt.text
            if alt.confidence < 0.75 or text == "":
                continue

            await self.ctx.room.local_participant.publish_data(
                json.dumps({"text": text}),
                topic="transcription",
            )

            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=text)
            chatgpt_stream = self.chatgpt_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))

    async def process_chatgpt_result(self, text_stream):
        # ChatGPT is streamed, so we'll flip the state immediately
        self.update_state(processing=True)

        stream = self.tts_plugin.stream(model_id="eleven_turbo_v2")
        self.ctx.create_task(self.send_audio_stream(stream))
        all_text = ""

        async for text in text_stream:
            stream.push_text(text)
            all_text += text

        self.update_state(processing=False)
        await self.send_message_from_agent(all_text)
        await stream.close()

    async def send_audio_stream(
        self, tts_events: AsyncIterable[agents.tts.SynthesisEvent]
    ):
        first = True
        async for e in tts_events:
            if first:
                first = False
                self.update_state(sending_audio=True)
            await self.line_out.capture_frame(e.audio.data)
        self._sending_audio = False
        self.update_state(sending_audio=False)

    async def send_message_from_agent(self, text):
        # TODO: display incremental tokens when clients support it
        await self.data_transport.send_chat_message(text)

    def update_state(self, sending_audio: bool = None, processing: bool = None):
        if sending_audio is not None:
            self._sending_audio = sending_audio
        if processing is not None:
            self._processing = processing

        state = AgentStatePreset.LISTENING
        if self._sending_audio:
            state = AgentStatePreset.SPEAKING
        elif self._processing:
            state = AgentStatePreset.THINKING

        self._agent_state = state
        self.ctx.create_task(self.data_transport.set_agent_state(state))


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
