# Copyright 2024 LiveKit, Inc.
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

import asyncio
import json
import logging
import time
from typing import AsyncIterable, Optional

from livekit import agents, rtc
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.fal import SDTurboHighFPS

INTRO_MESSAGE = """
Hi there! I will swap your face for a celebrity's face using FAL.AI's real-time stable diffusion api.
Just type in the name of the celebrity you want to swap with. For example, try typing in "tom cruise"
"""

BYE_MESSAGE = """
Thanks for giving this a try! Goodbye for now.
"""

_OUTPUT_HEIGHT = 720
_OUTPUT_WIDTH = 960
_ELEVEN_TTS_SAMPLE_RATE = 24000
_ELEVEN_TTS_CHANNELS = 1


class FalAI:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = FalAI(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        self.falai: Optional[SDTurboHighFPS] = SDTurboHighFPS()
        self.stream = self.falai.stream()
        self.tts_plugin = TTS(
            model_id="eleven_turbo_v2", sample_rate=_ELEVEN_TTS_SAMPLE_RATE, latency=2
        )
        self.video_out = rtc.VideoSource(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)
        self.audio_out = rtc.AudioSource(_ELEVEN_TTS_SAMPLE_RATE, _ELEVEN_TTS_CHANNELS)
        self.latest_results = []
        self.detecting = False
        self.chat = rtc.ChatManager(ctx.room)
        self.chat.on("message_received", self.on_chat_received)
        self.celebrity = "blankness"

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.process_track(track))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        video_track = rtc.LocalVideoTrack.create_video_track(
            "agent-video", self.video_out
        )
        await self.ctx.room.local_participant.publish_track(video_track)
        # Send an empty frame to initialize the video track
        argb_frame = rtc.ArgbFrame.create(
            format=rtc.VideoFormatType.FORMAT_ARGB,
            width=_OUTPUT_WIDTH,
            height=_OUTPUT_HEIGHT,
        )
        argb_frame.data[:] = bytearray(_OUTPUT_WIDTH * _OUTPUT_HEIGHT * 4)
        self.video_out.capture_frame(rtc.VideoFrame(argb_frame.to_i420()))

        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-mic", self.audio_out
        )
        await self.ctx.room.local_participant.publish_track(audio_track)
        self.tts_stream = self.tts_plugin.stream()
        self.ctx.create_task(self.send_audio_stream(self.tts_stream))

        self.update_state("idle")

        # give time for the subscriber to fully subscribe to the agent's tracks
        await asyncio.sleep(1)
        await self.send_chat_and_voice(INTRO_MESSAGE)

        self.ctx.create_task(self.send_frames())

        # limit to 2 mins
        self.ctx.create_task(self.end_session_after(2 * 60))

    async def send_frames(self):
        async for frame in self.stream:
            self.video_out.capture_frame(frame)

    def on_chat_received(self, message: rtc.ChatMessage):
        self.celebrity = message.message

    async def end_session_after(self, duration: int):
        await asyncio.sleep(duration)
        await self.send_chat_and_voice(BYE_MESSAGE)
        self.update_state("idle")
        await asyncio.sleep(5)
        await self.ctx.disconnect()

    async def send_chat_and_voice(self, message: str):
        self.tts_stream.push_text(message)
        await self.tts_stream.flush()
        await self.chat.send_message(message)

    async def process_track(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        last_time = time.time()
        async for frame in video_stream:
            # throttle to 10 fps
            if (time.time() - last_time) < 0.1:
                continue
            last_time = time.time()
            self.stream.push_frame(
                frame=frame, prompt=f"Replace face with {self.celebrity}", strength=0.4
            )

    async def send_audio_stream(
        self, tts_stream: AsyncIterable[agents.tts.SynthesisEvent]
    ):
        async for e in tts_stream:
            if e.type == agents.tts.SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
            elif e.type == agents.tts.SynthesisEventType.STARTED:
                self.update_state("speaking")
            elif e.type == agents.tts.SynthesisEventType.FINISHED:
                self.update_state("idle")

        await tts_stream.aclose()

    def update_state(self, state: str):
        metadata = json.dumps({"agent_state": state})
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for Fal AI")

        await job_request.accept(
            FalAI.create,
            identity="falai_agent",
            auto_subscribe=agents.AutoSubscribe.VIDEO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(job_request_cb)
    agents.run_app(worker)
