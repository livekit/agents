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
import datetime
import json
import logging
from typing import AsyncIterable

from fal_sd_turbo import FalSDTurbo
from game_state import GAME_STATE, GameState
from livekit import agents, rtc
from livekit.plugins.deepgram import STT
from livekit.plugins.elevenlabs import TTS

INTRO_MESSAGE = """
Hi there, this is a guessing game where you guess which celebrity I've swapped your face for. 
Just tell me when you are ready to start playing.
"""

BYE_MESSAGE = """
Thanks for giving this a try! Goodbye for now.
"""

_FAL_OUTPUT_WIDTH = 512
_FAL_OUTPUT_HEIGHT = 512
_ELEVEN_TTS_SAMPLE_RATE = 24000
_ELEVEN_TTS_CHANNELS = 1


class FalAI:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = FalAI(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        self.falai = FalSDTurbo()
        self.fal_stream = self.falai.stream()
        self.stt_plugin = STT()
        self.stt_stream = self.stt_plugin.stream()
        self.tts_plugin = TTS(
            model_id="eleven_turbo_v2", sample_rate=_ELEVEN_TTS_SAMPLE_RATE, latency=2
        )
        self.tts_stream = self.tts_plugin.stream()
        self.game_state = GameState(
            ctx=ctx,
            send_message=lambda msg: self.ctx.create_task(
                self.send_chat_and_voice(msg)
            ),
        )
        self.video_out = rtc.VideoSource(_FAL_OUTPUT_WIDTH, _FAL_OUTPUT_HEIGHT)
        self.audio_out = rtc.AudioSource(_ELEVEN_TTS_SAMPLE_RATE, _ELEVEN_TTS_CHANNELS)
        self.current_audio_level = 0
        self.chat = rtc.ChatManager(ctx.room)
        self.chat.on("message_received", self.on_chat_received)
        self.received_fal_frame = False

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            pub: rtc.TrackPublication,
            part: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self.ctx.create_task(self.process_video_track(track))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self.ctx.create_task(self.process_audio_track(track))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        self.update_state("idle")

        await self.publish_tracks()

        # give time for the subscriber to fully subscribe to the agent's tracks
        await asyncio.sleep(1)
        await self.send_chat_and_voice(INTRO_MESSAGE)

        self.ctx.create_task(self.send_fal_frames())
        self.ctx.create_task(self.send_audio_stream(self.tts_stream))
        self.ctx.create_task(self.process_stt())

        # limit to 2 mins
        self.ctx.create_task(self.end_session_after(2 * 60))

    # Publish Tracks
    async def publish_tracks(self):
        video_track = rtc.LocalVideoTrack.create_video_track(
            "agent-video", self.video_out
        )
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-mic", self.audio_out
        )

        await self.ctx.room.local_participant.publish_track(video_track)
        await self.ctx.room.local_participant.publish_track(audio_track)

        # Send an empty frame to initialize the video track
        argb_frame = rtc.VideoFrame(
            _FAL_OUTPUT_WIDTH,
            _FAL_OUTPUT_HEIGHT,
            rtc.VideoBufferType.ARGB,
            bytearray(_FAL_OUTPUT_WIDTH * _FAL_OUTPUT_HEIGHT * 4),
        )
        self.video_out.capture_frame(argb_frame)

    # Video processing
    async def process_video_track(self, track: rtc.Track):
        video_stream = rtc.VideoStream(track)
        async for video_frame_event in video_stream:
            if self.game_state.game_state == GAME_STATE.PLAYING:
                self.fal_stream.push_frame(
                    video_frame_event.frame,
                    prompt=f"webcam screenshot of {self.game_state.current_celebrity}. HD. High Quality.",
                    strength=0.625,
                )
                # Keep sending video frames until we receive a FAL frame
                if not self.received_fal_frame:
                    self.video_out.capture_frame(video_frame_event.frame)
            else:
                self.video_out.capture_frame(video_frame_event.frame)

    async def send_fal_frames(self):
        async for video_frame in self.fal_stream:
            self.received_fal_frame = True
            if self.game_state.game_state != GAME_STATE.PLAYING:
                continue
            self.video_out.capture_frame(video_frame)

    # Audio processing
    async def process_audio_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for audio_frame_event in audio_stream:
            self.stt_stream.push_frame(audio_frame_event.frame)

    async def process_stt(self):
        async for e in self.stt_stream:
            if e.is_final:
                if e.alternatives[0].text == "":
                    continue
                self.game_state.add_user_input(e.alternatives[0].text)
                await self.ctx.room.local_participant.publish_data(
                    json.dumps(
                        {
                            "text": e.alternatives[0].text,
                            "timestamp": int(
                                datetime.datetime.now().timestamp() * 1000
                            ),
                        }
                    ),
                    topic="transcription",
                )

    async def send_audio_stream(
        self, tts_stream: AsyncIterable[agents.tts.SynthesisEvent]
    ):
        audio_frame_queue = asyncio.Queue()
        async for e in tts_stream:
            if e.type == agents.tts.SynthesisEventType.AUDIO:
                await audio_frame_queue.put(e.audio.data)
                await self.audio_out.capture_frame(e.audio.data)
            elif e.type == agents.tts.SynthesisEventType.STARTED:
                audio_frame_queue = asyncio.Queue()
                self.update_state("speaking")
            elif e.type == agents.tts.SynthesisEventType.FINISHED:
                await audio_frame_queue.put(None)
                self.update_state("idle")

        await tts_stream.aclose()

    # Text processing
    def on_chat_received(self, message: rtc.ChatMessage):
        self.game_state.add_user_input(message.message)

    async def send_message(self, message: str):
        self.tts_stream.push_text(message)
        await self.tts_stream.flush()

    # Agent utils
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
            auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(job_request_cb)
    agents.run_app(worker)
