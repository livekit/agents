import asyncio
import json
import logging
from typing import Optional

from livekit import agents, rtc
from livekit.agents.vad import VADEventType
from livekit.plugins import openai, silero


class PainterAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = PainterAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        whisper_stt = openai.STT()
        self.vad = silero.VAD()
        self.stt = agents.stt.StreamAdapter(whisper_stt, self.vad.stream())
        self.dalle = openai.Dalle3()

        self.ctx = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.prompt: Optional[str] = None
        self.current_image: Optional[rtc.VideoFrame] = None

        # setup callbacks
        def subscribe_cb(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.audio_track_worker(track))

        self.ctx.room.on("track_subscribed", subscribe_cb)
        self.chat.on("message_received", self.process_chat)

    async def start(self):
        # sends welcome message as a new task
        self.ctx.create_task(
            self.chat.send_message(
                "Welcome to the painter agent! Speak or type what you'd like me to paint."
            )
        )

        self.ctx.create_task(self.image_generation_worker())
        self.ctx.create_task(self.image_publish_worker())
        self.update_agent_state("listening")

    def process_chat(self, msg: rtc.ChatMessage):
        self.prompt = msg.message

    def update_agent_state(self, state: str):
        metadata = json.dumps(
            {
                "agent_state": state,
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))

    async def audio_track_worker(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        self.stt_stream = self.stt.stream()
        self.ctx.create_task(self.stt_worker())

        async for frame in audio_stream:
            self.stt_stream.push_frame(frame)
        await self.stt.flush()

    async def stt_worker(self):
        async for event in self.stt_stream:
            # we only want to act when result is final
            if not event.is_final:
                continue

            # require 70% confidence before we act
            speech = event.alternatives[0]
            if speech.confidence < 0.7:
                self.ctx.create_task(
                    self.chat.send_message(
                        f"Sorry, I didn't catch that. Confidence {speech.confidence}"
                    )
                )
                continue
            self.prompt = speech.text

    async def image_generation_worker(self):
        # task will be canceled when Agent is disconnected
        while True:
            prompt, self.prompt = self.prompt, None
            if prompt:
                self.update_agent_state("generating")
                self.ctx.create_task(
                    self.chat.send_message(
                        f'Generating "{prompt}". It\'ll be just a minute.'
                    )
                )
                try:
                    argb_frame = await self.dalle.generate(prompt, size="1792x1024")
                    self.current_image = rtc.VideoFrame(argb_frame.to_i420())
                except Exception as e:
                    logging.error("failed to generate image: %s", e, exc_info=e)
                    self.ctx.create_task(
                        self.chat.send_message("Sorry, I ran into an error.")
                    )
                self.update_agent_state("listening")
            await asyncio.sleep(0.05)

    async def image_publish_worker(self):
        image: rtc.VideoFrame = None
        video_source: rtc.VideoSource = None
        while True:
            if self.current_image:
                image, self.current_image = self.current_image, None

            if not image:
                await asyncio.sleep(0.1)
                continue

            if not video_source:
                video_source = rtc.VideoSource(image.buffer.width, image.buffer.height)
                track = rtc.LocalVideoTrack.create_video_track("image", video_source)
                await self.ctx.room.local_participant.publish_track(track)

            video_source.capture_frame(image)
            # publish at 1fps
            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        await job_request.accept(
            PainterAgent.create,
            identity="painter",
            subscribe_cb=agents.SubscribeCallbacks.AUDIO_ONLY,
            auto_disconnect_cb=agents.AutoDisconnectCallbacks.DEFAULT,
        )

    worker = agents.Worker(job_request_cb=job_request_cb)
    agents.run_app(worker)
