import asyncio
from datetime import datetime
import json
import logging
from typing import Optional

from livekit import agents, rtc
from livekit.plugins import openai, silero


class PainterAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = PainterAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        self.whisper_stt = openai.STT()
        self.vad = silero.VAD()
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

        def process_chat(msg: rtc.ChatMessage):
            self.prompt = msg.message

        self.ctx.room.on("track_subscribed", subscribe_cb)
        self.chat.on("message_received", process_chat)

    async def start(self):
        # give a bit of time for the user to fully connect so they don't miss
        # the welcome message
        await asyncio.sleep(1)

        # create_task is used to run coroutines in the background
        self.ctx.create_task(
            self.chat.send_message(
                "Welcome to the painter agent! Speak or type what you'd like me to paint."
            )
        )

        self.ctx.create_task(self.image_generation_worker())
        self.ctx.create_task(self.image_publish_worker())
        self.update_agent_state("listening")

    def update_agent_state(self, state: str):
        metadata = json.dumps(
            {
                "agent_state": state,
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))

    async def audio_track_worker(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        vad_stream = self.vad.stream(min_silence_duration=2.0)
        stt = agents.stt.StreamAdapter(self.whisper_stt, vad_stream)
        stt_stream = stt.stream()
        self.ctx.create_task(self.stt_worker(stt_stream))

        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)
        await stt_stream.flush()

    async def stt_worker(self, stt_stream: agents.stt.SpeechStream):
        async for event in stt_stream:
            # we only want to act when result is final
            if not event.is_final:
                continue
            speech = event.alternatives[0]
            self.prompt = speech.text
        await stt_stream.aclose()

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
                started_at = datetime.now()
                try:
                    argb_frame = await self.dalle.generate(prompt, size="1792x1024")
                    self.current_image = argb_frame
                    elapsed = (datetime.now() - started_at).seconds
                    self.ctx.create_task(
                        self.chat.send_message(f"Done! Took {elapsed} seconds.")
                    )
                except Exception as e:
                    logging.error("failed to generate image: %s", e, exc_info=e)
                    self.ctx.create_task(
                        self.chat.send_message("Sorry, I ran into an error.")
                    )
                self.update_agent_state("listening")
            await asyncio.sleep(0.05)

    async def image_publish_worker(self):
        video_source = rtc.VideoSource(1792, 1024)
        track = rtc.LocalVideoTrack.create_video_track("image", video_source)
        await self.ctx.room.local_participant.publish_track(track)
        image: rtc.VideoFrame = None
        while True:
            if self.current_image:
                image, self.current_image = self.current_image, None

            if not image:
                await asyncio.sleep(0.1)
                continue

            video_source.capture_frame(image)
            # publish at 1fps
            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        await job_request.accept(
            PainterAgent.create,
            identity="painter",
            name="Painter",
            # subscribe to all audio tracks automatically
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            # disconnect when the last participant leaves
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
