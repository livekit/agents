import asyncio
import logging
import random

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli

# Load environment variables
load_dotenv()

WIDTH = 640
HEIGHT = 480


async def entrypoint(job: JobContext):
    await job.connect()

    room = job.room
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("single-color", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    publication = await room.local_participant.publish_track(track, options)
    logging.info("published track", extra={"track_sid": publication.sid})

    async def _draw_color():
        argb_frame = bytearray(WIDTH * HEIGHT * 4)
        while True:
            await asyncio.sleep(0.1)  # 100ms

            # Create a new random color
            r, g, b = (random.randint(0, 255) for _ in range(3))
            color = bytes([r, g, b, 255])

            # Fill the frame with the new random color
            argb_frame[:] = color * WIDTH * HEIGHT
            frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
            source.capture_frame(frame)

    await _draw_color()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
