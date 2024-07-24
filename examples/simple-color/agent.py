import asyncio
import logging

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli

WIDTH = 640
HEIGHT = 480

# change this color in dev mode and the agent will automatically update
COLOR = bytes([0, 255, 0, 255])


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

            argb_frame[:] = COLOR * WIDTH * HEIGHT
            frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
            source.capture_frame(frame)

    asyncio.create_task(_draw_color())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
