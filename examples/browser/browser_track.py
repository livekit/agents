import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins import browser

WIDTH = 1920
HEIGHT = 1080

load_dotenv()


async def entrypoint(job: JobContext):
    await job.connect()

    ctx = browser.BrowserContext(dev_mode=True)
    await ctx.initialize()

    page = await ctx.new_page(url="www.google.com")

    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("single-color", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    publication = await job.room.local_participant.publish_track(track, options)
    logging.info("published track", extra={"track_sid": publication.sid})

    @page.on("paint")
    def on_paint(data):
        source.capture_frame(data)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
