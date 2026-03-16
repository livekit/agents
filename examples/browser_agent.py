import logging

from dotenv import load_dotenv

from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins.browser import (
    AudioData,
    BrowserContext,
    BrowserSession,
    PaintData,
)

logger = logging.getLogger("browser-agent")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    browser_ctx = BrowserContext(dev_mode=False)
    await browser_ctx.initialize()

    page = await browser_ctx.new_page(
        url="https://news.ycombinator.com",
        width=1280,
        height=720,
        framerate=30,
    )

    # Access raw paint frames and audio data
    @page.on("paint")
    def on_paint(data: PaintData):
        # data.frame is an rtc.VideoFrame (BGRA), data.width/height, data.dirty_rects
        pass

    @page.on("audio")
    def on_audio(data: AudioData):
        # data.frame is an rtc.AudioFrame, data.pts is the presentation timestamp
        pass

    # Use Playwright for programmatic browser control (CDP)
    async with browser_ctx.playwright() as browser:
        pages = browser.contexts[0].pages
        if pages:
            pw_page = pages[0]
            title = await pw_page.title()
            logger.info("page title: %s", title)

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    session = BrowserSession(page=page, room=ctx.room)
    await session.start()

    async def cleanup():
        await session.aclose()
        await page.aclose()
        await browser_ctx.aclose()

    ctx.add_shutdown_callback(cleanup)


if __name__ == "__main__":
    cli.run_app(server)
