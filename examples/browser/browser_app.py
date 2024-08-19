# This example doesn't create any Worker!
# It's just a standalone script that create a new livekit-plugins-browser context.
import asyncio
import logging

from livekit.plugins import browser


logging.basicConfig(level=logging.DEBUG)


async def main():
    ctx = browser.BrowserContext(dev_mode=True)
    await ctx.initialize()

    page = await ctx.new_page(url="www.google.com")
    page = await ctx.new_page(url="www.facebook.com")


if __name__ == "__main__":
    asyncio.run(main())
