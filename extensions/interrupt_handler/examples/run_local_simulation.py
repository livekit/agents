"""
A simulation demonstrating typical flows:
- agent speaks
- user says 'uh' (filler) -> ignored
- user says 'umm okay stop' -> interrupt
- agent silent and user says 'umm' -> registered
"""
import asyncio
from livekit_interrupt_handler import InterruptHandler
from livekit_interrupt_handler.server import create_app
import aiohttp
import json

async def simulate():
    handler = InterruptHandler()
    # Start REST server in background (optional)
    app = create_app(handler)
    runner = None
    try:
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8081)
        await site.start()
    except Exception:
        # ignore if aiohttp not available; it's optional
        runner = None

    # Simulation sequence
    print("=== Simulation start ===")
    # Agent starts speaking
    await handler.set_agent_speaking(True)
    # Filler while agent speaks
    res = await handler.on_transcription({"text": "uh", "confidence": 0.95})
    print(res)
    # Mixed filler+command
    res = await handler.on_transcription({"text": "umm okay stop", "confidence": 0.98})
    print(res)
    # low confidence murmur while speaking
    res = await handler.on_transcription({"text": "hmm yeah", "confidence": 0.3})
    print(res)
    # Agent stops
    await handler.set_agent_speaking(False)
    res = await handler.on_transcription({"text": "umm", "confidence": 0.9})
    print(res)

    if runner:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(simulate())
