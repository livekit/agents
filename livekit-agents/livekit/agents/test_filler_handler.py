import asyncio
from filler_handler import FillerHandler

async def test():
    handler = FillerHandler()
    handler.agent_speaking = True
    print(await handler.handle_transcription("umm", 0.9))       # expect: ignore
    print(await handler.handle_transcription("wait stop", 0.9))  # expect: interrupt
    handler.agent_speaking = False
    print(await handler.handle_transcription("umm", 0.95))       # expect: register

asyncio.run(test())
