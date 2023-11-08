import asyncio
from livekit import agents
from livekit.plugins.openai import TTSPlugin
import dotenv

dotenv.load_dotenv()


async def main():
    text_iterator = agents.utils.AsyncIteratorList(
        ["Hello World, I'm testing audio"])
    texts_iterator = agents.utils.AsyncIteratorList([text_iterator])

    tts = TTSPlugin()

    resp = tts.start(texts_iterator).unwrap()

    print("Starting...", resp)

    async for r in resp:
        async for frame in r:
            print("Got frame", frame)

    print("Done")

asyncio.run(main())
