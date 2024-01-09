import asyncio

from livekit import agents
from livekit.plugins import elevenlabs


async def test_synthetize():
    tts = elevenlabs.TTS()
    audio = await tts.synthesize(text="Hello world")
    print(audio)


async def test_stream():
    tts = elevenlabs.TTS()

    session_1 = ["Hello", "world.", "This", "is a tts test"]
    session_2 = ["Hello", "world again.", "This", "is another tts test"]

    async def stream(tts: agents.tts.TTS):
        stream = await tts.stream()
        for word in session_1:
            stream.push_text(word)
            await asyncio.sleep(0.01)

        await stream.flush()

        for word in session_2:
            stream.push_text(word)
            await asyncio.sleep(0.01)

        await stream.flush()
        await stream.close()

        async for event in stream:
            print("Event: ", event)

        print("Done")

    async with asyncio.TaskGroup() as group:
        group.create_task(stream(tts))
