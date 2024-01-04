from livekit.plugins import elevenlabs


async def test_synthetize():
    tts = elevenlabs.TTS()
    audio = await tts.synthesize(text="Hello world")
    print(audio)
