from livekit.plugins import openai


async def test_synthetize():
    tts = openai.TTS()
    audio = await tts.synthesize("Hello world")
    print(audio)
    print(audio.data.samples_per_channel)
    print(audio.data.data)
