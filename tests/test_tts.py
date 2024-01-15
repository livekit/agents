import asyncio
import wave
from livekit import rtc, agents
from livekit.plugins import elevenlabs, openai

TEST_AUDIO_SYNTHESIZE = "the people who are crazy enough to think they can change the world are the ones who do"


def save_wave_file(filename: str, frame: rtc.AudioFrame) -> None:
    with wave.open(filename, "wb") as file:
        file.setnchannels(frame.num_channels)
        file.setsampwidth(2)
        file.setframerate(frame.sample_rate)
        file.writeframes(frame.data)


async def test_synthetize():
    ttss = [elevenlabs.TTS(), openai.TTS()]

    async def synthetize(tts: agents.tts.TTS):
        audio = await tts.synthesize(text=TEST_AUDIO_SYNTHESIZE)
        save_wave_file(tts.__class__.__module__ + ".wav", audio.data)

    async with asyncio.TaskGroup() as group:
        for tts in ttss:
            group.create_task(synthetize(tts))


async def test_stream():
    tts = elevenlabs.TTS()

    pattern = [1, 2, 4]
    text = TEST_AUDIO_SYNTHESIZE
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tts.stream()

    for chunk in chunks:
        stream.push_text(chunk)

    await stream.flush()

    frames = []
    assert (await anext(stream)).type == agents.tts.SynthesisEventType.STARTED

    async for event in stream:
        if event.type == agents.tts.SynthesisEventType.FINISHED:
            break

        assert event.type == agents.tts.SynthesisEventType.AUDIO
        frames.append(event.audio.data)

    audio = agents.utils.merge_frames(frames)
    save_wave_file("2.wav", audio)

    await stream.aclose()
