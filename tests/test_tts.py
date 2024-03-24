import asyncio
import wave

from livekit import agents
from livekit.plugins import elevenlabs, openai
from utils import compare_word_counts

TEST_AUDIO_SYNTHESIZE = "the people who are crazy enough to think they can change the world are the ones who do"


async def test_synthetize():
    ttss = [elevenlabs.TTS(), openai.TTS()]

    async def synthetize(tts: agents.tts.TTS):
        frames = []
        async for frame in tts.synthesize(text=TEST_AUDIO_SYNTHESIZE):
            frames.append(frame.data)

        result = await openai.STT().recognize(buffer=agents.utils.merge_frames(frames))
        assert (
            compare_word_counts(result.alternatives[0].text, TEST_AUDIO_SYNTHESIZE)
            > 0.9
        )

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

    result = await openai.STT().recognize(buffer=agents.utils.merge_frames(frames))
    assert compare_word_counts(result.alternatives[0].text, TEST_AUDIO_SYNTHESIZE) > 0.9

    await stream.aclose()
