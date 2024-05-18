# Check if all TTS is producing valid audio


import pytest
from livekit import agents
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.plugins import elevenlabs, google, openai
from utils import assert_similar_words

TEST_AUDIO_SYNTHESIZE = "the people who are crazy enough to think they can change the world are the ones who do"
SIMILARITY_THRESHOLD = 0.9


async def _assert_valid_synthesized_audio(
    frames: AudioBuffer, tts: agents.tts.TTS, text: str, threshold: float
):
    # use whisper as the source of truth to verify synthesized speech (accuracy is currently the highest)
    whisper_stt = openai.STT(model="whisper-1")
    res = await whisper_stt.recognize(buffer=frames)
    assert_similar_words(res.alternatives[0].text, text, threshold)

    merged_frame = merge_frames(frames)
    assert merged_frame.sample_rate == tts.sample_rate, "sample rate should be the same"
    assert (
        merged_frame.num_channels == tts.num_channels
    ), "num channels should be the same"


SYNTHESIZE_TTS = [elevenlabs.TTS(), openai.TTS(), google.TTS()]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts", SYNTHESIZE_TTS)
async def test_synthetize(tts: agents.tts.TTS):
    frames = []
    async for frame in tts.synthesize(text=TEST_AUDIO_SYNTHESIZE):
        frames.append(frame.data)

    await _assert_valid_synthesized_audio(
        frames, tts, TEST_AUDIO_SYNTHESIZE, SIMILARITY_THRESHOLD
    )


STREAM_TTS = [elevenlabs.TTS()]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts", STREAM_TTS)
async def test_stream(tts: agents.tts.TTS):
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

    frames = []
    assert (await stream.__anext__()).type == agents.tts.SynthesisEventType.STARTED

    async for event in stream:
        if event.type == agents.tts.SynthesisEventType.FINISHED:
            break

        assert event.type == agents.tts.SynthesisEventType.AUDIO
        assert event.audio is not None
        frames.append(event.audio.data)

    await stream.aclose(wait=True)
    await _assert_valid_synthesized_audio(
        frames, tts, TEST_AUDIO_SYNTHESIZE, SIMILARITY_THRESHOLD
    )
