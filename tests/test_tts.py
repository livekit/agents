"""
Check if all Text-To-Speech are producing valid audio.
We verify the content using a good STT model
"""

import os
import pathlib

import pytest
from livekit import agents
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.plugins import azure, cartesia, elevenlabs, google, nltk, openai

from .utils import wer

TEST_AUDIO_SYNTHESIZE = pathlib.Path(
    os.path.dirname(__file__), "long_synthesize.txt"
).read_text()
SIMILARITY_THRESHOLD = 0.9


async def _assert_valid_synthesized_audio(
    frames: AudioBuffer, tts: agents.tts.TTS, text: str, threshold: float
):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)
    whisper_stt = openai.STT(model="whisper-1")
    res = await whisper_stt.recognize(buffer=frames)
    assert wer(res.alternatives[0].text, text) <= 0.2

    merged_frame = merge_frames(frames)
    assert merged_frame.sample_rate == tts.sample_rate, "sample rate should be the same"
    assert (
        merged_frame.num_channels == tts.num_channels
    ), "num channels should be the same"


SYNTHESIZE_TTS = [
    elevenlabs.TTS(),
    elevenlabs.TTS(encoding="pcm_44100"),
    openai.TTS(),
    google.TTS(),
    azure.TTS(),
    cartesia.TTS(),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts", SYNTHESIZE_TTS)
async def test_synthetize(tts: agents.tts.TTS):
    frames = []
    async for audio in tts.synthesize(text=TEST_AUDIO_SYNTHESIZE):
        frames.append(audio.frame)

    await _assert_valid_synthesized_audio(
        frames, tts, TEST_AUDIO_SYNTHESIZE, SIMILARITY_THRESHOLD
    )


STREAM_SENT_TOKENIZER = nltk.SentenceTokenizer(min_sentence_len=20)
STREAM_TTS = [
    elevenlabs.TTS(),
    elevenlabs.TTS(encoding="pcm_44100"),
    agents.tts.StreamAdapter(
        tts=openai.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
    ),
    agents.tts.StreamAdapter(
        tts=google.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
    ),
    agents.tts.StreamAdapter(tts=azure.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER),
    agents.tts.StreamAdapter(
        tts=cartesia.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
    ),
]


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

    stream.flush()

    frames = []
    async for audio in stream:
        frames.append(audio.frame)

        if audio.end_of_segment:
            break

    await stream.aclose()
    await _assert_valid_synthesized_audio(
        frames, tts, TEST_AUDIO_SYNTHESIZE, SIMILARITY_THRESHOLD
    )
