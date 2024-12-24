"""
Check if all Text-To-Speech are producing valid audio.
We verify the content using a good STT model
"""

import dataclasses
from typing import Callable

import pytest
from livekit import agents
from livekit.agents import APIConnectionError, tokenize, tts
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.plugins import (
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    openai,
    playai,
)

from .conftest import TEST_CONNECT_OPTIONS
from .fake_tts import FakeTTS
from .utils import make_test_synthesize, wer

WER_THRESHOLD = 0.2


async def _assert_valid_synthesized_audio(
    frames: AudioBuffer, tts: agents.tts.TTS, text: str, threshold: float
):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)
    whisper_stt = openai.STT(model="whisper-1")
    res = await whisper_stt.recognize(buffer=frames)
    assert wer(res.alternatives[0].text, text) <= threshold

    merged_frame = merge_frames(frames)
    assert merged_frame.sample_rate == tts.sample_rate, "sample rate should be the same"
    assert (
        merged_frame.num_channels == tts.num_channels
    ), "num channels should be the same"


SYNTHESIZE_TTS: list[Callable[[], tts.TTS]] = [
    pytest.param(lambda: elevenlabs.TTS(), id="elevenlabs"),
    pytest.param(
        lambda: elevenlabs.TTS(encoding="pcm_44100"), id="elevenlabs.pcm_44100"
    ),
    pytest.param(lambda: openai.TTS(), id="openai"),
    pytest.param(lambda: google.TTS(), id="google"),
    pytest.param(lambda: azure.TTS(), id="azure"),
    pytest.param(lambda: cartesia.TTS(), id="cartesia"),
    pytest.param(lambda: deepgram.TTS(), id="deepgram"),
    pytest.param(lambda: playai.TTS(), id="playai"),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize(tts_factory):
    tts = tts_factory()

    synthesize_transcript = make_test_synthesize()

    frames = []
    async for audio in tts.synthesize(text=synthesize_transcript):
        frames.append(audio.frame)

    await _assert_valid_synthesized_audio(
        frames, tts, synthesize_transcript, WER_THRESHOLD
    )


STREAM_SENT_TOKENIZER = tokenize.basic.SentenceTokenizer(min_sentence_len=20)
STREAM_TTS: list[Callable[[], tts.TTS]] = [
    pytest.param(lambda: elevenlabs.TTS(), id="elevenlabs"),
    pytest.param(
        lambda: elevenlabs.TTS(encoding="pcm_44100"), id="elevenlabs.pcm_44100"
    ),
    pytest.param(lambda: cartesia.TTS(), id="cartesia"),
    pytest.param(
        lambda: agents.tts.StreamAdapter(
            tts=openai.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
        ),
        id="openai.stream",
    ),
    pytest.param(
        lambda: agents.tts.StreamAdapter(
            tts=google.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
        ),
        id="google.stream",
    ),
    pytest.param(
        lambda: agents.tts.StreamAdapter(
            tts=azure.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
        ),
        id="azure.stream",
    ),
    pytest.param(lambda: deepgram.TTS(), id="deepgram"),
    pytest.param(lambda: playai.TTS(), id="playai"),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", STREAM_TTS)
async def test_stream(tts_factory):
    tts: agents.tts.TTS = tts_factory()

    synthesize_transcript = make_test_synthesize()

    pattern = [1, 2, 4]
    text = synthesize_transcript
    chunks = []
    pattern_iter = iter(pattern * (len(text) // sum(pattern) + 1))

    for chunk_size in pattern_iter:
        if not text:
            break
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    stream = tts.stream()

    segments = set()
    # for i in range(2): # TODO(theomonnom): we should test 2 segments
    for chunk in chunks:
        stream.push_text(chunk)

    stream.flush()
    # if i == 1:
    stream.end_input()

    frames = []
    is_final = False
    async for audio in stream:
        is_final = audio.is_final
        segments.add(audio.segment_id)
        frames.append(audio.frame)

    assert is_final, "final audio should be marked as final"

    await _assert_valid_synthesized_audio(
        frames, tts, synthesize_transcript, WER_THRESHOLD
    )

    # assert len(segments) == 2
    await stream.aclose()


async def test_retry():
    fake_tts = FakeTTS(fake_exception=APIConnectionError("fake exception"))

    retry_options = dataclasses.replace(TEST_CONNECT_OPTIONS, max_retry=3)
    stream = fake_tts.synthesize("testing", conn_options=retry_options)

    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass

    assert fake_tts.synthesize_ch.recv_nowait()
    assert stream.attempt == 4


async def test_close():
    fake_tts = FakeTTS(fake_timeout=5.0)

    retry_options = dataclasses.replace(TEST_CONNECT_OPTIONS, max_retry=0)
    stream = fake_tts.synthesize("testing", conn_options=retry_options)

    await stream.aclose()

    async for _ in stream:
        pass
