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
    aws,
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    openai,
    playai,
    rime,
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
    assert merged_frame.num_channels == tts.num_channels, (
        "num channels should be the same"
    )


SYNTHESIZE_TTS: list[Callable[[], tts.TTS]] = [
    pytest.param(lambda: elevenlabs.TTS(), id="elevenlabs"),
    pytest.param(
        lambda: elevenlabs.TTS(encoding="pcm_44100"), id="elevenlabs.pcm_44100"
    ),
    pytest.param(lambda: openai.TTS(), id="openai"),
    pytest.param(lambda: google.TTS(), id="google"),
    pytest.param(lambda: azure.TTS(), id="azure"),
    pytest.param(lambda: aws.TTS(), id="aws"),
    pytest.param(lambda: cartesia.TTS(), id="cartesia"),
    pytest.param(lambda: deepgram.TTS(), id="deepgram"),
    pytest.param(lambda: playai.TTS(), id="playai"),
    pytest.param(lambda: rime.TTS(), id="rime"),
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
    pytest.param(
        lambda: agents.tts.StreamAdapter(
            tts=aws.TTS(), sentence_tokenizer=STREAM_SENT_TOKENIZER
        ),
        id="aws.stream",
    ),
]


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", STREAM_TTS)
async def test_stream(tts_factory):
    tts: agents.tts.TTS = tts_factory()

    synthesize_transcript = make_test_synthesize()

    # Split the transcript into two segments
    text_segments = [
        synthesize_transcript[: len(synthesize_transcript) // 2],
        synthesize_transcript[len(synthesize_transcript) // 2 :],
    ]

    stream = tts.stream()

    pattern = [1, 2, 4]
    for i, text in enumerate(text_segments):
        text_remaining = text
        chunk_iter = iter(pattern * (len(text) // sum(pattern) + 1))

        while text_remaining:
            size = next(chunk_iter)
            stream.push_text(text_remaining[:size])
            text_remaining = text_remaining[size:]

        stream.flush()
        if i == 1:
            stream.end_input()

    segment_ids = []
    final_indices = {}
    frames = []
    idx = 0
    async for event in stream:
        if event.frame is not None:
            frames.append(event.frame)
        if event.segment_id:
            if event.segment_id not in segment_ids:
                segment_ids.append(event.segment_id)
            if event.is_final:
                if event.segment_id in final_indices:
                    raise AssertionError(
                        f"Multiple final events for segment {event.segment_id}"
                    )
                final_indices[event.segment_id] = idx
        idx += 1
    await stream.aclose()

    assert len(segment_ids) == 2, (
        f"Expected 2 segments, got {len(segment_ids)}: {segment_ids}"
    )
    seg0, seg1 = segment_ids

    # Assert each segment has exactly one final event and ordering is correct.
    assert seg0 in final_indices, f"No final event for segment {seg0}"
    assert seg1 in final_indices, f"No final event for segment {seg1}"
    assert final_indices[seg0] < final_indices[seg1], (
        f"Segment {seg0} final event (index={final_indices[seg0]}) did not occur before "
        f"segment {seg1} final event (index={final_indices[seg1]})."
    )

    expected_text = "".join(text_segments)
    await _assert_valid_synthesized_audio(frames, tts, expected_text, WER_THRESHOLD)


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
