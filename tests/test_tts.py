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

    events = []
    segment_ids = []
    async for event in stream:
        events.append(event)
        if event.segment_id and event.segment_id not in segment_ids:
            segment_ids.append(event.segment_id)

    await stream.aclose()

    assert len(segment_ids) == 2, (
        f"Expected 2 unique segments, got {len(segment_ids)}: {segment_ids}"
    )

    seg0_id, seg1_id = segment_ids

    # Each segment should have at least one final frame
    seg0_final_indices = [
        i for i, e in enumerate(events) if e.segment_id == seg0_id and e.is_final
    ]
    seg1_final_indices = [
        i for i, e in enumerate(events) if e.segment_id == seg1_id and e.is_final
    ]
    assert seg0_final_indices, f"No final frame found for segment {seg0_id}"
    assert seg1_final_indices, f"No final frame found for segment {seg1_id}"

    # Ensure segment #0's final occurs before segment #1's final
    assert max(seg0_final_indices) < min(seg1_final_indices), (
        f"Segment #0 final (index={max(seg0_final_indices)}) did NOT occur "
        f"before segment #1 final (index={min(seg1_final_indices)})."
    )

    # Validate the synthesized audio frames
    frames = [e.frame for e in events if e.frame is not None]
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
