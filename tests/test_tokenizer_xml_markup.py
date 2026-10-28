"""Markup stays intact while the public sentence tokenizer streams."""

import asyncio

import pytest

from livekit.agents.tokenize.blingfire import SentenceTokenizer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("streaming", [False, True])
async def test_wrapping_markup_is_not_split_at_its_inner_sentence_boundary(streaming: bool) -> None:
    tokenizer = SentenceTokenizer(min_sentence_len=1, stream_context_len=5, xml_aware=True)
    text = "Read <spell>First sentence. Second sentence.</spell> now. Then continue."
    if streaming:
        stream = tokenizer.stream()
        try:
            for char in text:
                stream.push_text(char)
            stream.end_input()
            tokens = [event.token async for event in stream]
        finally:
            await stream.aclose()
    else:
        tokens = tokenizer.tokenize(text)
    containing = [token for token in tokens if "<spell>" in token]
    assert len(containing) == 1
    assert "First sentence. Second sentence.</spell>" in containing[0]


@pytest.mark.parametrize(
    "text",
    [
        "Note that 3 < 5 holds. And here is a second sentence to tokenize.",
        "Rate this from <1> to <5>. And here is a second sentence to split.",
    ],
)
async def test_angle_brackets_in_prose_do_not_hold_streaming_until_flush(text: str) -> None:
    tokenizer = SentenceTokenizer(min_sentence_len=1, stream_context_len=5, xml_aware=True)
    stream = tokenizer.stream()
    try:
        stream.push_text(text)
        first = await asyncio.wait_for(anext(stream), timeout=1)
        assert first.token == text.split(". ")[0] + "."
    finally:
        await stream.aclose()


async def test_a_markup_only_sound_is_emitted_at_end_input() -> None:
    tokenizer = SentenceTokenizer(min_sentence_len=1, xml_aware=True)
    stream = tokenizer.stream()
    try:
        stream.push_text('<expression value="laugh"/>')
        stream.end_input()
        assert [event.token async for event in stream] == ['<expression value="laugh"/>']
    finally:
        await stream.aclose()
