"""Native tokenizer wrappers retain words and punctuation across streamed chunks."""

import pytest

from livekit.agents.tokenize import basic, blingfire

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Version 1.2.3 is out today.", ["Version 1.2.3 is out today."]),
        ("The server is at 192.168.1.1 right now.", ["The server is at 192.168.1.1 right now."]),
        ("One 1. Two 2. Three 3.", ["One 1.", "Two 2.", "Three 3."]),
    ],
)
async def test_sentence_stream_preserves_numbers_and_the_final_character(
    text: str, expected: list[str]
) -> None:
    tokenizer = basic.SentenceTokenizer(min_sentence_len=0, stream_context_len=1)
    assert tokenizer.tokenize(text) == expected
    stream = tokenizer.stream()
    try:
        for char in text:
            stream.push_text(char)
        stream.end_input()
        assert [event.token async for event in stream] == expected
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("مرحبا، كيف حالك؟", ["مرحبا", "كيف", "حالك"]),
        ("आप कैसे हैं।", ["आप", "कैसे", "हैं"]),
        ("你好，我很好。", ["你", "好", "我", "很", "好"]),
    ],
)
async def test_word_stream_preserves_unicode_words_and_removes_punctuation(
    text: str, expected: list[str]
) -> None:
    tokenizer = basic.WordTokenizer(ignore_punctuation=True, split_character=True)
    assert tokenizer.tokenize(text) == expected
    stream = tokenizer.stream()
    try:
        for char in text:
            stream.push_text(char)
        stream.end_input()
        assert [event.token async for event in stream] == expected
    finally:
        await stream.aclose()


async def test_blingfire_stream_emits_each_sentence_without_losing_the_tail() -> None:
    tokenizer = blingfire.SentenceTokenizer(min_sentence_len=1, stream_context_len=1)
    text = "Hello there. How are you? I am fine."
    expected = ["Hello there.", "How are you?", "I am fine."]
    assert tokenizer.tokenize(text) == expected
    stream = tokenizer.stream()
    try:
        for char in text:
            stream.push_text(char)
        stream.end_input()
        assert [event.token async for event in stream] == expected
    finally:
        await stream.aclose()
