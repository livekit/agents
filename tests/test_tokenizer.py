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


def test_supplied_tokenizer_offsets_preserve_stream_batching_without_an_event_loop() -> None:
    from collections.abc import Iterator

    from livekit.agents import _core

    def offsets(text: str) -> Iterator[tuple[str, int, int]]:
        start = 0
        for index, char in enumerate(text):
            if char == "." and (index + 1 == len(text) or text[index + 1].isspace()):
                yield text[start : index + 1].strip(), start, index + 1
                start = index + 1
        if text[start:].strip():
            yield text[start:].strip(), start, len(text)

    buffer = _core.TokenBuffer(offsets, min_token_len=20, min_ctx_len=1)
    assert buffer.push("Sure. ") == []
    assert buffer.push("Let me look. ") == []
    assert buffer.push("And one more here. Tail.") == ["Sure. Let me look. And one more here."]
    assert buffer.flush() == ["Tail."]
    assert buffer.flush() == []


def test_supplied_text_tokenizer_keeps_streaming_boundaries_without_an_event_loop() -> None:
    from livekit.agents import _core

    buffer = _core.TokenBuffer(str.split, min_token_len=1, min_ctx_len=1)
    assert buffer.push("Hello wor") == ["Hello"]
    assert buffer.push("ld  ok") == ["world"]
    assert buffer.flush() == ["ok"]


def test_supplied_tokenizer_error_returns_the_original_exception() -> None:
    from livekit.agents import _core

    error = ValueError("tokenizer failed")

    def fail(text: str) -> list[str]:
        raise error

    buffer = _core.TokenBuffer(fail, min_token_len=1, min_ctx_len=1)
    with pytest.raises(ValueError) as raised:
        buffer.push("abc def. ghi")
    assert raised.value is error


def test_xml_tokenizer_error_preserves_markup_instead_of_failing() -> None:
    from livekit.agents import _core

    def fail(text: str) -> list[str]:
        raise ValueError("tokenizer failed")

    text = "Hello <break/> world."
    buffer = _core.TokenBuffer(fail, min_token_len=1, min_ctx_len=1, xml_aware=True)
    assert buffer.push(text) == []
    assert buffer.flush() == [text]
    assert _core.xml_tokenize(fail, text) == [(text, 0, len(text))]
