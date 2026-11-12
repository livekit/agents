"""Native tokenizer wrappers retain words and punctuation across streamed chunks."""

import pytest

from livekit.agents import tokenize
from livekit.agents.tokenize import TokenSpan

pytestmark = pytest.mark.unit


def test_supplied_tokenizer_offsets_preserve_stream_batching_without_an_event_loop() -> None:
    from collections.abc import Iterator

    from livekit.agents import _core

    def offsets(text: str) -> Iterator[TokenSpan]:
        start = 0
        for index, char in enumerate(text):
            if char == "." and (index + 1 == len(text) or text[index + 1].isspace()):
                yield TokenSpan(text[start : index + 1].strip(), start, index + 1)
                start = index + 1
        if text[start:].strip():
            yield TokenSpan(text[start:].strip(), start, len(text))

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


@pytest.mark.parametrize(
    "tokenizer",
    [
        tokenize.SentenceTokenizer(),
        tokenize.SentenceTokenizer(min_words=1, clause_words=4, max_token_len=40),
        tokenize.SentenceTokenizer(retain_format=True),
    ],
)
async def test_the_sentence_stream_says_what_tokenize_says(
    tokenizer: tokenize.SentenceTokenizer,
) -> None:
    text = (
        "Sure. Let me look that up for you, it should only take a moment. "
        "The total is 1,000 dollars, due at 12:30; after that, we close.\n\n"
        "这是一个中文句子，很长很长。Okay!"
    )
    stream = tokenizer.stream()
    try:
        for char in text:
            stream.push_text(char)
        stream.end_input()
        assert [event.token async for event in stream] == tokenizer.tokenize(text)
    finally:
        await stream.aclose()


async def test_a_sentence_streams_before_the_input_ends() -> None:
    stream = tokenize.SentenceTokenizer().stream()
    try:
        stream.push_text("This first sentence is long enough. And")
        assert (await anext(stream)).token == "This first sentence is long enough."
    finally:
        await stream.aclose()
