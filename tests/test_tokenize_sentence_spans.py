from __future__ import annotations

import functools

import pytest

from livekit.agents.tokenize import token_stream
from livekit.agents.tokenize._basic_sent import split_sentences

pytestmark = pytest.mark.unit


def test_basic_sent_last_span_covers_whole_text() -> None:
    # Regression: the basic sentence splitter used to return len(text) - 1 as
    # the end index of the final sentence, silently excluding its last
    # character from the span.
    text = "Hello world."
    sentences = split_sentences(text, min_sentence_len=20)
    assert sentences == [("Hello world.", 0, len(text))], sentences


def test_basic_sent_last_span_multi_sentence() -> None:
    text = "This is the first sentence. Second one here."
    sentences = split_sentences(text, min_sentence_len=20)
    assert sentences[-1][2] == len(text), sentences
    assert text[sentences[-1][1] : sentences[-1][2]].strip() == sentences[-1][0]


def test_basic_sent_xml_wrapper_keeps_last_char() -> None:
    # The xml-aware wrapper remaps the sentence spans back onto the original
    # text; with the old end index the final period was split into its own
    # sentence token.
    wrapped = token_stream._xml_wrap_tokenizer(
        functools.partial(split_sentences, min_sentence_len=20)
    )
    text = "<expr type='expression' label='happy'/>Hello world."
    toks = wrapped(text)
    assert len(toks) == 1, toks
    assert toks[0][0] == text
    assert toks[0][2] == len(text)
