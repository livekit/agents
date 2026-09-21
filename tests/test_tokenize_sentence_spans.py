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


def test_basic_sent_keeps_dotted_numbers_in_one_sentence() -> None:
    # Regression: only the first dot of a dotted number was protected, because
    # re.sub does not return to overlapping positions ("1.2.3" protected "1.2"
    # and left ".3" exposed). The remaining dot then ended the sentence, so a
    # version number or an IP address was split across two sentence tokens and
    # streamed to TTS as "1.2." + "3 is out today.".
    for text in (
        "Version 1.2.3 is out today.",
        "The server is at 192.168.1.1 right now.",
    ):
        sentences = split_sentences(text, min_sentence_len=20)
        assert sentences == [(text, 0, len(text))], sentences


def test_basic_sent_still_splits_after_a_number() -> None:
    # The protection must stay limited to dots between digits: a sentence ending
    # with a number still ends there.
    text = "One 1. Two 2. Three 3."
    sentences = split_sentences(text, min_sentence_len=0)
    assert [sent for sent, _, _ in sentences] == ["One 1.", "Two 2.", "Three 3."]
    for sent, start, end in sentences:
        assert text[start:end].strip() == sent


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
