import pytest

from livekit.plugins import nltk

from .tokenizer_data import TEXT

pytestmark = pytest.mark.plugin("nltk")


@pytest.fixture(scope="module")
def tokenizer() -> nltk.SentenceTokenizer:
    # Model downloads belong to explicit plugin runs, never unit-test collection.
    nltk.NltkPlugin().download_files()
    return nltk.SentenceTokenizer(min_sentence_len=20)


EXPECTED_MIN_20_NLTK = [
    "Hi! LiveKit is a platform for live audio and video applications and services.",
    "R.T.C stands for Real-Time Communication... again R.T.C.",
    "Mr. Theo is testing the sentence tokenizer.",
    "This is a test. Another test.",
    "A short sentence. A longer sentence that is longer than the previous sentence.",
    "f(x) = x * 2.54 + 42.",
    "Hey! Hi! Hello! This is a sentence.",
    # nltk does not support character-based languages like CJK
    "这是一个中文句子。これは日本語の文章です。你好！LiveKit是一个直播音频和视频应用程序和服务的平台。\nThis is a sentence contains   consecutive spaces.",  # noqa: E501
]


def test_sent_tokenizer(tokenizer: nltk.SentenceTokenizer):
    assert tokenizer.tokenize(text=TEXT) == EXPECTED_MIN_20_NLTK


async def test_streamed_sent_tokenizer(tokenizer: nltk.SentenceTokenizer):
    stream = tokenizer.stream()
    pattern = [1, 2, 4]
    offset = 0
    index = 0
    while offset < len(TEXT):
        chunk_size = pattern[index % len(pattern)]
        stream.push_text(TEXT[offset : offset + chunk_size])
        offset += chunk_size
        index += 1

    stream.end_input()
    assert [event.token async for event in stream] == EXPECTED_MIN_20_NLTK
