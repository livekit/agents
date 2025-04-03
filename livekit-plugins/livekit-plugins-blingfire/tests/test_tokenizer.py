import pytest

from livekit.plugins.blingfire import SentenceTokenizer


@pytest.fixture
def tokenizer():
    return SentenceTokenizer(min_sentence_len=1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world!", ["Hello, world!"]),
        (
            "Hello, world! Should this work?? This should work, too... How about this?",
            ["Hello, world!", "Should this work??", "This should work, too...", "How about this?"],
        ),
        ("你好，世界！ 这是一句测试文本。", ["你好，世界！", "这是一句测试文本。"]),
    ],
)
def test_tokenizer(tokenizer: SentenceTokenizer, text: str, expected: list[str]):
    assert tokenizer.tokenize(text) == expected
