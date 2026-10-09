from collections.abc import AsyncIterator

import pytest

from livekit.agents.tokenize.utils import replace_words

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("um", ""),
        ("um UM hello um world", "  hi  world"),
        ("Um, hello um. world", ", hi . world"),
        ("unknown hello", "unknown hi"),
    ],
)
def test_empty_replacement(text: str, expected: str) -> None:
    assert replace_words(text=text, replacements={"UM": "", "hello": "hi"}) == expected


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 100])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("um", ""),
        ("um UM hello um world", "  hi  world"),
        ("Um, hello um. world", ", hi . world"),
        ("unknown hello", "unknown hi"),
    ],
)
async def test_empty_replacement_stream(text: str, expected: str, chunk_size: int) -> None:
    async def chunks() -> AsyncIterator[str]:
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    result = replace_words(text=chunks(), replacements={"UM": "", "hello": "hi"})
    assert "".join([chunk async for chunk in result]) == expected
