from collections.abc import AsyncIterator
from typing import cast

import pytest

from livekit.agents.tokenize import replace_words

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


def test_case_colliding_replacements_follow_dictionary_insertion_order() -> None:
    assert replace_words(text="THE", replacements={"the": "first", "THE": "second"}) == "second"
    assert replace_words(text="THE", replacements={"THE": "second", "the": "first"}) == "first"


@pytest.mark.parametrize(
    "value", [float("nan"), 1 << 100, object()], ids=["nan", "large-integer", "object"]
)
def test_replacements_reject_non_string_values_with_type_error(value: object) -> None:
    with pytest.raises(TypeError, match="str|string"):
        replace_words(text="hello", replacements={"hello": cast(str, value)})
