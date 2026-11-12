"""NLTK splitting uses the shared stream without sharing conversation buffers."""

from __future__ import annotations

import asyncio

import nltk
import pytest

from livekit.plugins.nltk import SentenceTokenizer

pytestmark = pytest.mark.unit


async def test_nltk_stream_close_and_flush_leave_another_stream_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def split(text: str, language: str) -> list[str]:
        return [text.strip()] if text.strip() else []

    monkeypatch.setattr(nltk.tokenize, "sent_tokenize", split)
    tokenizer = SentenceTokenizer()
    first = tokenizer.stream()
    second = tokenizer.stream()
    try:
        first.push_text("First conversation.")
        first.flush()
        assert (await anext(first)).token == "First conversation."

        second.push_text("Second conversation.")
        reader = asyncio.create_task(anext(first))
        try:
            await first.aclose()
            with pytest.raises(StopAsyncIteration):
                await reader
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

        second.flush()
        second.push_text("Still here.")
        second.end_input()
        assert [event.token async for event in second] == ["Second conversation.", "Still here."]
    finally:
        await first.aclose()
        await second.aclose()
