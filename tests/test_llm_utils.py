from __future__ import annotations

import pytest

from livekit.agents.llm.utils import ThinkingTokenFilter, strip_thinking_tokens

pytestmark = pytest.mark.unit


def _collect_visible_text(chunks: list[str | None]) -> str:
    state = ThinkingTokenFilter()
    visible = []

    for chunk in chunks:
        content = strip_thinking_tokens(chunk, state)
        if content is not None:
            visible.append(content)

    content = strip_thinking_tokens(None, state, final=True)
    if content is not None:
        visible.append(content)

    return "".join(visible)


def test_preserves_content_without_thinking_tokens() -> None:
    assert _collect_visible_text([None, "", "Hello from LiveKit"]) == "Hello from LiveKit"


def test_strips_complete_gemma_reasoning_block() -> None:
    chunks = ["<|channel>thought\nprivate reasoning\n<channel|>answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_strips_empty_gemma_reasoning_block() -> None:
    chunks = ["<|channel>thought\n<channel|>answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_strips_gemma_reasoning_across_chunks() -> None:
    chunks = ["<|channel>thought\n", "private reasoning", "<channel|>", "answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_preserves_answer_after_streamed_gemma_closing_marker() -> None:
    chunks = ["<|channel>thought\n", "private reasoning", "<channel|>answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_strips_multiple_gemma_reasoning_blocks() -> None:
    chunks = [
        "<|channel>thought\nfirst thought<channel|>first answer; ",
        "<|channel>thought\nsecond thought<channel|>second answer",
    ]

    assert _collect_visible_text(chunks) == "first answer; second answer"


def test_handles_gemma_markers_split_at_arbitrary_boundaries() -> None:
    chunks = list("<|channel>thought\nprivate reasoning<channel|>answer")

    assert _collect_visible_text(chunks) == "answer"


def test_preserves_visible_text_before_gemma_reasoning() -> None:
    chunks = ["Let me check that.\n\n<|channel>thought\n<channel|>"]

    assert _collect_visible_text(chunks) == "Let me check that.\n\n"


def test_preserves_existing_think_token_behavior() -> None:
    chunks = ["<think>", "private reasoning", "</think>answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_preserves_incomplete_marker_at_end_of_stream() -> None:
    assert _collect_visible_text(["literal <|chan"]) == "literal <|chan"


def test_drops_unclosed_reasoning_at_end_of_stream() -> None:
    assert _collect_visible_text(["before<|channel>thought\nprivate reasoning"]) == "before"
