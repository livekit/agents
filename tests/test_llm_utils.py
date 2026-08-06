from __future__ import annotations

import pytest
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm.utils import ThinkingTokenFilter, strip_thinking_tokens

pytestmark = pytest.mark.unit


GEMMA_THINK_TAGS = ("<|channel>thought", "<channel|>")


def _collect_visible_text(
    chunks: list[str | None], *, think_tags: tuple[str, str] | None = None
) -> str:
    state = ThinkingTokenFilter(*think_tags) if think_tags else ThinkingTokenFilter()
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

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "answer"


def test_strips_empty_gemma_reasoning_block() -> None:
    chunks = ["<|channel>thought\n<channel|>answer"]

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "answer"


def test_strips_gemma_reasoning_across_chunks() -> None:
    chunks = ["<|channel>thought\n", "private reasoning", "<channel|>", "answer"]

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "answer"


def test_preserves_answer_after_streamed_gemma_closing_marker() -> None:
    chunks = ["<|channel>thought\n", "private reasoning", "<channel|>answer"]

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "answer"


def test_strips_multiple_gemma_reasoning_blocks() -> None:
    chunks = [
        "<|channel>thought\nfirst thought<channel|>first answer; ",
        "<|channel>thought\nsecond thought<channel|>second answer",
    ]

    assert (
        _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "first answer; second answer"
    )


def test_handles_gemma_markers_split_at_arbitrary_boundaries() -> None:
    chunks = list("<|channel>thought\nprivate reasoning<channel|>answer")

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "answer"


def test_preserves_visible_text_before_gemma_reasoning() -> None:
    chunks = ["Let me check that.\n\n<|channel>thought\n<channel|>"]

    assert _collect_visible_text(chunks, think_tags=GEMMA_THINK_TAGS) == "Let me check that.\n\n"


def test_preserves_gemma_markers_without_model_configuration() -> None:
    content = "<|channel>thought\nprivate reasoning<channel|>answer"

    assert _collect_visible_text([content]) == content


def test_preserves_existing_think_token_behavior() -> None:
    chunks = ["<think>", "private reasoning", "</think>answer"]

    assert _collect_visible_text(chunks) == "answer"


def test_preserves_incomplete_marker_at_end_of_stream() -> None:
    assert (
        _collect_visible_text(["literal <|chan"], think_tags=GEMMA_THINK_TAGS) == "literal <|chan"
    )


def test_drops_unclosed_reasoning_at_end_of_stream() -> None:
    assert (
        _collect_visible_text(
            ["before<|channel>thought\nprivate reasoning"], think_tags=GEMMA_THINK_TAGS
        )
        == "before"
    )


def test_flattens_list_shaped_delta_content() -> None:
    # OpenAI-compatible providers (e.g. Mistral) can emit delta.content as a
    # list of typed content parts instead of a string; this used to crash the
    # session with AttributeError/TypeError (#6323)
    state = ThinkingTokenFilter()
    assert strip_thinking_tokens([{"type": "text", "text": "Hallo"}], state) == "Hallo"


def test_flattens_multiple_and_string_parts() -> None:
    state = ThinkingTokenFilter()
    content = strip_thinking_tokens(
        [{"type": "text", "text": "Hello "}, "from ", {"type": "text", "text": "LiveKit"}],
        state,
    )
    assert content == "Hello from LiveKit"


def test_strips_thinking_tokens_inside_list_parts() -> None:
    state = ThinkingTokenFilter()
    visible = []
    for chunk in (
        [{"type": "text", "text": "<think>private "}],
        [{"type": "text", "text": "reasoning</think>answer"}],
    ):
        content = strip_thinking_tokens(chunk, state)
        if content is not None:
            visible.append(content)
    content = strip_thinking_tokens(None, state, final=True)
    if content is not None:
        visible.append(content)
    assert "".join(visible) == "answer"


def test_list_parts_without_text_are_ignored() -> None:
    state = ThinkingTokenFilter()
    content = strip_thinking_tokens(
        [{"type": "image_url", "image_url": {"url": "https://x"}}, {"type": "text", "text": "hi"}],
        state,
    )
    assert content == "hi"


def test_list_without_any_text_part_carries_no_content() -> None:
    # nothing textual arrived, so report "no content" rather than an empty
    # string the provider never sent
    state = ThinkingTokenFilter()
    assert strip_thinking_tokens([], state) is None
    assert (
        strip_thinking_tokens([{"type": "image_url", "image_url": {"url": "https://x"}}], state)
        is None
    )


def test_an_empty_text_part_is_kept_as_empty_content() -> None:
    # the provider did send text, it is just empty - same as a plain "" delta
    state = ThinkingTokenFilter()
    assert strip_thinking_tokens([{"type": "text", "text": ""}], state) == ""


def test_object_parts_with_text_attribute_are_flattened() -> None:
    class _Part:
        text = "typed part"

    state = ThinkingTokenFilter()
    assert strip_thinking_tokens([_Part()], state) == "typed part"


def test_strips_reasoning_from_text_alongside_tool_call() -> None:
    stream = LLMStream.__new__(LLMStream)
    stream._tool_call_id = None
    stream._fnc_name = None
    stream._fnc_raw_arguments = None
    stream._tool_extra = None
    stream._tool_index = None
    choice = Choice(
        index=0,
        finish_reason="tool_calls",
        delta=ChoiceDelta(
            content="Let me check that.\n\n<|channel>thought\n<channel|>",
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call-1",
                    function=ChoiceDeltaToolCallFunction(name="check", arguments="{}"),
                )
            ],
        ),
    )

    chunk = stream._parse_choice("chunk-1", choice, ThinkingTokenFilter(*GEMMA_THINK_TAGS))

    assert chunk is not None
    assert chunk.delta is not None
    assert chunk.delta.content == "Let me check that.\n\n"
