from __future__ import annotations

from typing import TypedDict

import pytest
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from typing_extensions import NotRequired, Required

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


class UserProfile(TypedDict):
    name: Required[str]
    age: NotRequired[int]


class OptionalProfile(TypedDict, total=False):
    bio: str


def test_to_response_format_param_typed_dict() -> None:
    from livekit.agents.llm.utils import to_openai_response_format, to_response_format_param

    name, model_cls = to_response_format_param(UserProfile)
    assert name == "UserProfile"
    schema = model_cls.model_json_schema()
    assert "name" in schema.get("required", [])
    assert "age" not in schema.get("required", [])

    name_opt, opt_cls = to_response_format_param(OptionalProfile)
    assert name_opt == "OptionalProfile"
    opt_schema = opt_cls.model_json_schema()
    assert "bio" not in opt_schema.get("required", [])

    openai_format = to_openai_response_format(UserProfile)
    assert openai_format["type"] == "json_schema"
    assert openai_format["json_schema"]["name"] == "UserProfile"
