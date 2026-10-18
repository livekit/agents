"""
Tests for Gemini 4 (argon) in the Google LLM plugin: thinking is configured with a level
rather than a token budget, sampling parameters are dropped, the output ceiling is pinned
because the surfaces disagree on a default, and a turn too large for one request is resumed
through continuation tokens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai import types

from livekit.agents import llm as agents_llm
from livekit.plugins.google.llm import (
    LLM,
    LONG_DECODE_CONTEXT_WINDOW,
    LONG_DECODE_HTTP_TIMEOUT_MS,
    LONG_DECODE_MAX_REQUESTS,
    LONG_DECODE_MIN_OUTPUT_TOKENS,
    _ignores_sampling_params,
    _is_gemini_4_model,
    _requires_thought_signatures,
    _supports_thinking_level,
)

pytestmark = pytest.mark.unit

ARGON = "gemini-4-argon"


def _llm(model: str = ARGON, **kwargs: Any) -> LLM:
    return LLM(model=model, api_key="test-key", **kwargs)


def _response(
    *,
    text: str | None = None,
    finish_reason: types.FinishReason | None = None,
    continuation_token: bytes | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> types.GenerateContentResponse:
    candidate = types.Candidate(
        content=types.Content(parts=[types.Part(text=text)]) if text is not None else None,
        finish_reason=finish_reason,
        continuation_token=continuation_token,
    )
    return types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
                total_token_count=(prompt_tokens or 0) + (completion_tokens or 0),
            )
            if prompt_tokens is not None or completion_tokens is not None
            else None
        ),
    )


class _FakeStreams:
    """Serves one scripted stream per call and records the config it was given."""

    def __init__(self, hops: list[list[types.GenerateContentResponse]]) -> None:
        self._hops = hops
        self.configs: list[types.GenerateContentConfig] = []

    async def __call__(self, **kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        self.configs.append(kwargs["config"])
        responses = self._hops[min(len(self.configs) - 1, len(self._hops) - 1)]

        async def _iter() -> AsyncIterator[types.GenerateContentResponse]:
            for response in responses:
                yield response

        return _iter()


async def _request_config(model: LLM, **chat_kwargs: Any) -> dict[str, Any]:
    """The config `chat()` built, with the stream closed again."""
    async with model.chat(chat_ctx=agents_llm.ChatContext.empty(), **chat_kwargs) as stream:
        return dict(stream._extra_kwargs)


async def _collect(model: LLM, streams: _FakeStreams) -> list[str]:
    text, _ = await _collect_with_usage(model, streams)
    return text


async def _collect_with_usage(
    model: LLM, streams: _FakeStreams
) -> tuple[list[str], agents_llm.CompletionUsage | None]:
    """The text chunks, and the usage a reader keeping only the last one would end up with."""
    model._client.aio.models.generate_content_stream = streams  # type: ignore[method-assign]
    chunks: list[str] = []
    usage: agents_llm.CompletionUsage | None = None
    async with model.chat(chat_ctx=agents_llm.ChatContext.empty()) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)
            if chunk.usage is not None:
                usage = chunk.usage
    return chunks, usage


class TestModelDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-4-argon", True),
            # the EAP name is not a listed id, but routes the same if passed as a raw string
            ("gemini-4-argon-eap", True),
            ("gemini-3-pro-preview", False),
            ("gemini-2.5-flash", False),
            ("gemma-4-27b-it", False),
        ],
    )
    def test_is_gemini_4(self, model: str, expected: bool) -> None:
        assert _is_gemini_4_model(model) is expected

    async def test_gemini_4_joins_the_gates_gemini_3_already_passed(self) -> None:
        # thinking_budget returns 400 on argon, so it must take the level path
        assert _supports_thinking_level(ARGON)
        assert _requires_thought_signatures(ARGON)

    async def test_only_gemini_4_drops_sampling_params(self) -> None:
        assert _ignores_sampling_params(ARGON)
        assert not _ignores_sampling_params("gemini-3-pro-preview")


class TestRequestShaping:
    async def test_sampling_params_are_dropped(self) -> None:
        extra = await _request_config(_llm(temperature=0.8, top_p=0.9, top_k=40))
        assert "temperature" not in extra
        assert "top_p" not in extra
        assert "top_k" not in extra

    async def test_sampling_params_stay_dropped_across_calls(self) -> None:
        # they are unset on the options once, so every later request is already clean
        model = _llm(temperature=0.8, top_p=0.9, top_k=40)
        for _ in range(3):
            extra = await _request_config(model)
            assert not {"temperature", "top_p", "top_k"} & set(extra)

    async def test_sampling_params_survive_on_older_models(self) -> None:
        extra = await _request_config(_llm("gemini-3-pro-preview", temperature=0.8))
        assert extra["temperature"] == 0.8

    async def test_the_output_ceiling_is_pinned_when_unset(self) -> None:
        extra = await _request_config(_llm())
        assert extra["max_output_tokens"] == LONG_DECODE_MIN_OUTPUT_TOKENS

    async def test_an_explicit_ceiling_is_left_alone(self) -> None:
        # a voice agent wanting short replies keeps them; the risk is logged, not overridden
        extra = await _request_config(_llm(max_output_tokens=256))
        assert extra["max_output_tokens"] == 256

    async def test_older_models_keep_their_own_default(self) -> None:
        extra = await _request_config(_llm("gemini-2.5-flash"))
        assert "max_output_tokens" not in extra

    async def test_a_thinking_budget_is_replaced_by_a_level(self) -> None:
        extra = await _request_config(
            _llm(thinking_config=types.ThinkingConfig(thinking_budget=1024, thinking_level="high"))
        )
        cfg = extra["thinking_config"]
        assert cfg.thinking_budget is None
        assert cfg.thinking_level == types.ThinkingLevel.HIGH


class TestLongDecode:
    async def test_the_sdk_loop_is_turned_off_so_this_one_can_bound_the_turn(self) -> None:
        streams = _FakeStreams([[_response(text="hi", finish_reason=types.FinishReason.STOP)]])
        await _collect(_llm(), streams)
        assert streams.configs[0].automatic_continuation is False

    async def test_older_models_keep_the_sdk_default(self) -> None:
        streams = _FakeStreams([[_response(text="hi", finish_reason=types.FinishReason.STOP)]])
        await _collect(_llm("gemini-2.5-flash"), streams)
        assert streams.configs[0].automatic_continuation is None

    async def test_a_continuation_is_resumed_with_its_token(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="part one "),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text="part two", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["part one ", "part two"]
        assert len(streams.configs) == 2
        assert streams.configs[0].continuation_token is None
        assert streams.configs[1].continuation_token == b"tok-1"
        # the budget is the caller's, and a split turn is no reason to spend more of it
        assert streams.configs[1].max_output_tokens == LONG_DECODE_MIN_OUTPUT_TOKENS

    async def test_requests_are_capped(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="x"),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok"
                    ),
                ],
            ]
        )
        await _collect(_llm(), streams)
        assert len(streams.configs) == LONG_DECODE_MAX_REQUESTS

    async def test_a_ceiling_this_plugin_picked_is_raised_to_finish(self) -> None:
        # the caller asked for no limit, so the one picked for them is not a reason to truncate
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off", " and more"]
        assert streams.configs[1].continuation_token == b"tok-1"
        assert [c.max_output_tokens for c in streams.configs] == [
            LONG_DECODE_MIN_OUTPUT_TOKENS,
            LONG_DECODE_MIN_OUTPUT_TOKENS * 2,
        ]

    async def test_a_ceiling_the_caller_set_is_not_raised(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(max_output_tokens=256), streams) == ["cut off"]
        assert [c.max_output_tokens for c in streams.configs] == [256]

    async def test_a_ceiling_from_extra_kwargs_counts_as_the_caller_s(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        model = _llm()
        model._client.aio.models.generate_content_stream = streams  # type: ignore[method-assign]
        async with model.chat(
            chat_ctx=agents_llm.ChatContext.empty(), extra_kwargs={"max_output_tokens": 500}
        ) as stream:
            async for _ in stream:
                pass
        assert [c.max_output_tokens for c in streams.configs] == [500]

    async def test_the_budget_is_not_raised_without_a_prompt_count(self) -> None:
        # with no count there is no way to tell what the window leaves, and a request over it
        # would fail after this turn has already spoken
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off"),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off"]
        assert len(streams.configs) == 1

    async def test_the_budget_cannot_outgrow_what_the_prompt_leaves(self) -> None:
        prompt_tokens = LONG_DECODE_CONTEXT_WINDOW - LONG_DECODE_MIN_OUTPUT_TOKENS
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=prompt_tokens),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off"]
        assert len(streams.configs) == 1

    async def test_a_token_alongside_another_reason_does_not_resume(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="done"),
                    _response(finish_reason=types.FinishReason.STOP, continuation_token=b"tok-1"),
                ],
            ]
        )
        assert await _collect(_llm(), streams) == ["done"]
        assert len(streams.configs) == 1

    async def test_older_models_never_resume(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="done"),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok"
                    ),
                ],
            ]
        )
        assert await _collect(_llm("gemini-2.5-flash"), streams) == ["done"]
        assert len(streams.configs) == 1

    async def test_the_read_timeout_covers_a_long_reasoning_turn(self) -> None:
        streams = _FakeStreams([[_response(text="hi", finish_reason=types.FinishReason.STOP)]])
        await _collect(_llm(), streams)
        http_options = streams.configs[0].http_options
        assert http_options is not None
        assert http_options.timeout == LONG_DECODE_HTTP_TIMEOUT_MS


class TestStreamCheckpoint:
    async def test_a_stream_that_dies_after_a_checkpoint_resumes(self) -> None:
        class _DyingThenFine(_FakeStreams):
            async def __call__(self, **kwargs: Any) -> Any:
                self.configs.append(kwargs["config"])
                if len(self.configs) == 1:

                    async def _die() -> AsyncIterator[types.GenerateContentResponse]:
                        yield _response(text="first half", continuation_token=b"ckpt")
                        raise RuntimeError("connection reset")

                    return _die()

                async def _finish() -> AsyncIterator[types.GenerateContentResponse]:
                    yield _response(text=" second half", finish_reason=types.FinishReason.STOP)

                return _finish()

        streams = _DyingThenFine([])
        assert await _collect(_llm(), streams) == ["first half", " second half"]
        assert streams.configs[1].continuation_token == b"ckpt"

    async def test_a_stream_that_just_stops_after_a_checkpoint_resumes(self) -> None:
        # no error, no finish reason: the request never finished, so the checkpoint resumes it
        class _StopsShort(_FakeStreams):
            async def __call__(self, **kwargs: Any) -> Any:
                self.configs.append(kwargs["config"])
                if len(self.configs) == 1:

                    async def _short() -> AsyncIterator[types.GenerateContentResponse]:
                        yield _response(text="first half", continuation_token=b"ckpt")

                    return _short()

                async def _finish() -> AsyncIterator[types.GenerateContentResponse]:
                    yield _response(text=" second half", finish_reason=types.FinishReason.STOP)

                return _finish()

        streams = _StopsShort([])
        assert await _collect(_llm(), streams) == ["first half", " second half"]
        assert streams.configs[1].continuation_token == b"ckpt"

    async def test_a_stream_that_stops_with_nothing_to_resume_from_stands(self) -> None:
        streams = _FakeStreams([[_response(text="all there was")]])
        assert await _collect(_llm(), streams) == ["all there was"]
        assert len(streams.configs) == 1

    async def test_a_stream_that_dies_without_one_surfaces_the_error(self) -> None:
        class _JustDies(_FakeStreams):
            async def __call__(self, **kwargs: Any) -> Any:
                self.configs.append(kwargs["config"])

                async def _die() -> AsyncIterator[types.GenerateContentResponse]:
                    yield _response(text="partial")
                    raise RuntimeError("connection reset")

                return _die()

        with pytest.raises(Exception, match="connection reset"):
            await _collect(_llm(), _JustDies([]))


class TestUsageAcrossRequests:
    async def test_the_last_chunk_covers_the_whole_turn(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="one", prompt_tokens=100, completion_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok-1"
                    ),
                ],
                [
                    _response(
                        text=" two",
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=1000,
                    )
                ],
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        # a reader keeping only the last chunk would otherwise see one request's worth
        assert usage.completion_tokens == 2000
        assert usage.prompt_tokens == 200
        assert usage.total_tokens == 2200

    async def test_a_single_request_reports_what_the_api_reported(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(
                        text="hi",
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=50,
                    )
                ]
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (100, 50)

    async def test_restated_counters_within_a_request_are_not_added_up(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="a", prompt_tokens=100, completion_tokens=10),
                    _response(text="b", prompt_tokens=100, completion_tokens=20),
                    _response(
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=30,
                    ),
                ]
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (100, 30)


class TestExtraKwargs:
    async def test_a_ceiling_passed_through_extra_kwargs_is_not_overwritten(self) -> None:
        extra = await _request_config(_llm(), extra_kwargs={"max_output_tokens": 500})
        assert extra["max_output_tokens"] == 500

    async def test_sampling_params_passed_through_extra_kwargs_are_dropped(self) -> None:
        extra = await _request_config(
            _llm(), extra_kwargs={"temperature": 0.7, "top_p": 0.5, "top_k": 10}
        )
        assert not {"temperature", "top_p", "top_k"} & set(extra)

    async def test_the_sdk_loop_cannot_be_handed_back_through_extra_kwargs(self) -> None:
        # it is the unbounded one; a caller re-enabling it would uncap the turn
        streams = _FakeStreams([[_response(text="hi", finish_reason=types.FinishReason.STOP)]])
        model = _llm()
        model._client.aio.models.generate_content_stream = streams  # type: ignore[method-assign]
        async with model.chat(
            chat_ctx=agents_llm.ChatContext.empty(),
            extra_kwargs={"automatic_continuation": True},
        ) as stream:
            async for _ in stream:
                pass
        assert streams.configs[0].automatic_continuation is False
