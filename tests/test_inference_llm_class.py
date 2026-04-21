from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import llm as agents_llm
from livekit.agents.inference import LLM


class _FakeOpenAIStream:
    """Minimal stand-in for ``openai.AsyncStream`` — yields nothing and supports
    ``async with`` / ``async for``."""

    async def __aenter__(self) -> _FakeOpenAIStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _FakeOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


def _install_capture_stub(llm_instance: LLM) -> dict[str, Any]:
    """Replace ``chat.completions.create`` with a stub that records kwargs and
    returns an empty stream. Returns the dict that will hold the captured kwargs."""
    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs: Any) -> _FakeOpenAIStream:
        captured.update(kwargs)
        return _FakeOpenAIStream()

    llm_instance._client.chat.completions.create = _fake_create  # type: ignore[method-assign]
    return captured


async def _drain(stream: agents_llm.LLMStream) -> None:
    async with stream:
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_inference_class_unset_emits_no_priority_header() -> None:
    """When ``inference_class`` is unset on both the instance and the call site,
    the ``X-LiveKit-Inference-Priority`` header must not be emitted."""
    async with LLM(
        model="openai/gpt-4o-mini",
        api_key="test-key",
        api_secret="test-secret",
    ) as model:
        captured = _install_capture_stub(model)

        await _drain(model.chat(chat_ctx=agents_llm.ChatContext()))

    assert "extra_headers" in captured, "expected extra_headers to be populated"
    assert "X-LiveKit-Inference-Priority" not in captured["extra_headers"]


@pytest.mark.asyncio
async def test_inference_class_priority_emits_priority_header() -> None:
    """When ``inference_class="priority"`` is passed per-call, the
    ``X-LiveKit-Inference-Priority`` header must be emitted with value ``"priority"``."""
    async with LLM(
        model="openai/gpt-4o-mini",
        api_key="test-key",
        api_secret="test-secret",
    ) as model:
        captured = _install_capture_stub(model)

        await _drain(model.chat(chat_ctx=agents_llm.ChatContext(), inference_class="priority"))

    extra_headers = captured.get("extra_headers", {})
    assert extra_headers.get("X-LiveKit-Inference-Priority") == "priority"
