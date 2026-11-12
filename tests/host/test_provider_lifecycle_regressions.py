"""A Python provider's lifecycle through the native session: prewarm, the model a node sees and
the usage a closed stream already received."""

from __future__ import annotations

from collections.abc import AsyncIterable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import (
    Agent,
    ModelSettings,
    RunContext,
    TokenUsage,
    VoiceSession,
    inference,
    llm,
)
from livekit.agents.metrics import LLMMetrics
from livekit.agents.types import FlushSentinel
from tests.fakes import FakeLLM, Hang, until


@pytest.mark.parametrize("behind_fallback", [False, True])
async def test_prewarm_reaches_the_provider_even_behind_a_fallback_adapter(
    behind_fallback: bool,
) -> None:
    """`prewarm()` reaches the Python LLM itself, also through a `FallbackAdapter`.
    https://github.com/livekit/agents/issues/6572"""
    model = FakeLLM()
    provider = llm.FallbackAdapter([model]) if behind_fallback else model

    provider.prewarm()
    await model.prewarmed.wait()
    await provider.aclose()


async def test_the_inference_gateway_is_prewarmed_as_the_session_starts() -> None:
    """Starting a session with the gateway LLM opens its connection with `GET /models` before
    any reply is asked for."""
    seen: list[str] = []

    async def models(request: web.Request) -> web.Response:
        seen.append(f"{request.method} {request.path}")
        return web.json_response({"object": "list", "data": []})

    app = web.Application()
    app.router.add_get("/models", models)
    async with TestServer(app) as gateway:
        model = inference.LLM(
            "openai/gpt-4.1-mini", base_url=str(gateway.make_url("")), api_key="k", api_secret="s"
        )
        async with VoiceSession(llm=model, vad=None) as session:
            await session.start(Agent(instructions="Answer."))
            await until(lambda: seen, "the gateway saw no request")

    assert seen == ["GET /models"]


async def test_a_failed_prewarm_leaves_the_provider_serving() -> None:
    """A prewarm that raises does not stop the provider from answering the next reply."""
    model = FakeLLM(["ready"], prewarm_error=RuntimeError("prewarm unavailable"))
    model.prewarm()
    await model.prewarmed.wait()

    result = await Agent(instructions="Help.", llm=model).run("hello")

    assert result.status == "completed" and result.output == "ready"


async def test_a_custom_llm_node_sees_the_sessions_configured_provider() -> None:
    """`session.llm` inside a session `llm_node` is the provider the session was built with.
    https://github.com/livekit/agents/issues/3742"""
    model = FakeLLM(["ready"])
    provider = llm.FallbackAdapter([model])
    seen: list[llm.LLM | llm.RealtimeModel | None] = []

    async with VoiceSession(llm=provider) as session:

        @session.llm_node
        def node(
            ctx: RunContext,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings: ModelSettings,
        ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
            seen.append(session.llm)
            return session.default.llm_node(ctx, chat_ctx, tools, model_settings)

        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="hello")

    assert seen == [provider]
    await provider.aclose()


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/2659 reproduces: early close drops usage already received",
)
async def test_closing_a_partial_llm_stream_exports_the_usage_already_received() -> None:
    """Closing a stream early still reports the usage it already received.
    https://github.com/livekit/agents/issues/2659"""
    partial = llm.ChatChunk(
        id="partial",
        delta=llm.ChoiceDelta(role="assistant", content="hello"),
        usage=TokenUsage(input_tokens=7, output_tokens=2),
    )
    model = FakeLLM([[partial, Hang()]])
    metrics: list[LLMMetrics] = []
    model.on("metrics_collected", metrics.append)

    async with model.chat(chat_ctx=llm.ChatContext.empty()) as stream:
        await anext(stream)

    assert [(m.tokens.input_tokens, m.tokens.output_tokens) for m in metrics] == [(7, 2)]
