from __future__ import annotations

import anthropic
import httpx2 as httpx
import openai
import pytest

from livekit.agents import APIConnectOptions, APIError, APIStatusError, llm
from livekit.plugins.anthropic import LLM as AnthropicLLM
from livekit.plugins.google.utils import genai_http_options
from livekit.plugins.mistralai._http import request_timeout_ms
from livekit.plugins.openai import LLM as OpenAILLM

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.asyncio
async def test_injected_sdk_client_sends_one_attempt_and_keeps_its_policy(provider: str) -> None:
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"error": {"type": "overloaded_error", "message": "busy"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client: anthropic.AsyncClient | openai.AsyncClient
        model: llm.LLM
        if provider == "anthropic":
            client = anthropic.AsyncClient(api_key="test", max_retries=4, http_client=transport)
            model = AnthropicLLM(client=client)
        else:
            client = openai.AsyncClient(api_key="test", max_retries=4, http_client=transport)
            model = OpenAILLM(client=client)
        try:
            stream = model.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(
                    max_retry=0,
                    pool_timeout=0.1,
                    connect_timeout=0.2,
                    write_timeout=0.3,
                    read_timeout=0.4,
                ),
            )
            with pytest.raises(APIStatusError):
                await stream.collect()
            assert len(requests) == 1
            assert requests[0].extensions["timeout"] == {
                "pool": 0.1,
                "connect": 0.2,
                "write": 0.3,
                "read": 0.4,
            }
            assert client.max_retries == 4
        finally:
            await model.aclose()
            await client.close()


def test_mistral_rejects_an_inexpressible_phase_policy() -> None:
    with pytest.raises(APIError, match="must be equal") as error:
        request_timeout_ms(APIConnectOptions(connect_timeout=0.1, read_timeout=5))
    assert not error.value.retryable


def test_genai_takes_one_attempt_without_mutating_application_options() -> None:
    from google.genai import types

    original = types.HttpOptions(timeout=123, retry_options=types.HttpRetryOptions(attempts=9))
    options = genai_http_options(APIConnectOptions(), original)
    assert options.retry_options is not None and options.retry_options.attempts == 1
    assert original.timeout == 123
    assert original.retry_options is not None and original.retry_options.attempts == 9


@pytest.mark.asyncio
async def test_fal_failure_does_not_trigger_hidden_sdk_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx as fal_httpx

    from livekit import rtc
    from livekit.plugins.fal import WizperSTT

    requests: list[fal_httpx.Request] = []

    async def respond(request: fal_httpx.Request) -> fal_httpx.Response:
        requests.append(request)
        return fal_httpx.Response(503, text="busy")

    transport = fal_httpx.MockTransport(respond)
    factory = fal_httpx.AsyncClient
    monkeypatch.setattr(
        fal_httpx, "AsyncClient", lambda **kwargs: factory(transport=transport, **kwargs)
    )
    model = WizperSTT(api_key="test")
    try:
        frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
        with pytest.raises(APIStatusError) as error:
            await model.recognize(frame, conn_options=APIConnectOptions(max_retry=0))
        assert error.value.status_code == 503
        assert len(requests) == 1
    finally:
        await model.aclose()
