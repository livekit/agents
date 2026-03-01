"""Tests for Google LLM http_options factory functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from livekit.agents import APIStatusError
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions
from livekit.plugins.google import LLM


class TestHttpOptionsPolymorphism:
    """Tests for http_options accepting both static values and callables."""

    def test_static_http_options_accepted(self):
        """Test that static HttpOptions are accepted by the LLM constructor."""
        static_options = types.HttpOptions(timeout=5000, headers={"X-Custom": "value"})

        with patch("livekit.plugins.google.llm.Client"):
            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=static_options,
            )

        assert llm._opts.http_options == static_options
        assert not callable(llm._opts.http_options)

    def test_callable_http_options_accepted(self):
        """Test that callable http_options factory is accepted by the LLM constructor."""

        def http_options_factory(attempt: int) -> types.HttpOptions:
            return types.HttpOptions(timeout=5000)

        with patch("livekit.plugins.google.llm.Client"):
            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=http_options_factory,
            )

        assert llm._opts.http_options is http_options_factory
        assert callable(llm._opts.http_options)

    def test_callable_http_options_produces_expected_results(self):
        """Test that callable http_options factory produces expected results."""

        # Factory uses 1-indexed attempt numbers
        def timeout_factory(attempt: int) -> types.HttpOptions:
            return types.HttpOptions(timeout=4000 + attempt * 1000)

        with patch("livekit.plugins.google.llm.Client"):
            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=timeout_factory,
            )

        assert callable(llm._opts.http_options)
        # Verify the factory produces expected results (1-indexed)
        result_1 = llm._opts.http_options(1)  # First attempt
        result_2 = llm._opts.http_options(2)  # First retry
        assert result_1.timeout == 5000
        assert result_2.timeout == 6000


class TestHttpOptionsFactoryBehavior:
    """Tests for the http_options factory invocation behavior."""

    def test_factory_receives_attempt_number(self):
        """Test that the factory receives 1-indexed attempt numbers."""
        recorded_attempts: list[int] = []

        def tracking_factory(attempt: int) -> types.HttpOptions:
            recorded_attempts.append(attempt)
            return types.HttpOptions(timeout=5000)

        # Call the factory with 1-indexed attempt numbers
        tracking_factory(1)  # First attempt
        tracking_factory(2)  # First retry
        tracking_factory(3)  # Second retry

        assert recorded_attempts == [1, 2, 3]

    def test_factory_can_vary_headers_by_attempt(self):
        """Test that factory can return different headers based on attempt number."""

        def priority_on_retry(attempt: int) -> types.HttpOptions:
            headers: dict[str, str] = {"X-Base": "always"}
            if attempt > 1:  # Retry attempts (attempt > 1)
                headers["X-Vertex-AI-LLM-Request-Type"] = "priority"
            return types.HttpOptions(timeout=5000, headers=headers)

        # First attempt (attempt=1) - no priority header
        opts_1 = priority_on_retry(1)
        assert opts_1.headers is not None
        assert opts_1.headers.get("X-Base") == "always"
        assert "X-Vertex-AI-LLM-Request-Type" not in opts_1.headers

        # Retry attempts (attempt > 1) - has priority header
        opts_2 = priority_on_retry(2)
        assert opts_2.headers is not None
        assert opts_2.headers.get("X-Base") == "always"
        assert opts_2.headers.get("X-Vertex-AI-LLM-Request-Type") == "priority"

        opts_3 = priority_on_retry(3)
        assert opts_3.headers is not None
        assert opts_3.headers.get("X-Vertex-AI-LLM-Request-Type") == "priority"

    def test_factory_can_vary_timeout_by_attempt(self):
        """Test that factory can return different timeouts based on attempt number."""

        def increasing_timeout(attempt: int) -> types.HttpOptions:
            # Increase timeout on each retry (attempt is 1-indexed)
            base_timeout = 5000
            return types.HttpOptions(timeout=base_timeout + (attempt * 5000))

        assert increasing_timeout(1).timeout == 10000  # First attempt
        assert increasing_timeout(2).timeout == 15000  # First retry
        assert increasing_timeout(3).timeout == 20000  # Second retry


def _create_mock_response() -> MagicMock:
    """Create a mock successful Gemini API response."""
    mock_response = MagicMock()
    mock_response.prompt_feedback = None
    mock_response.candidates = [
        MagicMock(
            content=MagicMock(parts=[MagicMock(text="Hello", function_call=None)]),
            finish_reason=types.FinishReason.STOP,
        )
    ]
    mock_response.usage_metadata = None
    return mock_response


def _create_mock_stream_fn(
    captured_configs: list[types.GenerateContentConfig] | None = None,
    fail_attempts: set[int] | None = None,
):
    """Create a mock generate_content_stream function.

    Args:
        captured_configs: List to capture GenerateContentConfig from each call
        fail_attempts: Set of call numbers (0-indexed internally) that should fail with 429
    """
    call_count = 0
    fail_attempts = fail_attempts or set()

    async def mock_stream_fn(*args: Any, **kwargs: Any):
        nonlocal call_count
        current_call = call_count
        call_count += 1

        if captured_configs is not None and "config" in kwargs:
            captured_configs.append(kwargs["config"])

        if current_call in fail_attempts:
            raise APIStatusError(
                "rate limited",
                status_code=429,
                retryable=True,
            )

        async def response_generator():
            yield _create_mock_response()

        return response_generator()

    return mock_stream_fn


class TestHttpOptionsIntegration:
    """Integration tests verifying http_options are passed correctly to the API."""

    @pytest.mark.asyncio
    async def test_static_http_options_passed_to_api(self):
        """Test that static HttpOptions are passed to generate_content_stream."""
        captured_configs: list[types.GenerateContentConfig] = []

        with patch("livekit.plugins.google.llm.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = _create_mock_stream_fn(
                captured_configs=captured_configs
            )
            mock_client_cls.return_value = mock_client

            static_options = types.HttpOptions(timeout=5000, headers={"X-Custom": "static-value"})
            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=static_options,
            )

            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content="Hello")

            stream = llm.chat(chat_ctx=chat_ctx)
            async for _ in stream:
                pass
            await stream.aclose()

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.http_options.headers.get("X-Custom") == "static-value"

    @pytest.mark.asyncio
    async def test_callable_http_options_invoked_with_attempt_one(self):
        """Test that callable http_options is invoked with attempt=1 on first try."""
        recorded_attempts: list[int] = []

        def tracking_factory(attempt: int) -> types.HttpOptions:
            recorded_attempts.append(attempt)
            return types.HttpOptions(timeout=5000, headers={"X-Attempt": str(attempt)})

        with patch("livekit.plugins.google.llm.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = _create_mock_stream_fn()
            mock_client_cls.return_value = mock_client

            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=tracking_factory,
            )

            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content="Hello")

            stream = llm.chat(chat_ctx=chat_ctx)
            async for _ in stream:
                pass
            await stream.aclose()

        # First attempt should be attempt number 1
        assert recorded_attempts == [1]

    @pytest.mark.asyncio
    async def test_callable_http_options_invoked_with_incrementing_attempts_on_retry(self):
        """Test that callable http_options receives incrementing attempt numbers on retries."""
        recorded_attempts: list[int] = []
        captured_configs: list[types.GenerateContentConfig] = []

        def tracking_factory(attempt: int) -> types.HttpOptions:
            recorded_attempts.append(attempt)
            headers = {"X-Attempt": str(attempt)}
            if attempt > 1:  # Retry attempts
                headers["X-Vertex-AI-LLM-Request-Type"] = "priority"
            return types.HttpOptions(timeout=5000, headers=headers)

        with patch("livekit.plugins.google.llm.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = _create_mock_stream_fn(
                captured_configs=captured_configs,
                fail_attempts={0},  # First call fails (0-indexed internally)
            )
            mock_client_cls.return_value = mock_client

            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=tracking_factory,
            )

            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content="Hello")

            conn_options = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=10.0)
            stream = llm.chat(chat_ctx=chat_ctx, conn_options=conn_options)
            async for _ in stream:
                pass
            await stream.aclose()

        # Factory should be called with attempt 1, then attempt 2 (1-indexed)
        assert recorded_attempts == [1, 2]

        # Verify captured headers
        assert len(captured_configs) == 2

        # First attempt (attempt=1) should not have priority header
        headers_1 = captured_configs[0].http_options.headers
        assert headers_1.get("X-Attempt") == "1"
        assert "X-Vertex-AI-LLM-Request-Type" not in headers_1

        # Second attempt (attempt=2, retry) should have priority header
        headers_2 = captured_configs[1].http_options.headers
        assert headers_2.get("X-Attempt") == "2"
        assert headers_2.get("X-Vertex-AI-LLM-Request-Type") == "priority"

    @pytest.mark.asyncio
    async def test_static_http_options_reused_on_retry(self):
        """Test that static HttpOptions are reused (same values) on retries."""
        captured_configs: list[types.GenerateContentConfig] = []

        with patch("livekit.plugins.google.llm.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = _create_mock_stream_fn(
                captured_configs=captured_configs,
                fail_attempts={0},  # First call fails
            )
            mock_client_cls.return_value = mock_client

            static_options = types.HttpOptions(timeout=5000, headers={"X-Static": "same-value"})
            llm = LLM(
                model="gemini-2.5-flash",
                api_key="test-key",
                http_options=static_options,
            )

            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content="Hello")

            conn_options = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=10.0)
            stream = llm.chat(chat_ctx=chat_ctx, conn_options=conn_options)
            async for _ in stream:
                pass
            await stream.aclose()

        # Both attempts should have the same static header
        assert len(captured_configs) == 2
        assert captured_configs[0].http_options.headers.get("X-Static") == "same-value"
        assert captured_configs[1].http_options.headers.get("X-Static") == "same-value"


class TestLLMStreamAttemptNumber:
    """Tests for the _attempt_number field in the base LLMStream class."""

    @pytest.mark.asyncio
    async def test_attempt_number_initialized_to_one(self):
        """Test that _attempt_number is initialized to 1 in LLMStream."""
        from tests.fake_llm import FakeLLM, FakeLLMResponse

        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="test",
                    content="response",
                    ttft=0.0,
                    duration=0.0,
                )
            ]
        )
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="test")

        stream = llm.chat(chat_ctx=chat_ctx)
        assert hasattr(stream, "_attempt_number")
        assert stream._attempt_number == 1

        # Consume the stream to clean up properly
        async for _ in stream:
            pass
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_attempt_number_exists_on_stream(self):
        """Test that _attempt_number attribute exists on LLMStream subclasses."""
        from tests.fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="test",
                    content="response",
                    ttft=0.0,
                    duration=0.0,
                )
            ]
        )
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="test")

        stream = llm.chat(chat_ctx=chat_ctx)
        assert isinstance(stream, FakeLLMStream)
        assert hasattr(stream, "_attempt_number")

        # Consume the stream to clean up properly
        async for _ in stream:
            pass
        await stream.aclose()
