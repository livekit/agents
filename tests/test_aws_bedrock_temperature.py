"""Hermetic tests for Bedrock temperature-rejection handling (#6581).

Bedrock has discontinued the ``temperature`` inference parameter for some newer
models and fails the whole request with a ValidationException when it is sent.
The plugin should strip the parameter and retry instead of failing every
request identically.
"""

import pytest
from botocore.exceptions import ClientError

from livekit.agents import APIConnectOptions
from livekit.agents.llm import ChatContext
from livekit.plugins.aws import llm as aws_llm

pytestmark = pytest.mark.unit

CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)


def _make_llm(**kwargs) -> aws_llm.LLM:
    return aws_llm.LLM(
        model="anthropic.claude-test-v1:0",
        api_key="test-key",
        api_secret="test-secret",
        **kwargs,
    )


def _validation_error(message: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": message}},
        operation_name="ConverseStream",
    )


def _make_chat_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    return chat_ctx


@pytest.fixture
def no_run(monkeypatch: pytest.MonkeyPatch):
    """Prevent LLMStream._run from performing any I/O."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(aws_llm.LLMStream, "_run", _noop)


class TestStripRejectedTemperature:
    async def test_strips_and_flags_on_temperature_validation_error(self, no_run) -> None:
        llm_instance = _make_llm(temperature=0.5)
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            assert stream._opts["inferenceConfig"]["temperature"] == 0.5

            err = _validation_error(
                "This model doesn't support the temperature inference parameter."
            )
            assert stream._maybe_strip_rejected_temperature(err) is True
            assert "temperature" not in stream._opts["inferenceConfig"]
            assert llm_instance._temperature_rejected is True
        finally:
            await stream.aclose()

    async def test_ignores_unrelated_validation_error(self, no_run) -> None:
        llm_instance = _make_llm(temperature=0.5)
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            err = _validation_error("The provided model identifier is invalid.")
            assert stream._maybe_strip_rejected_temperature(err) is False
            assert stream._opts["inferenceConfig"]["temperature"] == 0.5
            assert llm_instance._temperature_rejected is False
        finally:
            await stream.aclose()

    async def test_ignores_when_temperature_not_sent(self, no_run) -> None:
        llm_instance = _make_llm()
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            err = _validation_error(
                "This model doesn't support the temperature inference parameter."
            )
            assert stream._maybe_strip_rejected_temperature(err) is False
        finally:
            await stream.aclose()

    async def test_ignores_non_client_error(self, no_run) -> None:
        llm_instance = _make_llm(temperature=0.5)
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            assert stream._maybe_strip_rejected_temperature(RuntimeError("temperature")) is False
            assert stream._opts["inferenceConfig"]["temperature"] == 0.5
        finally:
            await stream.aclose()


class TestSubsequentRequestsOmitTemperature:
    async def test_chat_omits_temperature_after_rejection(self, no_run) -> None:
        llm_instance = _make_llm(temperature=0.5)
        llm_instance._temperature_rejected = True
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            assert "temperature" not in stream._opts["inferenceConfig"]
        finally:
            await stream.aclose()

    async def test_chat_includes_temperature_by_default(self, no_run) -> None:
        llm_instance = _make_llm(temperature=0.5)
        stream = llm_instance.chat(chat_ctx=_make_chat_ctx(), conn_options=CONN_OPTIONS)
        try:
            assert stream._opts["inferenceConfig"]["temperature"] == 0.5
        finally:
            await stream.aclose()
