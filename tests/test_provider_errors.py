"""Provider error reports retain the original exception for the final caller."""

from __future__ import annotations

from typing import Any, Never

import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    llm,
    stt,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

pytestmark = pytest.mark.unit


class _FailingSTT(stt.STT[Never]):
    def __init__(self, failure: APIError, *, transcript: bool = True) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self.failure = failure
        self.attempts = 0
        self.transcript = transcript

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        self.attempts += 1
        raise self.failure

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return _FailingSpeech(stt=self, conn_options=conn_options)


class _FailingSpeech(stt.RecognizeStream):
    async def _run(self) -> None:
        async for _ in self.input:
            pass
        assert isinstance(self._stt, _FailingSTT)
        if self._stt.transcript:
            self.send(stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT))
        self._stt.attempts += 1
        raise self._stt.failure


async def test_fallback_error_report_preserves_the_original_raised_exception() -> None:
    original = APIError("the provider's own failure", retryable=False)
    provider = _FailingSTT(original)
    reports: list[Exception] = []
    provider.on("error", lambda event: reports.append(event.error))
    fallback = stt.FallbackAdapter([provider], max_retry_per_stt=0)
    try:
        with pytest.raises(APIError) as raised:
            async with fallback.stream(conn_options=APIConnectOptions(max_retry=0)) as source:
                source.end_input()
                async for _ in source:
                    pass
        assert raised.value is original
        assert len(reports) == 1 and reports[0] is original
    finally:
        await fallback.aclose()
        await provider.aclose()


class _FailingLLM(llm.LLM[Never]):
    def __init__(self, failure: APIError, *, partial: bool = False) -> None:
        super().__init__()
        self.failure = failure
        self.attempts = 0
        self.partial = partial

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        return _FailingCompletion(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _FailingCompletion(llm.LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _FailingLLM)
        self._llm.attempts += 1
        if self._llm.partial:
            self.send(llm.ChatChunk(id="partial", delta=llm.ChoiceDelta(content="partial")))
        raise self._llm.failure


@pytest.mark.parametrize("kind", ["llm", "stt_batch", "stt_stream"])
async def test_host_retry_exhaustion_reports_attempts_and_retains_the_last_cause(kind: str) -> None:
    original = APIError("the last provider failure", retryable=True)
    conn = APIConnectOptions(max_retry=1, retry_interval=0)
    provider = _FailingLLM(original) if kind == "llm" else _FailingSTT(original, transcript=False)
    activity = "generate LLM completion" if kind == "llm" else "recognize speech"
    reports: list[tuple[Exception, bool]] = []
    provider.on("error", lambda event: reports.append((event.error, event.recoverable)))
    try:
        with pytest.raises(
            APIConnectionError, match=f"failed to {activity} after 2 attempts"
        ) as error:
            if isinstance(provider, _FailingLLM):
                async with provider.chat(
                    chat_ctx=llm.ChatContext.empty(), conn_options=conn
                ) as stream:
                    await stream.collect()
            elif kind == "stt_batch":
                await provider.recognize([], conn_options=conn)
            else:
                async with provider.stream(conn_options=conn) as stream:
                    stream.end_input()
                    async for _ in stream:
                        pass
        assert provider.attempts == 2
        assert error.value.__cause__ is original
        assert [recoverable for _, recoverable in reports] == [True, False]
        assert all(reported is original for reported, _ in reports), reports
    finally:
        await provider.aclose()


async def test_host_retry_exhaustion_preserves_the_original_timeout_phase() -> None:
    original = APITimeoutError("response stalled", phase="read")
    provider = _FailingLLM(original)
    try:
        with pytest.raises(APITimeoutError) as error:
            async with provider.chat(
                chat_ctx=llm.ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            ) as stream:
                await stream.collect()
        assert provider.attempts == 2
        assert error.value is original
        assert error.value.phase == "read"
    finally:
        await provider.aclose()


async def test_partial_host_output_keeps_its_original_failure_without_exhaustion() -> None:
    original = APIError("failed after output", retryable=True)
    provider = _FailingLLM(original, partial=True)
    try:
        with pytest.raises(APIError) as error:
            async with provider.chat(
                chat_ctx=llm.ChatContext.empty(), conn_options=APIConnectOptions(max_retry=2)
            ) as stream:
                await stream.collect()
        assert provider.attempts == 1
        assert error.value is original
    finally:
        await provider.aclose()


def test_generated_exception_classes_keep_public_identity_and_pickle_state() -> None:
    import pickle

    from livekit.agents import APIStatusError
    from livekit.agents._exceptions import AssignmentTimeoutError, StreamStateError
    from livekit.agents.storage import RunRejected, StorageError, ThreadNotFound

    errors = [
        APIError("failure", body={"detail": 1}, retryable=False),
        APIStatusError("limited", status_code=429, request_id="request", body={"detail": 2}),
        APIConnectionError(),
        APITimeoutError(phase="read", retryable=False),
        AssignmentTimeoutError("assignment"),
        StreamStateError("stream"),
        StorageError("storage"),
        ThreadNotFound("thread"),
        RunRejected("occupied", running_task_id="task"),
    ]
    for error in errors:
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error)
        assert restored.args == error.args
        assert restored.__dict__ == error.__dict__
        assert str(restored) == str(error)
        assert repr(restored) == repr(error)
        expected_module = (
            "livekit.agents.storage"
            if isinstance(error, StorageError)
            else "livekit.agents._exceptions"
        )
        assert type(error).__module__ == expected_module
    assert isinstance(errors[-1], RuntimeError)
    assert isinstance(errors[3], APIConnectionError)


@pytest.mark.parametrize("status", [400, 401, 408, 429, 499, 500])
@pytest.mark.parametrize("requested", [None, True, False])
def test_generated_status_constructor_keeps_retry_policy(
    status: int, requested: bool | None
) -> None:
    from livekit.agents import APIStatusError

    error = APIStatusError("failure", status_code=status, retryable=requested)
    expected = True if requested is None else requested
    if 400 <= status < 500 and status not in (408, 429, 499):
        expected = False
    assert error.retryable is expected
