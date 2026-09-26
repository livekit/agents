from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents import Agent, AgentSession, APIConnectionError, APIStatusError, utils
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FallbackAdapter,
    GenerationCreatedEvent,
    LLMStream,
    RealtimeModelError,
)
from livekit.agents.metrics import (
    ProviderRequestAttempt,
    ProviderRequestLedger,
    RealtimeModelMetrics,
)
from livekit.agents.stt import FallbackAdapter as STTFallbackAdapter
from livekit.agents.tts import FallbackAdapter as TTSFallbackAdapter
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.report import SessionReport

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel
from .fake_session import FakeActions, create_session, run_session
from .fake_stt import FakeRecognizeStream, FakeSTT
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit]


def _attempt(index: int) -> ProviderRequestAttempt:
    return ProviderRequestAttempt(
        component="llm",
        provider="test",
        model="test-model",
        operation_id="operation",
        sdk_request_id=f"sdk-{index}",
        started_at=float(index),
        completed_at=float(index + 1),
        outcome="success",
        retry_index=0,
        fallback_index=0,
    )


def test_ledger_is_bounded_and_snapshots_are_immutable() -> None:
    ledger = ProviderRequestLedger(capacity=3)
    for index in range(5):
        ledger.record(_attempt(index))

    snapshot = ledger.snapshot()
    assert isinstance(snapshot, tuple)
    assert [attempt.sdk_request_id for attempt in snapshot] == ["sdk-2", "sdk-3", "sdk-4"]
    assert len(ledger) == 3

    with pytest.raises(ValueError, match="capacity must be greater than 0"):
        ProviderRequestLedger(capacity=0)


def test_attempt_contains_only_safe_operational_metadata() -> None:
    fields = set(_attempt(0).model_dump())
    assert fields == {
        "component",
        "provider",
        "model",
        "operation_id",
        "sdk_request_id",
        "provider_request_ids",
        "provider_trace_ids",
        "speech_id",
        "purpose",
        "started_at",
        "completed_at",
        "outcome",
        "retry_index",
        "fallback_index",
        "error_type",
        "status_code",
        "retryable",
    }
    assert not fields & {"prompt", "transcript", "audio", "request", "response", "body", "api_key"}


def test_session_report_serializes_attempts_only_when_present() -> None:
    session = AgentSession(vad=None)
    base = {
        "job_id": "job",
        "room_id": "room-id",
        "room": "room",
        "options": session.options,
        "events": [],
        "chat_history": ChatContext.empty(),
    }

    assert "provider_request_attempts" not in SessionReport(**base).to_dict()
    report = SessionReport(**base, provider_request_attempts=(_attempt(0),)).to_dict()
    assert report["provider_request_attempts"] == [_attempt(0).model_dump()]


def test_session_report_preserves_existing_positional_field_order() -> None:
    session = AgentSession(vad=None)
    report = SessionReport(
        "job",
        "room-id",
        "room",
        session.options,
        [],
        ChatContext.empty(),
        None,
        None,
        2.0,
        1.0,
        3.0,
        None,
        "test-version",
    )

    assert report.duration == 2.0
    assert report.started_at == 1.0
    assert report.timestamp == 3.0
    assert report.sdk_version == "test-version"
    assert report.provider_request_attempts == ()


class _AttemptLLM(FakeLLM):
    def __init__(self, *, fail: bool, provider: str) -> None:
        super().__init__()
        self.fail = fail
        self.provider_name = provider

    @property
    def provider(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return f"{self.provider_name}-model"

    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        return _AttemptStream(
            self,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs.get("tools") or [],
            conn_options=kwargs.get("conn_options", APIConnectOptions()),
        )


class _AttemptStream(LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _AttemptLLM)
        if self._llm.fail:
            raise APIStatusError(
                "secret provider response body",
                status_code=503,
                request_id="provider-error-id",
                body={"secret": "must not be retained"},
            )
        self._note_provider_request_id("provider-success-id")
        self._note_provider_trace_id("provider-trace-id")
        self._event_ch.send_nowait(
            ChatChunk(id="plugin-local-chunk-id", delta=ChoiceDelta(content="hello"))
        )


class _RetryAttemptLLM(_AttemptLLM):
    def __init__(self) -> None:
        super().__init__(fail=False, provider="retrying")
        self.attempts = 0

    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        return _RetryAttemptStream(
            self,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs.get("tools") or [],
            conn_options=kwargs.get("conn_options", APIConnectOptions()),
        )


class _RetryAttemptStream(_AttemptStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _RetryAttemptLLM)
        self._llm.attempts += 1
        if self._llm.attempts == 1:
            raise APIConnectionError("transient")
        await super()._run()


async def test_retry_attempts_have_stable_operation_and_distinct_sdk_ids() -> None:
    model = _RetryAttemptLLM()
    attempts: list[ProviderRequestAttempt] = []
    model.on("provider_request_completed", attempts.append)

    async with model.chat(
        chat_ctx=ChatContext.empty(),
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
    ) as stream:
        assert [chunk async for chunk in stream]

    assert [(attempt.outcome, attempt.retry_index) for attempt in attempts] == [
        ("error", 0),
        ("success", 1),
    ]
    assert len({attempt.operation_id for attempt in attempts}) == 1
    assert len({attempt.sdk_request_id for attempt in attempts}) == 2


async def test_fallback_records_failed_and_successful_attempts_with_lineage() -> None:
    primary = _AttemptLLM(fail=True, provider="primary")
    backup = _AttemptLLM(fail=False, provider="backup")
    adapter = FallbackAdapter([primary, backup], max_retry_per_llm=0)
    attempts: list[ProviderRequestAttempt] = []
    adapter.on("provider_request_completed", attempts.append)

    try:
        async with adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            assert [chunk async for chunk in stream]

        assert [(a.provider, a.outcome, a.fallback_index) for a in attempts[:2]] == [
            ("primary", "error", 0),
            ("backup", "success", 1),
        ]
        assert len({a.operation_id for a in attempts[:2]}) == 1
        assert len({a.sdk_request_id for a in attempts[:2]}) == 2
        assert attempts[0].provider_request_ids == ("provider-error-id",)
        assert attempts[0].error_type == "APIStatusError"
        assert attempts[0].status_code == 503
        assert attempts[1].provider_request_ids == ("provider-success-id",)
        assert attempts[1].provider_trace_ids == ("provider-trace-id",)
        assert "plugin-local-chunk-id" not in attempts[1].provider_request_ids
        assert "secret" not in attempts[0].model_dump_json()
        assert all(attempt.provider == "primary" for attempt in attempts[2:])
    finally:
        await adapter.aclose()


async def test_fallback_retry_lineage_is_exact_and_recovery_is_distinct() -> None:
    primary = _AttemptLLM(fail=True, provider="primary")
    backup = _AttemptLLM(fail=False, provider="backup")
    adapter = FallbackAdapter(
        [primary, backup],
        max_retry_per_llm=1,
        retry_interval=0,
    )
    attempts: list[ProviderRequestAttempt] = []
    adapter.on("provider_request_completed", attempts.append)

    try:
        async with adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            assert [chunk async for chunk in stream]

        recovery_task = adapter._status[0].recovering_task
        assert recovery_task is not None
        await recovery_task

        foreground = [attempt for attempt in attempts if attempt.purpose == "foreground"]
        assert [attempt.retry_index for attempt in foreground] == [0, 1, 0]
        assert [attempt.fallback_index for attempt in foreground] == [0, 0, 1]
        assert [attempt.outcome for attempt in foreground] == ["error", "error", "success"]
        assert len({attempt.operation_id for attempt in foreground}) == 1
        assert len({attempt.sdk_request_id for attempt in foreground}) == 3

        recovery = [attempt for attempt in attempts if attempt.purpose == "recovery"]
        assert recovery
        assert all(attempt.operation_id != foreground[0].operation_id for attempt in recovery)
    finally:
        await adapter.aclose()


async def test_cancelled_tts_attempt_is_retained() -> None:
    tts = FakeTTS(fake_timeout=30.0)
    attempts: list[ProviderRequestAttempt] = []
    tts.on("provider_request_completed", attempts.append)
    stream = tts.synthesize("private text")

    await asyncio.sleep(0)
    await stream.aclose()

    assert len(attempts) == 1
    assert attempts[0].component == "tts"
    assert attempts[0].outcome == "cancelled"
    assert "private text" not in attempts[0].model_dump_json()


async def test_tts_fallback_records_primary_error_and_backup_success() -> None:
    primary = FakeTTS(fake_exception=APIConnectionError("primary failed"), fake_exception_count=1)
    backup = FakeTTS(fake_audio_duration=0.1)
    adapter = TTSFallbackAdapter([primary, backup], max_retry_per_tts=0)
    attempts: list[ProviderRequestAttempt] = []
    adapter.on("provider_request_completed", attempts.append)

    try:
        async with adapter.synthesize("private text") as stream:
            assert [frame async for frame in stream]

        primary_error = next(
            attempt
            for attempt in attempts
            if attempt.outcome == "error" and attempt.fallback_index == 0
        )
        backup_success = next(
            attempt
            for attempt in attempts
            if attempt.outcome == "success" and attempt.fallback_index == 1
        )
        assert primary_error.operation_id == backup_success.operation_id
    finally:
        await adapter.aclose()


def _realtime_activity() -> tuple[AgentActivity, ProviderRequestLedger]:
    ledger = ProviderRequestLedger(capacity=4)
    session = AgentSession(vad=None, provider_request_ledger=ledger)
    activity = object.__new__(AgentActivity)
    activity._session = session
    activity._realtime_spans = None
    activity._realtime_request_trackers = utils.BoundedDict(maxsize=100)
    activity._realtime_pending_errors = utils.BoundedDict(maxsize=100)
    activity._realtime_completed_request_ids = utils.BoundedDict(maxsize=100)
    activity._agent = Agent(instructions="test", llm=FakeRealtimeModel())
    return activity, ledger


def _realtime_metrics(**kwargs: object) -> RealtimeModelMetrics:
    return RealtimeModelMetrics(
        request_id=str(kwargs.pop("request_id", "")),
        timestamp=float(kwargs.pop("timestamp", 10.0)),
        input_token_details=RealtimeModelMetrics.InputTokenDetails(),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
        **kwargs,
    )


def test_connection_acquisition_metrics_do_not_create_realtime_attempt() -> None:
    activity, ledger = _realtime_activity()

    AgentActivity._on_metrics_collected(
        activity,
        _realtime_metrics(acquire_time=0.25),
    )

    assert ledger.snapshot() == ()


async def _empty_stream():
    if False:
        yield None


@pytest.mark.parametrize("cancelled", [False, True])
def test_realtime_metrics_preserve_local_id_and_generation_lineage(cancelled: bool) -> None:
    activity, ledger = _realtime_activity()
    generation = GenerationCreatedEvent(
        message_stream=_empty_stream(),
        function_stream=_empty_stream(),
        user_initiated=True,
        response_id="framework-response-id",
    )
    AgentActivity._on_generation_created(activity, generation)
    operation_id = activity._realtime_request_trackers["framework-response-id"].operation_id

    AgentActivity._on_metrics_collected(
        activity,
        _realtime_metrics(
            request_id="framework-response-id",
            duration=2.0,
            cancelled=cancelled,
        ),
    )

    attempt = ledger.snapshot()[0]
    assert attempt.component == "realtime"
    assert attempt.outcome == ("cancelled" if cancelled else "success")
    assert attempt.operation_id == operation_id
    assert attempt.sdk_request_id == "framework-response-id"
    assert attempt.provider_request_ids == ()


@pytest.mark.parametrize("cancelled", [False, True])
def test_recoverable_realtime_error_then_metrics_has_one_terminal_attempt(
    cancelled: bool,
) -> None:
    activity, ledger = _realtime_activity()
    generation = GenerationCreatedEvent(
        message_stream=_empty_stream(),
        function_stream=_empty_stream(),
        user_initiated=True,
        response_id="framework-response-id",
    )
    AgentActivity._on_generation_created(activity, generation)
    operation_id = activity._realtime_request_trackers["framework-response-id"].operation_id

    AgentActivity._on_error(
        activity,
        RealtimeModelError(
            timestamp=time.time(),
            label="fake",
            error=APIConnectionError("private provider error"),
            recoverable=True,
            request_id="framework-response-id",
        ),
    )

    assert ledger.snapshot() == ()
    assert "framework-response-id" in activity._realtime_pending_errors

    AgentActivity._record_realtime_metrics_attempt(
        activity,
        _realtime_metrics(
            request_id="framework-response-id",
            duration=2.0,
            cancelled=cancelled,
        ),
    )

    attempt = ledger.snapshot()[0]
    assert len(ledger.snapshot()) == 1
    assert attempt.operation_id == operation_id
    assert attempt.sdk_request_id == "framework-response-id"
    assert attempt.outcome == ("cancelled" if cancelled else "success")
    assert attempt.provider_request_ids == ()
    assert "private provider error" not in attempt.model_dump_json()
    assert "framework-response-id" not in activity._realtime_request_trackers
    assert "framework-response-id" not in activity._realtime_pending_errors
    assert "framework-response-id" in activity._realtime_completed_request_ids


def test_unrecoverable_realtime_error_is_terminal_before_late_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity, ledger = _realtime_activity()
    generation = GenerationCreatedEvent(
        message_stream=_empty_stream(),
        function_stream=_empty_stream(),
        user_initiated=True,
        response_id="framework-response-id",
    )
    AgentActivity._on_generation_created(activity, generation)
    operation_id = activity._realtime_request_trackers["framework-response-id"].operation_id

    monkeypatch.setattr(activity._session, "_on_error", lambda _: None)
    AgentActivity._on_error(
        activity,
        RealtimeModelError(
            timestamp=time.time(),
            label="fake",
            error=APIConnectionError("private provider error"),
            recoverable=False,
            request_id="framework-response-id",
        ),
    )
    AgentActivity._record_realtime_metrics_attempt(
        activity,
        _realtime_metrics(request_id="framework-response-id", duration=2.0),
    )

    assert len(ledger.snapshot()) == 1
    attempt = ledger.snapshot()[0]
    assert attempt.operation_id == operation_id
    assert attempt.sdk_request_id == "framework-response-id"
    assert attempt.outcome == "error"
    assert attempt.provider_request_ids == ()
    assert "framework-response-id" not in activity._realtime_request_trackers
    assert "framework-response-id" not in activity._realtime_pending_errors


def test_realtime_error_without_request_id_is_recorded_without_guessing() -> None:
    activity, ledger = _realtime_activity()

    AgentActivity._on_error(
        activity,
        RealtimeModelError(
            timestamp=time.time(),
            label="fake",
            error=APIConnectionError("private provider error"),
            recoverable=True,
        ),
    )

    attempt = ledger.snapshot()[0]
    assert attempt.outcome == "error"
    assert attempt.sdk_request_id.startswith("req_")
    assert attempt.provider_request_ids == ()
    assert activity._realtime_pending_errors == {}


def test_pending_realtime_error_is_flushed_if_metrics_never_arrive() -> None:
    activity, ledger = _realtime_activity()
    AgentActivity._record_realtime_error_attempt(
        activity,
        RealtimeModelError(
            timestamp=time.time(),
            label="fake",
            error=APIConnectionError("private provider error"),
            recoverable=True,
            request_id="unknown-response-id",
        ),
    )

    AgentActivity._flush_realtime_pending_errors(activity)

    attempt = ledger.snapshot()[0]
    assert attempt.outcome == "error"
    assert attempt.sdk_request_id == "unknown-response-id"
    assert attempt.provider_request_ids == ()
    assert activity._realtime_pending_errors == {}


def test_realtime_correlation_state_is_bounded() -> None:
    activity, ledger = _realtime_activity()

    for index in range(101):
        AgentActivity._record_realtime_error_attempt(
            activity,
            RealtimeModelError(
                timestamp=time.time(),
                label="fake",
                error=APIConnectionError("private provider error"),
                recoverable=True,
                request_id=f"response-{index}",
            ),
        )

    assert len(activity._realtime_pending_errors) == 100
    assert "response-0" not in activity._realtime_pending_errors
    assert ledger.snapshot()[0].sdk_request_id == "response-0"
    assert ledger.snapshot()[0].outcome == "error"
    assert len(activity._realtime_completed_request_ids) == 1


class _ProviderIdRaceStream(FakeRecognizeStream):
    async def _run(self) -> None:
        self._event_ch.send_nowait(self._make_provider_event())

    @staticmethod
    def _make_provider_event():  # type: ignore[no-untyped-def]
        from livekit.agents.stt import SpeechEvent, SpeechEventType

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            provider_request_ids=("provider-stt-id",),
            provider_trace_ids=("provider-stt-trace",),
        )


class _ProviderIdRaceSTT(FakeSTT):
    def stream(self, **kwargs):  # type: ignore[no-untyped-def]
        return _ProviderIdRaceStream(
            stt=self,
            conn_options=kwargs.get("conn_options", APIConnectOptions()),
        )


class _NamedLedgerSTT(FakeSTT):
    def __init__(self, *, provider: str, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._provider_name = provider

    @property
    def provider(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return f"{self._provider_name}-model"


async def test_streaming_stt_captures_provider_ids_before_terminal_attempt() -> None:
    stt = _ProviderIdRaceSTT()
    attempts: list[ProviderRequestAttempt] = []
    stt.on("provider_request_completed", attempts.append)
    stream = stt.stream()

    await stream._task

    assert len(attempts) == 1
    assert attempts[0].provider_request_ids == ("provider-stt-id",)
    assert attempts[0].provider_trace_ids == ("provider-stt-trace",)
    await stream.aclose()


async def test_non_streaming_stt_fallback_inherits_operation_lineage() -> None:
    primary = _NamedLedgerSTT(
        provider="primary",
        fake_exception=APIConnectionError("primary failed"),
    )
    backup = _NamedLedgerSTT(provider="backup", fake_transcript="hello")
    adapter = STTFallbackAdapter(
        [primary, backup],
        max_retry_per_stt=1,
        retry_interval=0,
    )
    attempts: list[ProviderRequestAttempt] = []
    adapter.on("provider_request_completed", attempts.append)

    try:
        await adapter.recognize([])
        foreground = [attempt for attempt in attempts if attempt.purpose == "foreground"]
        assert [attempt.retry_index for attempt in foreground] == [0, 1, 0]
        assert [attempt.fallback_index for attempt in foreground] == [0, 0, 1]
        assert [attempt.outcome for attempt in foreground] == ["error", "error", "success"]
        assert len({attempt.operation_id for attempt in foreground}) == 1
        assert len({attempt.sdk_request_id for attempt in foreground}) == 3
    finally:
        await adapter.aclose()


async def test_streaming_stt_fallback_preserves_equivalent_lineage() -> None:
    primary = _NamedLedgerSTT(
        provider="primary",
        fake_exception=APIConnectionError("primary failed"),
    )
    backup = _NamedLedgerSTT(provider="backup", fake_transcript="hello")
    adapter = STTFallbackAdapter(
        [primary, backup],
        max_retry_per_stt=1,
        retry_interval=0,
    )
    attempts: list[ProviderRequestAttempt] = []
    adapter.on("provider_request_completed", attempts.append)

    try:
        async with adapter.stream() as stream:
            stream.end_input()
            assert [event async for event in stream]

        foreground = [attempt for attempt in attempts if attempt.purpose == "foreground"]
        assert [attempt.retry_index for attempt in foreground] == [0, 1, 0]
        assert [attempt.fallback_index for attempt in foreground] == [0, 0, 1]
        assert [attempt.outcome for attempt in foreground] == ["error", "error", "success"]
        assert len({attempt.operation_id for attempt in foreground}) == 1
        assert len({attempt.sdk_request_id for attempt in foreground}) == 3
    finally:
        await adapter.aclose()


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_agent_session_collects_pipeline_attempts() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.1, 0.2, "private transcript", stt_delay=0.01)
    actions.add_llm("private answer", ttft=0.01, duration=0.02)
    actions.add_tts(0.1, ttfb=0.01, duration=0.02)
    ledger = ProviderRequestLedger(capacity=16)
    session = create_session(
        actions,
        speed_factor=10,
        extra_kwargs={"provider_request_ledger": ledger},
    )

    await asyncio.wait_for(run_session(session, Agent(instructions="test"), drain_delay=0.2), 5)

    attempts = ledger.snapshot()
    assert {attempt.component for attempt in attempts} >= {"stt", "llm", "tts"}
    assert all(
        attempt.speech_id is not None
        for attempt in attempts
        if attempt.component in ("llm", "tts") and attempt.outcome == "success"
    )
    serialized = "".join(attempt.model_dump_json() for attempt in attempts)
    assert "private transcript" not in serialized
    assert "private answer" not in serialized
    assert session.provider_request_ledger is ledger
