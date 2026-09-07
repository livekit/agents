from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import ssl
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from google.rpc import error_details_pb2, status_pb2

from livekit.agents import Agent, AgentSession
from livekit.agents.telemetry.traces import _upload_session_report
from livekit.agents.voice.agent_session import (
    _RECORDING_ALL_OFF,
    _RECORDING_ALL_ON,
    RecordingOptions,
)
from livekit.protocol import metrics as proto_metrics

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_AGENT_SESSION_MOD = "livekit.agents.voice.agent_session"
_TRACES_MOD = "livekit.agents.telemetry.traces"


class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a test agent.")


def _create_simple_session() -> AgentSession:
    """Create a minimal AgentSession without TranscriptSynchronizer to avoid leaked tasks."""
    session = AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[], min_silence_duration=0.5, min_speech_duration=0.05),
        stt=FakeSTT(fake_user_speeches=[]),
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session


async def _cleanup(session: AgentSession) -> None:
    """Drain and close a session, suppressing errors from missing context."""
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()


def _make_mock_job_ctx(enable_recording: bool = True) -> MagicMock:
    """Build a mock JobContext with the fields that agent_session.start() accesses."""
    mock_ctx = MagicMock()
    mock_ctx.job.enable_recording = enable_recording
    mock_ctx.job.id = "test-job-id"
    mock_ctx.job.room.sid = "test-room-sid"
    mock_ctx.job.agent_name = "test-agent"
    mock_ctx.room.name = "test-room"
    mock_ctx._primary_agent_session = None
    mock_ctx.session_directory = Path("/tmp/test-session")
    return mock_ctx


@contextlib.contextmanager
def _patch_job_ctx(mock_ctx: MagicMock, *, patch_recorder: bool = False) -> Iterator[None]:
    """Patch get_job_context and optionally RecorderIO."""
    with patch(f"{_AGENT_SESSION_MOD}.get_job_context", return_value=mock_ctx):
        if patch_recorder:
            with patch(f"{_AGENT_SESSION_MOD}.RecorderIO") as mock_cls:
                recorder = MagicMock()
                recorder.record_input.side_effect = lambda x: x
                recorder.record_output.side_effect = lambda x: x
                recorder.start = AsyncMock()
                recorder.aclose = AsyncMock()
                mock_cls.return_value = recorder
                yield
        else:
            yield


def _make_mock_report(recording_options: RecordingOptions | None = None) -> MagicMock:
    """Create a minimal mock SessionReport for upload tests."""
    report = MagicMock()
    report.job_id = "job-1"
    report.room_id = "room-1"
    report.room = "test-room"
    report.events = []
    report.chat_history.items = []
    report.chat_history.to_dict.return_value = {"items": []}
    report.audio_recording_path = None
    report.audio_recording_started_at = None
    report.duration = 10.0
    report.started_at = 1000.0
    report.timestamp = 1010.0
    report.options = MagicMock()
    report.options.recording_options = recording_options or _RECORDING_ALL_ON.copy()
    return report


def _make_mock_tagger(
    *, evaluations: list | None = None, outcome_reason: str | None = None
) -> MagicMock:
    mock = MagicMock()
    mock.evaluations = evaluations or []
    mock.outcome_reason = outcome_reason
    mock.tags = set()
    mock._tags = {}
    mock.outcome = "pass" if outcome_reason else None
    return mock


def _make_mock_http() -> MagicMock:
    """Create a mock aiohttp.ClientSession with async post."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_post_cm = AsyncMock()
    mock_post_cm.__aenter__.return_value = mock_resp
    mock_http.post.return_value = mock_post_cm
    return mock_http


def _observability_endpoint_arg(func: Any) -> dict[str, str]:
    """Build endpoint kwargs for old/new telemetry function signatures."""
    if "observability_url" in inspect.signature(func).parameters:
        return {"observability_url": "https://test.livekit.cloud"}
    return {"cloud_hostname": "test.livekit.cloud"}


def _stub_access_token(mock_at: MagicMock) -> None:
    """Make a patched api.AccessToken produce a fixed JWT."""
    mock_token = MagicMock()
    mock_token.with_observability_grants.return_value = mock_token
    mock_token.with_ttl.return_value = mock_token
    mock_token.to_jwt.return_value = "test-jwt"
    mock_at.return_value = mock_token


@contextlib.contextmanager
def _patch_upload_deps() -> Iterator[MagicMock]:
    """Patch OTel logger provider and AccessToken. Yields the mock logger for assertions."""
    mock_logger = MagicMock()
    with (
        patch(f"{_TRACES_MOD}.get_logger_provider") as mock_glp,
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
    ):
        provider = mock_glp.return_value
        provider.get_logger.return_value = mock_logger
        mock_logger.provider = provider
        _stub_access_token(mock_at)
        yield mock_logger


@contextlib.contextmanager
def _fresh_telemetry_state() -> Iterator[None]:
    """Swap in a fresh _CloudTelemetry and restore process-wide telemetry state.

    _setup_cloud_tracer / _shutdown_telemetry mutate process-lifetime state (the
    singleton, the _DynamicTracer provider slot, the root logger); tests must not
    leak it into each other.
    """
    from livekit.agents.telemetry import traces as traces_mod

    prev_cloud = traces_mod._cloud
    prev_provider = traces_mod.tracer._tracer_provider
    traces_mod._cloud = traces_mod._CloudTelemetry()
    try:
        yield
    finally:
        if (handler := traces_mod._cloud.log_handler) is not None:
            logging.getLogger().removeHandler(handler)
        traces_mod._cloud = prev_cloud
        traces_mod.tracer.set_provider(prev_provider)


@contextlib.contextmanager
def _stub_cloud_tracer_deps() -> Iterator[SimpleNamespace]:
    """Fresh telemetry state with auth, the OTLP exporters, the batch processors,
    and the set-once OTel logger/meter globals stubbed out, so tests never touch
    the network or the real process globals."""
    from opentelemetry.metrics._internal import _ProxyMeterProvider

    # stateful stand-ins for the set-once OTel globals; tests can pre-seed these
    # with an "integrator" provider
    meter_state: dict[str, Any] = {"provider": _ProxyMeterProvider()}
    logger_state: dict[str, Any] = {"provider": MagicMock()}  # not an SDK LoggerProvider

    with (
        _fresh_telemetry_state(),
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
        patch(f"{_TRACES_MOD}.OTLPSpanExporter") as mock_span_exporter,
        patch(f"{_TRACES_MOD}.OTLPLogExporter") as mock_log_exporter,
        patch(f"{_TRACES_MOD}.OTLPMetricExporter") as mock_metric_exporter,
        patch(f"{_TRACES_MOD}.BatchSpanProcessor") as mock_span_batch,
        patch(f"{_TRACES_MOD}.BatchLogRecordProcessor") as mock_log_batch,
        patch(f"{_TRACES_MOD}.PeriodicExportingMetricReader"),
        patch(f"{_TRACES_MOD}.SdkMeterProvider") as mock_meter_provider_cls,
        patch(
            f"{_TRACES_MOD}.get_logger_provider",
            side_effect=lambda: logger_state["provider"],
        ),
        patch(
            f"{_TRACES_MOD}.set_logger_provider",
            side_effect=lambda p: logger_state.__setitem__("provider", p),
        ) as mock_set_logger_provider,
        patch(
            f"{_TRACES_MOD}.metrics_api.get_meter_provider",
            side_effect=lambda: meter_state["provider"],
        ),
        patch(
            f"{_TRACES_MOD}.metrics_api.set_meter_provider",
            side_effect=lambda p: meter_state.__setitem__("provider", p),
        ) as mock_set_meter_provider,
        patch(f"{_TRACES_MOD}.atexit"),
    ):
        _stub_access_token(mock_at)
        yield SimpleNamespace(
            span_exporter=mock_span_exporter,
            log_exporter=mock_log_exporter,
            metric_exporter=mock_metric_exporter,
            span_batch=mock_span_batch,
            log_batch=mock_log_batch,
            meter_provider_cls=mock_meter_provider_cls,
            set_logger_provider=mock_set_logger_provider,
            set_meter_provider=mock_set_meter_provider,
            meter_state=meter_state,
            logger_state=logger_state,
        )


def _setup_cloud_tracer_for_job(job_id: str = "job-1", **kwargs: Any) -> None:
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    params: dict[str, Any] = {
        "room_id": "room-1",
        "job_id": job_id,
        **_observability_endpoint_arg(_setup_cloud_tracer),
        "enable_traces": True,
        "enable_logs": True,
    }
    params.update(kwargs)
    _setup_cloud_tracer(**params)


async def _call_upload(
    report: MagicMock,
    *,
    tagger: MagicMock | None = None,
    http_session: MagicMock | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Call _upload_session_report with sensible defaults."""
    await _upload_session_report(
        agent_name="test-agent",
        **_observability_endpoint_arg(_upload_session_report),
        report=report,
        tagger=tagger or _make_mock_tagger(),
        http_session=http_session or _make_mock_http(),
        metadata=metadata,
    )


def _get_multipart_part_names(mp_writer: aiohttp.MultipartWriter) -> list[str]:
    """Extract part names from a MultipartWriter."""
    names = []
    for payload, _enc, _te in mp_writer._parts:
        cd = payload.headers.get("Content-Disposition", "")
        for name in ("header", "chat_history", "audio"):
            if f'name="{name}"' in cd:
                names.append(name)
    return names


def _get_multipart_parts(mp_writer: aiohttp.MultipartWriter) -> dict[str, Any]:
    parts = {}
    for payload, _enc, _te in mp_writer._parts:
        cd = payload.headers.get("Content-Disposition", "")
        for name in ("header", "chat_history", "audio"):
            if f'name="{name}"' in cd:
                parts[name] = payload
    return parts


def _retry_info_body(delay_seconds: int = 0) -> bytes:
    retry_info = error_details_pb2.RetryInfo()
    retry_info.retry_delay.seconds = delay_seconds
    status = status_pb2.Status()
    status.details.add().Pack(retry_info)
    return status.SerializeToString()


# ---------------------------------------------------------------------------
# Group 1: RecordingOptions normalization (no JobContext)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record, expected",
    [
        pytest.param(True, _RECORDING_ALL_ON, id="record=True"),
        pytest.param(False, _RECORDING_ALL_OFF, id="record=False"),
        pytest.param(
            {"audio": False},
            {
                "audio": False,
                "traces": True,
                "logs": True,
                "transcript": True,
                "redaction": False,
            },
            id="partial",
        ),
        pytest.param(
            {"redaction": True},
            {
                "audio": True,
                "traces": True,
                "logs": True,
                "transcript": True,
                "redaction": True,
            },
            id="redaction",
        ),
    ],
)
async def test_record_normalization(
    record: bool | RecordingOptions, expected: RecordingOptions
) -> None:
    session = _create_simple_session()
    await session.start(SimpleAgent(), record=record)
    assert session.options.recording_options == expected
    await _cleanup(session)


async def test_record_not_given_without_job_ctx() -> None:
    """When record is omitted and no JobContext is available, all options should be False."""
    session = _create_simple_session()
    await session.start(SimpleAgent())
    assert session.options.recording_options == _RECORDING_ALL_OFF
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 2: init_recording() interaction with mock JobContext
# ---------------------------------------------------------------------------


async def test_init_recording_called_with_options() -> None:
    """init_recording should be called with the correct RecordingOptions."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()
    custom: RecordingOptions = {
        "audio": True,
        "traces": True,
        "logs": False,
        "transcript": True,
        "redaction": True,
    }

    with _patch_job_ctx(mock_ctx, patch_recorder=True):
        await session.start(SimpleAgent(), record=custom)

    # _resolve_recording_options merges with defaults, so the result should match
    mock_ctx.init_recording.assert_called_once()
    assert mock_ctx.init_recording.call_args[0][0] == {
        "audio": True,
        "traces": True,
        "logs": False,
        "transcript": True,
        "redaction": True,
    }
    await _cleanup(session)


async def test_init_recording_called_even_when_all_false() -> None:
    """init_recording is always called when job context exists (evals need OTel infrastructure)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with _patch_job_ctx(mock_ctx):
        await session.start(SimpleAgent(), record=False)

    mock_ctx.init_recording.assert_called_once()
    assert mock_ctx.init_recording.call_args[0][0] == _RECORDING_ALL_OFF
    await _cleanup(session)


async def test_init_recording_defers_to_job_enable_recording() -> None:
    """When record= is omitted, the value should come from job.enable_recording."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=True)

    with _patch_job_ctx(mock_ctx, patch_recorder=True):
        await session.start(SimpleAgent())

    mock_ctx.init_recording.assert_called_once()
    assert mock_ctx.init_recording.call_args[0][0] == _RECORDING_ALL_ON
    await _cleanup(session)


async def test_init_recording_called_when_job_recording_disabled() -> None:
    """init_recording should be called even when job.enable_recording=False (evals need it)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=False)

    with _patch_job_ctx(mock_ctx):
        await session.start(SimpleAgent())

    mock_ctx.init_recording.assert_called_once()
    assert session.options.recording_options == _RECORDING_ALL_OFF
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 3: _upload_session_report() conditional upload
# ---------------------------------------------------------------------------


async def test_upload_returns_early_when_none() -> None:
    """When all options are False, no HTTP request and no session report log should be made."""
    report = _make_mock_report(
        {"audio": False, "traces": False, "logs": False, "transcript": False, "redaction": True}
    )
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post = MagicMock()

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, http_session=mock_http)

    mock_http.post.assert_not_called()
    for call in mock_logger.emit.call_args_list:
        assert call.kwargs.get("body") != "session report"


async def test_upload_transcript_only() -> None:
    """When transcript=True and audio=False, upload should include header + chat_history."""
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http)

    mock_http.post.assert_called_once()
    mp_writer = mock_http.post.call_args.kwargs.get("data") or mock_http.post.call_args[1]["data"]
    part_names = _get_multipart_part_names(mp_writer)
    assert "header" in part_names
    assert "chat_history" in part_names
    assert "audio" not in part_names


async def test_upload_uses_extended_timeout() -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http)

    timeout = mock_http.post.call_args.kwargs["timeout"]
    assert timeout.total == 900
    assert timeout.sock_connect == 30


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(aiohttp.ConnectionTimeoutError("connect timed out"), id="connect-timeout"),
        pytest.param(
            aiohttp.ClientConnectorError(MagicMock(), OSError("connection failed")),
            id="connector-error",
        ),
    ],
)
async def test_upload_retries_connection_failure(error: Exception) -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    failure_cm = AsyncMock()
    failure_cm.__aenter__.side_effect = error

    success_resp = AsyncMock()
    success_resp.status = 200
    success_cm = AsyncMock()
    success_cm.__aenter__.return_value = success_resp

    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post.side_effect = [failure_cm, success_cm]

    with (
        _patch_upload_deps(),
        patch(f"{_TRACES_MOD}._recording_upload_retry_delay", return_value=0.0),
    ):
        await _call_upload(report, http_session=mock_http)

    assert mock_http.post.call_count == 2


async def test_upload_retries_response_with_retry_info() -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    retry_resp = AsyncMock()
    retry_resp.status = 503
    retry_resp.read.return_value = _retry_info_body()
    retry_cm = AsyncMock()
    retry_cm.__aenter__.return_value = retry_resp

    success_resp = AsyncMock()
    success_resp.status = 200
    success_cm = AsyncMock()
    success_cm.__aenter__.return_value = success_resp

    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post.side_effect = [retry_cm, success_cm]

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http)

    assert mock_http.post.call_count == 2


async def test_upload_does_not_retry_response_without_retry_info() -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    response_error = aiohttp.ClientResponseError(
        MagicMock(), (), status=503, message="service unavailable"
    )
    response = AsyncMock()
    response.status = 503
    response.read.return_value = b""
    response.raise_for_status = MagicMock(side_effect=response_error)
    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = response
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post.return_value = response_cm

    with _patch_upload_deps(), pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await _call_upload(report, http_session=mock_http)

    assert exc_info.value.status == 503
    assert mock_http.post.call_count == 1


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(asyncio.TimeoutError("upload timed out"), id="total-timeout"),
        pytest.param(aiohttp.ServerDisconnectedError("response lost"), id="disconnected"),
        pytest.param(
            aiohttp.ClientConnectorSSLError(MagicMock(), ssl.SSLError("TLS failed")),
            id="tls-error",
        ),
    ],
)
async def test_upload_does_not_retry_ambiguous_or_tls_failure(error: Exception) -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    failure_cm = AsyncMock()
    failure_cm.__aenter__.side_effect = error
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post.return_value = failure_cm

    with _patch_upload_deps(), pytest.raises(type(error)):
        await _call_upload(report, http_session=mock_http)

    assert mock_http.post.call_count == 1


async def test_upload_stops_after_connection_retries_are_exhausted() -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    timeout_cm = AsyncMock()
    timeout_cm.__aenter__.side_effect = aiohttp.ConnectionTimeoutError("connect timed out")
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post.return_value = timeout_cm

    with (
        _patch_upload_deps(),
        patch(f"{_TRACES_MOD}._recording_upload_retry_delay", return_value=0.0),
        pytest.raises(aiohttp.ConnectionTimeoutError, match="connect timed out"),
    ):
        await _call_upload(report, http_session=mock_http)

    assert mock_http.post.call_count == 4


async def test_upload_session_report_sent_without_transcript() -> None:
    """Session report log should be emitted even when transcript=False, if other options are on."""
    report = _make_mock_report({"audio": True, "traces": True, "logs": False, "transcript": False})
    mock_http = _make_mock_http()

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, http_session=mock_http)

    bodies = [c.kwargs.get("body") for c in mock_logger.emit.call_args_list]
    assert "session report" in bodies
    assert "chat item" not in bodies


async def test_upload_session_report_marks_stt_keyterms_as_pii() -> None:
    report = _make_mock_report({"audio": False, "traces": True, "logs": False, "transcript": False})
    stt_context_options = {
        "keyterms": ["Acme Corp"],
        "keyterm_detection": {"enabled": False},
        "forward_chat_context": True,
    }
    report.options.stt_context_options = stt_context_options

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report)

    session_report_call = next(
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "session report"
    )
    serialized_stt_options = session_report_call.kwargs["attributes"]["session.options"][
        "stt_context_options"
    ]
    assert serialized_stt_options["lk.pii.keyterms"] == ["Acme Corp"]
    assert "keyterms" not in serialized_stt_options
    assert stt_context_options["keyterms"] == ["Acme Corp"]


def test_session_report_constructor_includes_recording_options_in_options() -> None:
    from livekit.agents.voice.report import SessionReport

    recording_options: RecordingOptions = {
        "audio": False,
        "traces": True,
        "logs": False,
        "transcript": False,
        "redaction": True,
    }
    session = _create_simple_session()
    session.options.recording_options = recording_options
    report = SessionReport(
        job_id="job-1",
        room_id="room-1",
        room="test-room",
        options=session.options,
        events=[],
        chat_history=session.history,
    )

    assert report.options.recording_options == recording_options
    serialized_recording_options = report.to_dict()["options"]["recording_options"]
    assert serialized_recording_options == recording_options

    serialized_recording_options["audio"] = True
    assert report.options.recording_options["audio"] is False


async def test_upload_audio_only_no_file() -> None:
    """When transcript=False, audio=True but no audio file exists, no upload is made."""
    report = _make_mock_report({"audio": True, "traces": False, "logs": False, "transcript": False})
    report.audio_recording_path = None
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http)

    mock_http.post.assert_not_called()


async def test_upload_evaluations_emitted_without_logs() -> None:
    """Evaluations should be emitted even when logs=False, as long as something is recorded."""
    report = _make_mock_report({"audio": True, "traces": False, "logs": False, "transcript": False})
    tagger = _make_mock_tagger(
        evaluations=[{"name": "test-eval", "verdict": "pass"}],
        outcome_reason="all good",
    )

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, tagger=tagger)

    bodies = [c.kwargs.get("body") for c in mock_logger.emit.call_args_list]
    assert bodies.count("evaluation") == 1
    assert bodies.count("outcome") == 1


async def test_upload_session_report_includes_simulation_metadata() -> None:
    report = _make_mock_report({"audio": False, "traces": True, "logs": False, "transcript": False})
    metadata = {
        "lk.simulation.enabled": True,
    }

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, metadata=metadata)

    attrs = mock_logger.provider.get_logger.call_args_list[0].kwargs["attributes"]
    assert attrs["lk.simulation.enabled"] is True
    session_report_call = next(
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "session report"
    )
    assert "session.simulation" not in session_report_call.kwargs["attributes"]


async def test_upload_session_report_includes_redaction_metadata() -> None:
    report = _make_mock_report({"audio": False, "traces": True, "logs": False, "transcript": False})

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, metadata={"lk.redaction.enabled": True})

    attrs = mock_logger.provider.get_logger.call_args_list[0].kwargs["attributes"]
    assert attrs["lk.redaction.enabled"] is True


async def test_upload_multipart_header_carries_simulation_redaction() -> None:
    report = _make_mock_report({"audio": False, "traces": False, "logs": False, "transcript": True})
    metadata = {
        "lk.simulation.enabled": True,
        "lk.redaction.enabled": True,
    }
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http, metadata=metadata)

    mp_writer = mock_http.post.call_args.kwargs.get("data") or mock_http.post.call_args[1]["data"]
    parts = _get_multipart_parts(mp_writer)
    header = proto_metrics.MetricsRecordingHeader.FromString(parts["header"]._value)
    assert header.simulated is True
    assert header.redaction_enabled is True


def test_job_context_otel_metadata_includes_redaction_option() -> None:
    from livekit.agents.job import JobContext

    ctx = object.__new__(JobContext)
    ctx.simulation_context = MagicMock(return_value=None)

    assert ctx._otel_metadata({"redaction": True}) == {"lk.redaction.enabled": True}


def _make_simulation_context(*, run_id: str, job_id: str) -> object:
    from livekit.agents.simulation import SimulationContext
    from livekit.protocol import agent_simulation as sim_pb

    dispatch = sim_pb.SimulationDispatch(simulation_run_id=run_id, job_id=job_id)
    return SimulationContext(dispatch, MagicMock())


def test_job_context_otel_metadata_includes_simulation_identity() -> None:
    from livekit.agents.job import JobContext

    ctx = object.__new__(JobContext)
    sim = _make_simulation_context(run_id="run_abc", job_id="job_def")
    ctx.simulation_context = MagicMock(return_value=sim)

    assert ctx._otel_metadata() == {
        "lk.simulation.enabled": True,
        "lk.simulation.run_id": "run_abc",
        "lk.simulation.job_id": "job_def",
    }


def test_job_context_otel_metadata_omits_blank_simulation_ids() -> None:
    """A dispatch with no ids must not stamp empty ones: downstream reads an absent
    attribute as "not a simulation" and an empty one as a run named ""."""
    from livekit.agents.job import JobContext

    ctx = object.__new__(JobContext)
    sim = _make_simulation_context(run_id="", job_id="")
    ctx.simulation_context = MagicMock(return_value=sim)

    assert ctx._otel_metadata() == {"lk.simulation.enabled": True}


def test_job_context_init_recording_enables_session_redaction() -> None:
    from livekit.agents.job import JobContext

    ctx = object.__new__(JobContext)
    ctx._info = SimpleNamespace(
        job=SimpleNamespace(enable_redaction=False),
        url="",
    )
    ctx._recording_initialized = False
    ctx._redaction_enabled = False
    ctx._early_log_handler = None

    ctx.init_recording(
        {
            "audio": False,
            "traces": False,
            "logs": False,
            "transcript": False,
            "redaction": True,
        }
    )

    assert ctx._redaction_enabled is True


@pytest.mark.parametrize(
    ("project_redaction", "session_redaction"),
    [
        pytest.param(True, False, id="project-redaction"),
        pytest.param(False, True, id="session-redaction"),
    ],
)
def test_job_context_init_recording_rejects_audio_without_transcript_when_redacted(
    project_redaction: bool, session_redaction: bool
) -> None:
    from livekit.agents.job import JobContext

    ctx = object.__new__(JobContext)
    ctx._info = SimpleNamespace(
        job=SimpleNamespace(enable_redaction=project_redaction),
        url="",
    )
    ctx._recording_initialized = False
    ctx._redaction_enabled = project_redaction
    ctx._early_log_handler = None

    with pytest.raises(
        ValueError, match="audio upload requires transcript upload when redaction is enabled"
    ):
        ctx.init_recording(
            {
                "audio": True,
                "traces": False,
                "logs": False,
                "transcript": False,
                "redaction": session_redaction,
            }
        )


async def test_upload_session_report_omits_simulation_metadata_for_normal_session() -> None:
    report = _make_mock_report({"audio": False, "traces": True, "logs": False, "transcript": False})

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report)

    attrs = mock_logger.provider.get_logger.call_args_list[0].kwargs["attributes"]
    assert not any(k.startswith("lk.simulation.") for k in attrs)
    session_report_call = next(
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "session report"
    )
    assert "session.simulation" not in session_report_call.kwargs["attributes"]


def test_setup_cloud_tracer_logger_provider_always_created() -> None:
    """LoggerProvider should be set up even when enable_logs=False."""
    with (
        _stub_cloud_tracer_deps() as stubs,
        patch(f"{_TRACES_MOD}.Resource.create") as mock_resource_create,
    ):
        _setup_cloud_tracer_for_job(
            enable_traces=False,
            enable_logs=False,
            metadata={"lk.simulation.enabled": True},
        )

    stubs.set_logger_provider.assert_called_once()
    service_resource_calls = [
        call.args[0]
        for call in mock_resource_create.call_args_list
        if "service.name" in call.args[0]
    ]
    assert service_resource_calls
    assert not any(k.startswith("lk.simulation.") for k in service_resource_calls[0])
    # OTLP exporter should NOT be created when enable_logs=False
    stubs.log_exporter.assert_not_called()
    stubs.log_batch.assert_not_called()


def _resource_attrs_for_env(env: dict[str, str]) -> dict[str, Any]:
    """Run _setup_cloud_tracer under the given os.environ and return the dict
    passed to Resource.create."""
    with (
        patch.dict("os.environ", env, clear=True),
        _stub_cloud_tracer_deps(),
        patch(f"{_TRACES_MOD}.Resource.create") as mock_resource_create,
    ):
        _setup_cloud_tracer_for_job(enable_traces=False, enable_logs=False)
    # Resource.create is also called internally by LoggerProvider() with an
    # empty dict, so select the call that built the tracing resource (the one
    # carrying service.name) rather than relying on call ordering.
    for call in mock_resource_create.call_args_list:
        attrs = call.args[0]
        if "service.name" in attrs:
            return attrs
    raise AssertionError("Resource.create was not called with the service resource")


def test_setup_cloud_tracer_adds_identity_from_env() -> None:
    """LIVEKIT_AGENT_ID / LIVEKIT_AGENT_DEPLOYMENT become
    lk.cloud_agent_id / lk.deployment_id on the tracing resource."""
    attrs = _resource_attrs_for_env(
        {"LIVEKIT_AGENT_ID": "CA_test123", "LIVEKIT_AGENT_DEPLOYMENT": "canary"}
    )
    assert attrs["lk.cloud_agent_id"] == "CA_test123"
    assert attrs["lk.deployment_id"] == "canary"


def test_setup_cloud_tracer_omits_identity_when_env_unset() -> None:
    """Neither identity attr is set when the env vars are absent."""
    attrs = _resource_attrs_for_env({})
    assert "lk.cloud_agent_id" not in attrs
    assert "lk.deployment_id" not in attrs


def test_setup_cloud_tracer_omits_empty_deployment() -> None:
    """An empty LIVEKIT_AGENT_DEPLOYMENT is omitted rather than emitted."""
    attrs = _resource_attrs_for_env(
        {"LIVEKIT_AGENT_ID": "CA_test123", "LIVEKIT_AGENT_DEPLOYMENT": ""}
    )
    assert attrs["lk.cloud_agent_id"] == "CA_test123"
    assert "lk.deployment_id" not in attrs


# ---------------------------------------------------------------------------
# Group 3.5: telemetry ownership & process reuse (_CloudTelemetry)
# ---------------------------------------------------------------------------


def test_shutdown_telemetry_keeps_integrator_providers_alive() -> None:
    """Providers the integrator configured at process start (Langfuse per the
    tracing docs, Logfire, dd-trace) must survive per-job teardown: worker
    processes are reused across jobs, so shutting them down would kill the
    integrator's export for every later job."""
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry, set_tracer_provider

    integrator_tracer = MagicMock(spec=SdkTracerProvider)
    integrator_logger = MagicMock(spec=LoggerProvider)
    integrator_meter = MagicMock(spec=SdkMeterProvider)

    with _stub_cloud_tracer_deps() as stubs:
        set_tracer_provider(integrator_tracer)
        stubs.logger_state["provider"] = integrator_logger
        stubs.meter_state["provider"] = integrator_meter

        _setup_cloud_tracer_for_job()

        # sanity: the framework adopted the integrator's providers and attached
        # its own processors to them
        assert traces_mod.tracer._tracer_provider is integrator_tracer
        integrator_tracer.add_span_processor.assert_called()
        integrator_logger.add_log_record_processor.assert_called()

        _shutdown_telemetry("job-1")

        # per-job teardown flushes the framework's own exporters...
        stubs.span_batch.return_value.force_flush.assert_called_once()
        stubs.log_batch.return_value.force_flush.assert_called_once()

    # ...but never shuts down anything the integrator owns
    integrator_tracer.shutdown.assert_not_called()
    integrator_logger.shutdown.assert_not_called()
    integrator_meter.shutdown.assert_not_called()
    # the framework did not build a competing meter provider either (there is no
    # API to attach a reader to the integrator's)
    stubs.meter_provider_cls.assert_not_called()


def test_framework_created_providers_live_for_the_process() -> None:
    """Providers the framework creates are reused by the next job in the process
    (the OTel logger/meter globals are set-once) and shut down only at process
    exit — never per job: the set-once OTel globals cannot be replaced, so a
    per-job shutdown would leave every later job on dead providers."""
    from opentelemetry.trace import NoOpTracerProvider

    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry, set_tracer_provider

    with _stub_cloud_tracer_deps() as stubs:
        set_tracer_provider(NoOpTracerProvider())

        # job 1
        _setup_cloud_tracer_for_job(job_id="job-1")
        created = traces_mod.tracer._tracer_provider
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

        assert isinstance(created, SdkTracerProvider)

        with patch.object(created, "shutdown") as mock_tracer_shutdown:
            _shutdown_telemetry("job-1")

            # job 2 in the same process: everything is reused, nothing rebuilt
            _setup_cloud_tracer_for_job(job_id="job-2")
            assert traces_mod.tracer._tracer_provider is created
            _shutdown_telemetry("job-2")

            stubs.span_exporter.assert_called_once()
            stubs.log_exporter.assert_called_once()
            stubs.set_logger_provider.assert_called_once()
            stubs.set_meter_provider.assert_called_once()

            # per-job teardown never shut anything down...
            mock_tracer_shutdown.assert_not_called()
            stubs.meter_provider_cls.return_value.shutdown.assert_not_called()
            # ...but flushed the exporters at the end of each job
            assert stubs.span_batch.return_value.force_flush.call_count == 2
            assert stubs.meter_provider_cls.return_value.force_flush.call_count == 2

        # process exit shuts down everything the framework created, exactly once
        with patch(f"{_TRACES_MOD}._run_bounded") as mock_run_bounded:
            traces_mod._cloud.shutdown_at_exit()
        target_fns = [fn for _, fn in mock_run_bounded.call_args.args[1]]
        assert created.shutdown in target_fns
        assert stubs.meter_provider_cls.return_value.shutdown in target_fns
        assert target_fns.count(created.shutdown) == 1


def test_slot_metadata_follows_job_lifecycle() -> None:
    """The fallback stamp for out-of-job-context telemetry is set while a job is
    running and cleared when the last job releases — unstamped records between
    jobs are then dropped by the stamp-gated exporters."""
    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry

    with _stub_cloud_tracer_deps():
        _setup_cloud_tracer_for_job(job_id="job-1")
        cloud = traces_mod._cloud
        assert cloud._span_metadata_processor is not None
        assert cloud._span_metadata_processor._metadata["job_id"] == "job-1"
        assert cloud._log_metadata_processor is not None
        assert cloud._log_metadata_processor._metadata["job_id"] == "job-1"

        _shutdown_telemetry("job-1")
        assert cloud._span_metadata_processor._metadata == {}
        assert cloud._log_metadata_processor._metadata == {}


def test_exporters_only_upload_records_of_registered_jobs() -> None:
    """Cloud export is registry-gated: a record uploads only while its stamped
    job_id is in the exportable-jobs registry, and only for the signals that
    job enabled. Attribution is never stripped from any record — upload policy
    and attributes are independent."""
    from opentelemetry.sdk._logs.export import LogRecordExportResult
    from opentelemetry.sdk.trace.export import SpanExportResult

    from livekit.agents.telemetry.traces import (
        _GatedLogExporter,
        _GatedSpanExporter,
        _JobTelemetry,
    )

    export_jobs = {
        "job-on": _JobTelemetry(attributes={}, traces_enabled=True, logs_enabled=True),
        "job-off": _JobTelemetry(attributes={}, traces_enabled=False, logs_enabled=False),
    }

    recorded = SimpleNamespace(attributes={"room_id": "room-1", "job_id": "job-on"})
    disabled = SimpleNamespace(attributes={"room_id": "room-2", "job_id": "job-off"})
    released = SimpleNamespace(attributes={"room_id": "room-3", "job_id": "job-gone"})
    unstamped = SimpleNamespace(attributes={"some": "attr"})

    inner = MagicMock()
    gate = _GatedSpanExporter(inner, export_jobs)
    assert gate.export([disabled, released, unstamped]) is SpanExportResult.SUCCESS
    inner.export.assert_not_called()
    gate.export([disabled, recorded, released, unstamped])
    inner.export.assert_called_once_with([recorded])

    def _rec(attrs: dict[str, Any] | None) -> SimpleNamespace:
        return SimpleNamespace(log_record=SimpleNamespace(attributes=attrs))

    inner = MagicMock()
    log_gate = _GatedLogExporter(inner, export_jobs)
    assert (
        log_gate.export([_rec({"job_id": "job-off"}), _rec(None)]) is LogRecordExportResult.SUCCESS
    )
    inner.export.assert_not_called()
    recorded_rec = _rec({"room_id": "room-1", "job_id": "job-on"})
    log_gate.export([_rec({"job_id": "job-gone"}), recorded_rec])
    inner.export.assert_called_once_with([recorded_rec])


def test_shutdown_telemetry_is_idempotent_and_safe_without_setup() -> None:
    """_on_cleanup runs for every job, including ones that never initialized
    recording; and a second teardown must not double-flush."""
    from livekit.agents.telemetry.traces import _shutdown_telemetry

    with _stub_cloud_tracer_deps() as stubs:
        _shutdown_telemetry("job-1")  # no configure ran — must be a no-op

        _setup_cloud_tracer_for_job()
        _shutdown_telemetry("job-1")
        _shutdown_telemetry("job-1")
        assert stubs.span_batch.return_value.force_flush.call_count == 1


def test_concurrent_jobs_keep_exporting_until_the_last_release() -> None:
    """THREAD-mode workers run multiple jobs in one process: the first job's
    cleanup must not turn off export or detach the log handler while another
    job is still running."""
    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry

    with _stub_cloud_tracer_deps():
        _setup_cloud_tracer_for_job(job_id="job-a")
        _setup_cloud_tracer_for_job(job_id="job-b")
        cloud = traces_mod._cloud
        root = logging.getLogger()

        _shutdown_telemetry("job-a")  # job A ends; job B is still running
        assert cloud._span_metadata_processor is not None
        assert cloud._span_metadata_processor._metadata  # slot still stamped
        assert cloud.log_handler is not None and cloud.log_handler in root.handlers

        _shutdown_telemetry("job-b")  # job B ends
        assert cloud._span_metadata_processor._metadata == {}
        assert cloud.log_handler not in root.handlers


def test_shutdown_telemetry_leaves_foreign_log_handlers_attached() -> None:
    """Only the framework's own OTLP handler is detached from the root logger."""
    from opentelemetry.sdk._logs import LoggingHandler

    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry

    foreign_handler = LoggingHandler(logger_provider=MagicMock())
    root = logging.getLogger()
    root.addHandler(foreign_handler)
    try:
        with _stub_cloud_tracer_deps():
            _setup_cloud_tracer_for_job()
            ours = traces_mod._cloud.log_handler
            assert ours is not None and ours in root.handlers

            _shutdown_telemetry("job-1")
            assert ours not in root.handlers
        assert foreign_handler in root.handlers
    finally:
        root.removeHandler(foreign_handler)


def test_flush_early_log_buffer_replays_into_framework_handler() -> None:
    """Buffered crash logs replay into the framework's own OTLP handler — not
    into an integrator's LoggingHandler that was installed on root first."""
    from opentelemetry.sdk._logs import LoggingHandler

    from livekit.agents.job import JobContext
    from livekit.agents.telemetry.traces import _BufferingHandler

    ctx = object.__new__(JobContext)
    buffering = _BufferingHandler()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "boom", None, None)
    buffering.buffer.append(record)
    ctx._early_log_handler = buffering

    foreign_handler = MagicMock(spec=LoggingHandler)
    framework_handler = MagicMock()
    root = logging.getLogger()
    root.addHandler(foreign_handler)  # like an integrator handler, installed first
    root.addHandler(buffering)
    root.addHandler(framework_handler)
    try:
        with patch("livekit.agents.job._cloud_log_handler", return_value=framework_handler):
            ctx._flush_early_log_buffer(replay=True)
    finally:
        root.removeHandler(foreign_handler)
        root.removeHandler(framework_handler)

    framework_handler.emit.assert_called_once_with(record)
    foreign_handler.emit.assert_not_called()
    assert buffering not in root.handlers


def test_meter_provider_skipped_when_integrator_set_a_custom_one() -> None:
    """A non-SDK meter provider set by the integrator is left alone, and no
    orphan reader is built (set_meter_provider would silently refuse ours)."""
    with _stub_cloud_tracer_deps() as stubs:
        stubs.meter_state["provider"] = MagicMock()  # custom, not Proxy/NoOp/SDK

        _setup_cloud_tracer_for_job()

    stubs.meter_provider_cls.assert_not_called()
    stubs.set_meter_provider.assert_not_called()


def test_meter_provider_resource_carries_no_job_identity() -> None:
    """The meter provider outlives the job (set-once global), so its resource
    must not carry the first job's room_id/job_id — per-job identity rides on
    each measurement instead (otel_metrics._job_identity_attrs)."""
    with _stub_cloud_tracer_deps() as stubs:
        _setup_cloud_tracer_for_job(agent_name="test-agent")

    resource = stubs.meter_provider_cls.call_args.kwargs["resource"]
    assert "room_id" not in resource.attributes
    assert "job_id" not in resource.attributes
    # process-stable identity is kept
    assert resource.attributes["service.name"] == "livekit-agents"
    assert resource.attributes["lk.agent_name"] == "test-agent"


def test_metric_measurements_carry_job_identity() -> None:
    """Usage metrics are stamped with the job's identity and session metadata —
    the same per-job fields that ride on spans and logs."""
    from livekit.agents.job import _JobContextVar
    from livekit.agents.metrics.base import LLMMetrics
    from livekit.agents.telemetry import otel_metrics
    from livekit.agents.telemetry.traces import _JobTelemetry

    mock_ctx = MagicMock()
    mock_ctx.job.id = "job-42"
    mock_ctx.job.room.sid = "room-42"
    # what init_recording() stashes: identity + session metadata
    mock_ctx._telemetry_state = _JobTelemetry(
        attributes={"room_id": "room-42", "job_id": "job-42", "lk.simulation.enabled": True},
        traces_enabled=True,
        logs_enabled=True,
    )

    ev = MagicMock(spec=LLMMetrics)
    ev.metadata = None
    ev.prompt_tokens = 5
    ev.prompt_cached_tokens = 0
    ev.completion_tokens = 0
    # also read by the GenAI semconv instruments (gen_ai.client.operation.duration,
    # gen_ai.client.operation.time_to_first_chunk)
    ev.duration = 0.0
    ev.ttft = -1.0

    with patch.object(otel_metrics, "_llm_input_tokens") as mock_counter:
        token = _JobContextVar.set(mock_ctx)
        try:
            otel_metrics.collect_usage(ev)
        finally:
            _JobContextVar.reset(token)

    attrs = mock_counter.add.call_args.kwargs["attributes"]
    assert attrs["room_id"] == "room-42"
    assert attrs["job_id"] == "job-42"
    assert attrs["lk.simulation.enabled"] is True

    # recording never initialized (disabled / crash path): identity still stamped
    mock_ctx._telemetry_state = None
    with patch.object(otel_metrics, "_llm_input_tokens") as mock_counter:
        token = _JobContextVar.set(mock_ctx)
        try:
            otel_metrics.collect_usage(ev)
        finally:
            _JobContextVar.reset(token)
    attrs = mock_counter.add.call_args.kwargs["attributes"]
    assert attrs == {"room_id": "room-42", "job_id": "job-42"}

    # outside a job context the attributes are simply absent, never wrong
    with patch.object(otel_metrics, "_llm_input_tokens") as mock_counter:
        otel_metrics.collect_usage(ev)
    attrs = mock_counter.add.call_args.kwargs["attributes"]
    assert "room_id" not in attrs and "job_id" not in attrs


def test_provider_swap_still_shuts_down_current_span_pipeline_at_exit() -> None:
    """If the integrator replaces a framework-created tracer provider
    mid-process, process exit must shut down BOTH the old owned provider and the
    batch processor attached to the integrator's provider, or its queued spans
    are lost — and it must never shut down the integrator's provider itself."""
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.trace import NoOpTracerProvider

    from livekit.agents.telemetry import traces as traces_mod
    from livekit.agents.telemetry.traces import _shutdown_telemetry, set_tracer_provider

    with _stub_cloud_tracer_deps() as stubs:
        batch_instances: list[MagicMock] = []
        stubs.span_batch.side_effect = lambda *a, **k: (
            batch_instances.append(MagicMock()) or batch_instances[-1]
        )

        # job 1: no integrator provider — the framework creates its own
        set_tracer_provider(NoOpTracerProvider())
        _setup_cloud_tracer_for_job(job_id="job-1")
        owned = traces_mod.tracer._tracer_provider
        assert isinstance(owned, SdkTracerProvider) and len(batch_instances) == 1
        _shutdown_telemetry("job-1")

        # the integrator replaces the provider mid-process
        integrator = MagicMock(spec=SdkTracerProvider)
        set_tracer_provider(integrator)

        # job 2: the pipeline re-attaches to the integrator's provider and the
        # old one is retired (shut down on a background thread)
        _setup_cloud_tracer_for_job(job_id="job-2")
        assert len(batch_instances) == 2
        old_batch, new_batch = batch_instances
        integrator.add_span_processor.assert_any_call(new_batch)
        for _ in range(50):  # retire happens on a daemon thread
            if old_batch.shutdown.called:
                break
            time.sleep(0.02)
        old_batch.shutdown.assert_called_once()
        _shutdown_telemetry("job-2")

        # the exit targets cover the owned provider AND the current pipeline on
        # the integrator's provider — never the integrator's provider itself
        with patch(f"{_TRACES_MOD}._run_bounded") as mock_run_bounded:
            traces_mod._cloud.shutdown_at_exit()
        target_fns = [fn for _, fn in mock_run_bounded.call_args.args[1]]
        assert owned.shutdown in target_fns
        assert new_batch.shutdown in target_fns
        assert integrator.shutdown not in target_fns


def _make_telemetry_job_ctx(
    *,
    room_id: str,
    job_id: str,
    initialized: bool = True,
    traces: bool = True,
    logs: bool = True,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """A JobContext stand-in carrying the per-record telemetry state that
    init_recording() stashes."""
    from livekit.agents.telemetry.traces import _JobTelemetry

    ctx = MagicMock()
    ctx.job.id = job_id
    ctx.job.room.sid = room_id
    ctx._telemetry_state = (
        _JobTelemetry(
            attributes={"room_id": room_id, "job_id": job_id, **(metadata or {})},
            traces_enabled=traces,
            logs_enabled=logs,
        )
        if initialized
        else None
    )
    return ctx


@contextlib.contextmanager
def _job_ctx_active(ctx: MagicMock) -> Iterator[None]:
    from livekit.agents.job import _JobContextVar

    token = _JobContextVar.set(ctx)
    try:
        yield
    finally:
        _JobContextVar.reset(token)


def test_unconfigured_job_cleanup_does_not_release_telemetry() -> None:
    """_on_cleanup runs for every job, including ones that never configured
    telemetry (recording disabled, no observability URL). Such a job must not
    release a concurrent recorded job's registration — in THREAD mode that would
    turn export off underneath it."""
    from livekit.agents.job import JobContext
    from livekit.agents.telemetry.traces import _JobTelemetry

    def _make_ctx(configured: bool) -> JobContext:
        ctx = object.__new__(JobContext)
        ctx._info = SimpleNamespace(job=SimpleNamespace(id="job-x"))
        ctx._early_log_handler = None
        ctx._recording_initialized = True
        ctx._telemetry_state = (
            _JobTelemetry(attributes={}, traces_enabled=True, logs_enabled=True)
            if configured
            else None
        )
        ctx._tempdir = MagicMock()
        ctx._handlers_with_filter = []
        ctx._log_filter = MagicMock()
        return ctx

    with patch("livekit.agents.job._shutdown_telemetry") as mock_shutdown:
        _make_ctx(configured=False)._on_cleanup()
    mock_shutdown.assert_not_called()

    with patch("livekit.agents.job._shutdown_telemetry") as mock_shutdown:
        _make_ctx(configured=True)._on_cleanup()
    mock_shutdown.assert_called_once_with("job-x")


def test_spans_of_a_job_that_disabled_traces_are_not_uploaded() -> None:
    """Concurrent THREAD-mode jobs share the provider and the exporter: a
    recorded job must not cause a disabled job's spans to upload. The disabled
    job's spans keep their full attribution for the integrator's exporters on
    the same provider — Cloud upload is denied by the exportable-jobs registry,
    not by stripping attributes."""
    from livekit.agents.telemetry.traces import (
        _GatedSpanExporter,
        _JobTelemetry,
        _MetadataSpanProcessor,
    )

    processor = _MetadataSpanProcessor()
    disabled_ctx = _make_telemetry_job_ctx(room_id="room-b", job_id="job-b", traces=False)
    span = MagicMock()
    with _job_ctx_active(disabled_ctx):
        processor.on_start(span)
    # attribution is unconditional — the integrator's copy keeps room/job ids
    span.set_attributes.assert_called_once_with({"room_id": "room-b", "job_id": "job-b"})

    export_jobs = {
        "job-a": _JobTelemetry(attributes={}, traces_enabled=True, logs_enabled=True),
        "job-b": _JobTelemetry(attributes={}, traces_enabled=False, logs_enabled=True),
    }
    inner = MagicMock()
    gate = _GatedSpanExporter(inner, export_jobs)

    disabled_span = SimpleNamespace(attributes={"room_id": "room-b", "job_id": "job-b"})
    recorded_span = SimpleNamespace(attributes={"room_id": "room-a", "job_id": "job-a"})

    gate.export([disabled_span])
    inner.export.assert_not_called()

    gate.export([disabled_span, recorded_span])
    inner.export.assert_called_once_with([recorded_span])


def test_concurrent_jobs_stamp_their_own_identity_on_spans() -> None:
    """Span metadata resolves from the originating job's context, not from the
    shared last-configured slot — concurrent jobs must not inherit each other's
    room/job identity."""
    from livekit.agents.telemetry.traces import _MetadataSpanProcessor

    processor = _MetadataSpanProcessor()
    # a second job configured after the first: the shared slot holds job B's data
    processor.set_metadata({"room_id": "room-b", "job_id": "job-b"})

    ctx_a = _make_telemetry_job_ctx(
        room_id="room-a", job_id="job-a", metadata={"lk.simulation.enabled": True}
    )
    span = MagicMock()
    with _job_ctx_active(ctx_a):
        processor.on_start(span)
    span.set_attributes.assert_called_once_with(
        {"room_id": "room-a", "job_id": "job-a", "lk.simulation.enabled": True}
    )

    # outside any job context, the slot is still the fallback
    span = MagicMock()
    processor.on_start(span)
    span.set_attributes.assert_called_once_with({"room_id": "room-b", "job_id": "job-b"})


def test_logs_of_a_job_that_disabled_logs_are_not_uploaded() -> None:
    """Log counterpart: records keep their full attribution for every
    destination; Cloud upload is denied by the exportable-jobs registry."""
    from livekit.agents.telemetry.traces import (
        _GatedLogExporter,
        _JobTelemetry,
        _MetadataLogProcessor,
    )

    processor = _MetadataLogProcessor()

    def _emit(ctx: MagicMock) -> dict[str, Any]:
        log_data = MagicMock()
        log_data.log_record.attributes = {}
        log_data.instrumentation_scope = None
        with _job_ctx_active(ctx):
            processor.on_emit(log_data)
        return log_data.log_record.attributes

    # attribution is unconditional, recording options notwithstanding
    disabled_ctx = _make_telemetry_job_ctx(room_id="room-b", job_id="job-b", logs=False)
    attrs = _emit(disabled_ctx)
    assert attrs["room_id"] == "room-b" and attrs["job_id"] == "job-b"

    export_jobs = {
        "job-a": _JobTelemetry(attributes={}, traces_enabled=True, logs_enabled=True),
        "job-b": _JobTelemetry(attributes={}, traces_enabled=True, logs_enabled=False),
    }
    inner = MagicMock()
    gate = _GatedLogExporter(inner, export_jobs)
    disabled = SimpleNamespace(log_record=SimpleNamespace(attributes=attrs))
    recorded = SimpleNamespace(
        log_record=SimpleNamespace(attributes={"room_id": "room-a", "job_id": "job-a"})
    )

    gate.export([disabled])
    inner.export.assert_not_called()
    gate.export([recorded, disabled])
    inner.export.assert_called_once_with([recorded])


# ---------------------------------------------------------------------------
# Group 4: RecorderIO conditional creation
# ---------------------------------------------------------------------------


async def test_recorder_io_created_when_audio_true() -> None:
    """RecorderIO should be created when recording_options.audio=True and job context exists."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with _patch_job_ctx(mock_ctx, patch_recorder=True):
        await session.start(SimpleAgent(), record=True)
        assert session._recorder_io is not None
        await _cleanup(session)


async def test_recorder_io_not_created_when_audio_false() -> None:
    """RecorderIO should NOT be created when recording_options.audio=False."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with _patch_job_ctx(mock_ctx):
        await session.start(SimpleAgent(), record={"audio": False})

    assert session._recorder_io is None
    await _cleanup(session)
