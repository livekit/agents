from __future__ import annotations

import asyncio
import contextlib
import inspect
import ssl
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
        mock_token = MagicMock()
        mock_token.with_observability_grants.return_value = mock_token
        mock_token.with_ttl.return_value = mock_token
        mock_token.to_jwt.return_value = "test-jwt"
        mock_at.return_value = mock_token
        yield mock_logger


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
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    with (
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
        patch(f"{_TRACES_MOD}.get_logger_provider") as mock_glp,
        patch(f"{_TRACES_MOD}.set_logger_provider") as mock_slp,
        patch(f"{_TRACES_MOD}.OTLPLogExporter") as mock_exporter,
        patch(f"{_TRACES_MOD}.BatchLogRecordProcessor") as mock_blrp,
        patch(f"{_TRACES_MOD}.Resource.create") as mock_resource_create,
        patch(f"{_TRACES_MOD}.logging"),
    ):
        mock_token = MagicMock()
        mock_token.with_observability_grants.return_value = mock_token
        mock_token.with_ttl.return_value = mock_token
        mock_token.to_jwt.return_value = "test-jwt"
        mock_at.return_value = mock_token
        # Return a non-LoggerProvider so the code creates a new one
        mock_glp.return_value = MagicMock()

        _setup_cloud_tracer(
            room_id="room-1",
            job_id="job-1",
            **_observability_endpoint_arg(_setup_cloud_tracer),
            enable_traces=False,
            enable_logs=False,
            metadata={"lk.simulation.enabled": True},
        )

    mock_slp.assert_called_once()
    assert not any(k.startswith("lk.simulation.") for k in mock_resource_create.call_args.args[0])
    # OTLP exporter should NOT be created when enable_logs=False
    mock_exporter.assert_not_called()
    mock_blrp.assert_not_called()


def _resource_attrs_for_env(env: dict[str, str]) -> dict[str, Any]:
    """Run _setup_cloud_tracer under the given os.environ and return the dict
    passed to Resource.create."""
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    with (
        patch.dict("os.environ", env, clear=True),
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
        patch(f"{_TRACES_MOD}.get_logger_provider", return_value=MagicMock()),
        patch(f"{_TRACES_MOD}.set_logger_provider"),
        patch(f"{_TRACES_MOD}.OTLPLogExporter"),
        patch(f"{_TRACES_MOD}.BatchLogRecordProcessor"),
        patch(f"{_TRACES_MOD}.Resource.create") as mock_resource_create,
        patch(f"{_TRACES_MOD}.logging"),
    ):
        mock_token = MagicMock()
        mock_token.with_observability_grants.return_value = mock_token
        mock_token.with_ttl.return_value = mock_token
        mock_token.to_jwt.return_value = "test-jwt"
        mock_at.return_value = mock_token

        _setup_cloud_tracer(
            room_id="room-1",
            job_id="job-1",
            **_observability_endpoint_arg(_setup_cloud_tracer),
            enable_traces=False,
            enable_logs=False,
        )
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
