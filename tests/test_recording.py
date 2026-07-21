from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.telemetry.traces import _upload_session_report
from livekit.agents.voice.agent_session import (
    _RECORDING_ALL_OFF,
    _RECORDING_ALL_ON,
    RecordingOptions,
)
from livekit.agents.voice.recorder_io.recorder_io import _split_frame
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
    report.recording_options = recording_options or _RECORDING_ALL_ON.copy()
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
    assert session._recording_options == expected
    await _cleanup(session)


async def test_record_not_given_without_job_ctx() -> None:
    """When record is omitted and no JobContext is available, all options should be False."""
    session = _create_simple_session()
    await session.start(SimpleAgent())
    assert session._recording_options == _RECORDING_ALL_OFF
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
    assert session._recording_options == _RECORDING_ALL_OFF
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


async def test_upload_session_report_sent_without_transcript() -> None:
    """Session report log should be emitted even when transcript=False, if other options are on."""
    report = _make_mock_report({"audio": True, "traces": True, "logs": False, "transcript": False})
    mock_http = _make_mock_http()

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, http_session=mock_http)

    bodies = [c.kwargs.get("body") for c in mock_logger.emit.call_args_list]
    assert "session report" in bodies
    assert "chat item" not in bodies


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
        "lk.simulation.run_id": "run-1",
    }

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, metadata=metadata)

    attrs = mock_logger.provider.get_logger.call_args_list[0].kwargs["attributes"]
    assert attrs["lk.simulation.enabled"] is True
    assert attrs["lk.simulation.run_id"] == "run-1"
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
        "lk.simulation.run_id": "run-1",
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
            metadata={"lk.simulation.enabled": True, "lk.simulation.run_id": "run-1"},
        )

    mock_slp.assert_called_once()
    assert not any(k.startswith("lk.simulation.") for k in mock_resource_create.call_args.args[0])
    # OTLP exporter should NOT be created when enable_logs=False
    mock_exporter.assert_not_called()
    mock_blrp.assert_not_called()


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


# ---------------------------------------------------------------------------
# Group 5: _split_frame (encode-path helper)
# ---------------------------------------------------------------------------


def _ramp_frame(num_samples: int, num_channels: int, sample_rate: int = 24000) -> rtc.AudioFrame:
    """A frame whose samples are a monotonic ramp, so splits can be checked for alignment."""
    arr = np.arange(num_samples * num_channels, dtype=np.int16)
    return rtc.AudioFrame(
        data=arr.tobytes(),
        num_channels=num_channels,
        samples_per_channel=num_samples,
        sample_rate=sample_rate,
    )


@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_split_frame_is_consistent_and_lossless(num_channels: int, fraction: float) -> None:
    """`rtc.AudioFrame.data` is a memoryview of int16 *samples*, not bytes.

    A split must keep each half's data length in sync with its samples_per_channel and
    must neither drop nor duplicate samples. This guards the regression where the helper
    indexed the buffer in bytes and produced corrupt frames on interrupted/paused playback.
    """
    n = 240
    frame = _ramp_frame(n, num_channels)
    left, right = _split_frame(frame, frame.duration * fraction)

    # each half is internally consistent
    assert len(left.data) == left.samples_per_channel * left.num_channels
    assert len(right.data) == right.samples_per_channel * right.num_channels

    # no samples lost or duplicated across the split
    assert left.samples_per_channel + right.samples_per_channel == n
    recon = np.concatenate(
        [
            np.frombuffer(bytes(left.data), dtype=np.int16),
            np.frombuffer(bytes(right.data), dtype=np.int16),
        ]
    )
    assert np.array_equal(recon, np.arange(n * num_channels, dtype=np.int16))


def test_split_frame_boundaries() -> None:
    """Splitting at or beyond the edges returns an empty half and the original."""
    frame = _ramp_frame(100, 1)

    empty, whole = _split_frame(frame, 0.0)
    assert empty.samples_per_channel == 0 and len(empty.data) == 0
    assert whole.samples_per_channel == 100

    whole2, empty2 = _split_frame(frame, frame.duration * 2)
    assert whole2.samples_per_channel == 100
    assert empty2.samples_per_channel == 0 and len(empty2.data) == 0
