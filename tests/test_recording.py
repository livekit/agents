from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import astuple
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.telemetry.traces import _upload_session_report
from livekit.agents.voice.agent_session import RecordingOptions

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

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
    report.recording_options = recording_options or RecordingOptions()
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
    return mock


def _make_mock_http() -> MagicMock:
    """Create a mock aiohttp.ClientSession with async post."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_post_cm = AsyncMock()
    mock_post_cm.__aenter__.return_value = mock_resp
    mock_http.post.return_value = mock_post_cm
    return mock_http


@contextlib.contextmanager
def _patch_upload_deps() -> Iterator[MagicMock]:
    """Patch OTel logger provider and AccessToken. Yields the mock logger for assertions."""
    mock_logger = MagicMock()
    with (
        patch(f"{_TRACES_MOD}.get_logger_provider") as mock_glp,
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
    ):
        mock_glp.return_value.get_logger.return_value = mock_logger
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
) -> None:
    """Call _upload_session_report with sensible defaults."""
    await _upload_session_report(
        agent_name="test-agent",
        cloud_hostname="test.livekit.cloud",
        report=report,
        tagger=tagger or _make_mock_tagger(),
        http_session=http_session or _make_mock_http(),
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


# ---------------------------------------------------------------------------
# Group 1: RecordingOptions normalization (no JobContext)
# ---------------------------------------------------------------------------

_ALL_TRUE = (True, True, True, True)
_ALL_FALSE = (False, False, False, False)


@pytest.mark.parametrize(
    "record, expected",
    [
        pytest.param(True, _ALL_TRUE, id="record=True"),
        pytest.param(False, _ALL_FALSE, id="record=False"),
        pytest.param(RecordingOptions(audio=False), (False, True, True, True), id="partial"),
    ],
)
async def test_record_normalization(
    record: bool | RecordingOptions, expected: tuple[bool, ...]
) -> None:
    session = _create_simple_session()
    await session.start(SimpleAgent(), record=record)
    opts = session._recording_options
    assert astuple(opts) == expected
    await _cleanup(session)


async def test_record_not_given_without_job_ctx() -> None:
    """When record is omitted and no JobContext is available, all options should be False."""
    session = _create_simple_session()
    await session.start(SimpleAgent())
    assert astuple(session._recording_options) == _ALL_FALSE
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 2: init_recording() interaction with mock JobContext
# ---------------------------------------------------------------------------


async def test_init_recording_called_with_options() -> None:
    """init_recording should be called with the correct RecordingOptions."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()
    custom = RecordingOptions(audio=True, traces=True, logs=False, transcript=True)

    with _patch_job_ctx(mock_ctx, patch_recorder=True):
        await session.start(SimpleAgent(), record=custom)

    mock_ctx.init_recording.assert_called_once_with(custom)
    await _cleanup(session)


async def test_init_recording_called_even_when_all_false() -> None:
    """init_recording is always called when job context exists (evals need OTel infrastructure)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with _patch_job_ctx(mock_ctx):
        await session.start(SimpleAgent(), record=False)

    mock_ctx.init_recording.assert_called_once()
    assert astuple(mock_ctx.init_recording.call_args[0][0]) == _ALL_FALSE
    await _cleanup(session)


async def test_init_recording_defers_to_job_enable_recording() -> None:
    """When record= is omitted, the value should come from job.enable_recording."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=True)

    with _patch_job_ctx(mock_ctx, patch_recorder=True):
        await session.start(SimpleAgent())

    mock_ctx.init_recording.assert_called_once()
    assert astuple(mock_ctx.init_recording.call_args[0][0]) == _ALL_TRUE
    await _cleanup(session)


async def test_init_recording_called_when_job_recording_disabled() -> None:
    """init_recording should be called even when job.enable_recording=False (evals need it)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=False)

    with _patch_job_ctx(mock_ctx):
        await session.start(SimpleAgent())

    mock_ctx.init_recording.assert_called_once()
    assert astuple(session._recording_options) == _ALL_FALSE
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 3: _upload_session_report() conditional upload
# ---------------------------------------------------------------------------


async def test_upload_returns_early_when_none() -> None:
    """When all options are False, no HTTP request and no session report log should be made."""
    report = _make_mock_report(
        RecordingOptions(audio=False, traces=False, logs=False, transcript=False)
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
    report = _make_mock_report(
        RecordingOptions(audio=False, traces=False, logs=False, transcript=True)
    )
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
    report = _make_mock_report(
        RecordingOptions(audio=True, traces=True, logs=False, transcript=False)
    )
    mock_http = _make_mock_http()

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, http_session=mock_http)

    bodies = [c.kwargs.get("body") for c in mock_logger.emit.call_args_list]
    assert "session report" in bodies
    assert "chat item" not in bodies


async def test_upload_audio_only_no_file() -> None:
    """When transcript=False, audio=True but no audio file exists, no upload is made."""
    report = _make_mock_report(
        RecordingOptions(audio=True, traces=False, logs=False, transcript=False)
    )
    report.audio_recording_path = None
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _call_upload(report, http_session=mock_http)

    mock_http.post.assert_not_called()


async def test_upload_evaluations_emitted_without_logs() -> None:
    """Evaluations should be emitted even when logs=False, as long as something is recorded."""
    report = _make_mock_report(
        RecordingOptions(audio=True, traces=False, logs=False, transcript=False)
    )
    tagger = _make_mock_tagger(
        evaluations=[{"name": "test-eval", "verdict": "pass"}],
        outcome_reason="all good",
    )

    with _patch_upload_deps() as mock_logger:
        await _call_upload(report, tagger=tagger)

    bodies = [c.kwargs.get("body") for c in mock_logger.emit.call_args_list]
    assert bodies.count("evaluation") == 1
    assert bodies.count("outcome") == 1


def test_setup_cloud_tracer_logger_provider_always_created() -> None:
    """LoggerProvider should be set up even when enable_logs=False (needed for evals)."""
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    with (
        patch(f"{_TRACES_MOD}.api.AccessToken") as mock_at,
        patch(f"{_TRACES_MOD}.get_logger_provider") as mock_glp,
        patch(f"{_TRACES_MOD}.set_logger_provider") as mock_slp,
        patch(f"{_TRACES_MOD}.OTLPLogExporter"),
        patch(f"{_TRACES_MOD}.BatchLogRecordProcessor"),
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
            cloud_hostname="test.livekit.cloud",
            enable_traces=False,
            enable_logs=False,
        )

    mock_slp.assert_called_once()


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
        await session.start(SimpleAgent(), record=RecordingOptions(audio=False))

    assert session._recorder_io is None
    await _cleanup(session)
