from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.agent_session import RecordingOptions

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD


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


# ---------------------------------------------------------------------------
# Group 1: RecordingOptions normalization (no JobContext)
# ---------------------------------------------------------------------------


async def test_record_true() -> None:
    """record=True should set all recording options to True."""
    session = _create_simple_session()
    await session.start(SimpleAgent(), record=True)
    opts = session._recording_options
    assert opts.audio is True
    assert opts.traces is True
    assert opts.logs is True
    assert opts.transcript is True
    await _cleanup(session)


async def test_record_false() -> None:
    """record=False should set all recording options to False."""
    session = _create_simple_session()
    await session.start(SimpleAgent(), record=False)
    opts = session._recording_options
    assert opts.audio is False
    assert opts.traces is False
    assert opts.logs is False
    assert opts.transcript is False
    await _cleanup(session)


async def test_record_options_partial() -> None:
    """Passing RecordingOptions(audio=False) should disable audio but default the rest to True."""
    session = _create_simple_session()
    custom = RecordingOptions(audio=False)
    await session.start(SimpleAgent(), record=custom)
    opts = session._recording_options
    assert opts.audio is False
    assert opts.traces is True
    assert opts.logs is True
    assert opts.transcript is True
    await _cleanup(session)


async def test_record_not_given_without_job_ctx() -> None:
    """When record is omitted and no JobContext is available, all options should be False."""
    session = _create_simple_session()
    # record= is NOT_GIVEN by default, and get_job_context() raises RuntimeError
    await session.start(SimpleAgent())
    opts = session._recording_options
    assert opts.audio is False
    assert opts.traces is False
    assert opts.logs is False
    assert opts.transcript is False
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 2: init_recording() interaction with mock JobContext
# ---------------------------------------------------------------------------


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


async def test_init_recording_called_with_options() -> None:
    """init_recording should be called with the correct RecordingOptions."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    custom = RecordingOptions(audio=True, traces=True, logs=False, transcript=True)
    with (
        patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx),
        patch("livekit.agents.voice.agent_session.RecorderIO") as MockRecorderIO,
    ):
        mock_recorder = MagicMock()
        mock_recorder.record_input.side_effect = lambda x: x
        mock_recorder.record_output.side_effect = lambda x: x
        mock_recorder.start = AsyncMock()
        mock_recorder.aclose = AsyncMock()
        MockRecorderIO.return_value = mock_recorder

        await session.start(SimpleAgent(), record=custom)

    mock_ctx.init_recording.assert_called_once_with(custom)
    await _cleanup(session)


async def test_init_recording_called_even_when_all_false() -> None:
    """init_recording should still be called when record=False (evals need the OTel infrastructure)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx):
        await session.start(SimpleAgent(), record=False)

    mock_ctx.init_recording.assert_called_once()
    call_opts = mock_ctx.init_recording.call_args[0][0]
    assert isinstance(call_opts, RecordingOptions)
    assert call_opts.audio is False
    assert call_opts.traces is False
    assert call_opts.logs is False
    assert call_opts.transcript is False
    await _cleanup(session)


async def test_init_recording_defers_to_job_enable_recording() -> None:
    """When record= is omitted, the value should come from job.enable_recording."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=True)

    with (
        patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx),
        patch("livekit.agents.voice.agent_session.RecorderIO") as MockRecorderIO,
    ):
        mock_recorder = MagicMock()
        mock_recorder.record_input.side_effect = lambda x: x
        mock_recorder.record_output.side_effect = lambda x: x
        mock_recorder.start = AsyncMock()
        mock_recorder.aclose = AsyncMock()
        MockRecorderIO.return_value = mock_recorder

        await session.start(SimpleAgent())

    # enable_recording=True -> record=True -> all options True -> init_recording called
    mock_ctx.init_recording.assert_called_once()
    call_opts = mock_ctx.init_recording.call_args[0][0]
    assert isinstance(call_opts, RecordingOptions)
    assert call_opts.audio is True
    assert call_opts.traces is True
    assert call_opts.logs is True
    assert call_opts.transcript is True
    await _cleanup(session)


async def test_init_recording_called_when_job_recording_disabled() -> None:
    """init_recording should be called even when job.enable_recording=False (evals need it)."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx(enable_recording=False)

    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx):
        await session.start(SimpleAgent())

    mock_ctx.init_recording.assert_called_once()
    opts = session._recording_options
    assert opts.audio is False
    assert opts.traces is False
    assert opts.logs is False
    assert opts.transcript is False
    await _cleanup(session)


# ---------------------------------------------------------------------------
# Group 3: _upload_session_report() conditional upload
# ---------------------------------------------------------------------------


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


def _make_mock_tagger() -> MagicMock:
    mock = MagicMock()
    mock.evaluations = []
    mock.outcome_reason = None
    return mock


def _get_multipart_part_names(mp_writer: aiohttp.MultipartWriter) -> list[str]:
    """Extract part names from a MultipartWriter by inspecting internal _parts."""
    names = []
    for payload, _enc, _te in mp_writer._parts:
        cd = payload.headers.get("Content-Disposition", "")
        if 'name="header"' in cd:
            names.append("header")
        elif 'name="chat_history"' in cd:
            names.append("chat_history")
        elif 'name="audio"' in cd:
            names.append("audio")
    return names


async def test_upload_returns_early_when_none() -> None:
    """When all options are False, no HTTP request and no session report log should be made."""
    from livekit.agents.telemetry.traces import _upload_session_report

    opts = RecordingOptions(audio=False, traces=False, logs=False, transcript=False)
    report = _make_mock_report(opts)
    tagger = _make_mock_tagger()

    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_http.post = MagicMock()

    mock_logger = MagicMock()
    with patch("livekit.agents.telemetry.traces.get_logger_provider") as mock_glp:
        mock_glp.return_value.get_logger.return_value = mock_logger

        await _upload_session_report(
            agent_name="test-agent",
            cloud_hostname="test.livekit.cloud",
            report=report,
            recording_options=opts,
            tagger=tagger,
            http_session=mock_http,
        )

    mock_http.post.assert_not_called()
    # "session report" log should not be emitted
    for call in mock_logger.emit.call_args_list:
        assert call.kwargs.get("body") != "session report"


def _patch_upload_deps():
    """Context manager to patch AccessToken and logger provider for upload tests."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("livekit.agents.telemetry.traces.get_logger_provider") as mock_glp,
            patch("livekit.agents.telemetry.traces.api.AccessToken") as mock_at,
        ):
            mock_glp.return_value.get_logger.return_value = MagicMock()
            mock_token = MagicMock()
            mock_token.with_observability_grants.return_value = mock_token
            mock_token.with_ttl.return_value = mock_token
            mock_token.to_jwt.return_value = "test-jwt"
            mock_at.return_value = mock_token
            yield

    return _ctx()


def _make_mock_http() -> MagicMock:
    """Create a mock aiohttp.ClientSession with async post."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()

    mock_http = MagicMock(spec=aiohttp.ClientSession)
    mock_post_cm = AsyncMock()
    mock_post_cm.__aenter__.return_value = mock_resp
    mock_http.post.return_value = mock_post_cm
    return mock_http


async def test_upload_transcript_only() -> None:
    """When transcript=True and audio=False, upload should include header + chat_history."""
    from livekit.agents.telemetry.traces import _upload_session_report

    opts = RecordingOptions(audio=False, traces=False, logs=False, transcript=True)
    report = _make_mock_report(opts)
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _upload_session_report(
            agent_name="test-agent",
            cloud_hostname="test.livekit.cloud",
            report=report,
            recording_options=opts,
            tagger=_make_mock_tagger(),
            http_session=mock_http,
        )

    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args
    mp_writer = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert isinstance(mp_writer, aiohttp.MultipartWriter)

    part_names = _get_multipart_part_names(mp_writer)
    assert "header" in part_names
    assert "chat_history" in part_names
    assert "audio" not in part_names


async def test_upload_session_report_sent_without_transcript() -> None:
    """Session report log should be emitted even when transcript=False, if other options are on."""
    from livekit.agents.telemetry.traces import _upload_session_report

    opts = RecordingOptions(audio=True, traces=True, logs=False, transcript=False)
    report = _make_mock_report(opts)
    mock_http = _make_mock_http()

    mock_logger = MagicMock()
    with (
        patch("livekit.agents.telemetry.traces.get_logger_provider") as mock_glp,
        patch("livekit.agents.telemetry.traces.api.AccessToken") as mock_at,
    ):
        mock_glp.return_value.get_logger.return_value = mock_logger
        mock_token = MagicMock()
        mock_token.with_observability_grants.return_value = mock_token
        mock_token.with_ttl.return_value = mock_token
        mock_token.to_jwt.return_value = "test-jwt"
        mock_at.return_value = mock_token

        await _upload_session_report(
            agent_name="test-agent",
            cloud_hostname="test.livekit.cloud",
            report=report,
            recording_options=opts,
            tagger=_make_mock_tagger(),
            http_session=mock_http,
        )

    # "session report" should have been logged
    session_report_calls = [
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "session report"
    ]
    assert len(session_report_calls) == 1

    # but no "chat item" logs (transcript is disabled)
    chat_item_calls = [
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "chat item"
    ]
    assert len(chat_item_calls) == 0


async def test_upload_audio_only_no_file() -> None:
    """When transcript=False, audio=True but no audio file exists, upload has header only."""
    from livekit.agents.telemetry.traces import _upload_session_report

    opts = RecordingOptions(audio=True, traces=False, logs=False, transcript=False)
    report = _make_mock_report(opts)
    report.audio_recording_path = None
    mock_http = _make_mock_http()

    with _patch_upload_deps():
        await _upload_session_report(
            agent_name="test-agent",
            cloud_hostname="test.livekit.cloud",
            report=report,
            recording_options=opts,
            tagger=_make_mock_tagger(),
            http_session=mock_http,
        )

    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args
    mp_writer = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")

    part_names = _get_multipart_part_names(mp_writer)
    assert "chat_history" not in part_names
    assert "audio" not in part_names


async def test_upload_evaluations_emitted_without_logs() -> None:
    """Evaluations should be emitted even when logs=False, as long as something is recorded."""
    from livekit.agents.telemetry.traces import _upload_session_report

    opts = RecordingOptions(audio=True, traces=False, logs=False, transcript=False)
    report = _make_mock_report(opts)
    mock_http = _make_mock_http()

    tagger = _make_mock_tagger()
    tagger.evaluations = [{"name": "test-eval", "verdict": "pass"}]
    tagger.outcome_reason = "all good"

    mock_logger = MagicMock()
    with (
        patch("livekit.agents.telemetry.traces.get_logger_provider") as mock_glp,
        patch("livekit.agents.telemetry.traces.api.AccessToken") as mock_at,
    ):
        mock_glp.return_value.get_logger.return_value = mock_logger
        mock_token = MagicMock()
        mock_token.with_observability_grants.return_value = mock_token
        mock_token.with_ttl.return_value = mock_token
        mock_token.to_jwt.return_value = "test-jwt"
        mock_at.return_value = mock_token

        await _upload_session_report(
            agent_name="test-agent",
            cloud_hostname="test.livekit.cloud",
            report=report,
            recording_options=opts,
            tagger=tagger,
            http_session=mock_http,
        )

    eval_calls = [
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "evaluation"
    ]
    assert len(eval_calls) == 1

    outcome_calls = [
        c for c in mock_logger.emit.call_args_list if c.kwargs.get("body") == "outcome"
    ]
    assert len(outcome_calls) == 1


def test_setup_cloud_tracer_logger_provider_always_created() -> None:
    """LoggerProvider should be set up even when enable_logs=False (needed for evals)."""
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    with (
        patch("livekit.agents.telemetry.traces.api.AccessToken") as mock_at,
        patch("livekit.agents.telemetry.traces.get_logger_provider") as mock_glp,
        patch("livekit.agents.telemetry.traces.set_logger_provider") as mock_slp,
        patch("livekit.agents.telemetry.traces.OTLPLogExporter"),
        patch("livekit.agents.telemetry.traces.BatchLogRecordProcessor"),
        patch("livekit.agents.telemetry.traces.logging"),
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

    # LoggerProvider should still be created and set
    mock_slp.assert_called_once()


# ---------------------------------------------------------------------------
# Group 4: RecorderIO conditional creation
# ---------------------------------------------------------------------------


async def test_recorder_io_created_when_audio_true() -> None:
    """RecorderIO should be created when recording_options.audio=True and job context exists."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    with (
        patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx),
        patch("livekit.agents.voice.agent_session.RecorderIO") as MockRecorderIO,
    ):
        mock_recorder = MagicMock()
        mock_recorder.record_input.side_effect = lambda x: x
        mock_recorder.record_output.side_effect = lambda x: x
        mock_recorder.start = AsyncMock()
        mock_recorder.aclose = AsyncMock()
        MockRecorderIO.return_value = mock_recorder

        await session.start(SimpleAgent(), record=True)
        assert session._recorder_io is not None
        await _cleanup(session)


async def test_recorder_io_not_created_when_audio_false() -> None:
    """RecorderIO should NOT be created when recording_options.audio=False."""
    session = _create_simple_session()
    mock_ctx = _make_mock_job_ctx()

    custom = RecordingOptions(audio=False, traces=True, logs=True, transcript=True)
    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=mock_ctx):
        await session.start(SimpleAgent(), record=custom)

    assert session._recorder_io is None
    await _cleanup(session)
