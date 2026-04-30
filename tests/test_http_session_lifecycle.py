"""
Tests for AgentSession-owned http_context lifecycle.

When running outside a job context (tests, scripts, ad-hoc usage) there is no
process-level http_session bound to the event loop. AgentSession sets one up in
start() and tears it down in aclose() so that STT/TTS/etc. can call
``utils.http_context.http_session()`` without a job process running.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    NotGivenOr,
    stt as stt_module,
    tts as tts_module,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import http_context

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_vad import FakeVAD

_AGENT_SESSION_MOD = "livekit.agents.voice.agent_session"


class _CapturingSTT(stt_module.STT):
    """STT that records the http session it sees during stream() — no network."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=stt_module.STTCapabilities(streaming=True, interim_results=False),
        )
        self.captured_session: aiohttp.ClientSession | None = None

    async def _recognize_impl(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _NoopSTTStream:
        # The point of the test: this call must succeed inside an active
        # AgentSession, regardless of whether a job context is set.
        self.captured_session = http_context.http_session()
        return _NoopSTTStream(stt=self, conn_options=conn_options)


class _NoopSTTStream(stt_module.RecognizeStream):
    async def _run(self) -> None:
        async for _ in self._input_ch:
            pass


class _CapturingTTS(tts_module.TTS):
    """TTS that records the http session it sees during synthesize() — no network."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts_module.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.captured_session: aiohttp.ClientSession | None = None

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _NoopChunkedStream:
        self.captured_session = http_context.http_session()
        return _NoopChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _NoopChunkedStream(tts_module.ChunkedStream):
    async def _run(self, output_emitter: tts_module.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="noop",
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.flush()


class _NoopAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="noop")


def _make_session(
    stt: _CapturingSTT | None = None, tts: _CapturingTTS | None = None
) -> AgentSession:
    session = AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[], min_silence_duration=0.5, min_speech_duration=0.05),
        stt=stt or _CapturingSTT(),
        llm=FakeLLM(fake_responses=[]),
        tts=tts or _CapturingTTS(),
        # disable AEC warmup so we don't leak the timer
        aec_warmup_duration=None,
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session


async def test_http_session_available_during_agent_session() -> None:
    """Inside a started AgentSession, http_context.http_session() returns a working session.

    After aclose, the context is reset and http_session() raises again.
    """
    # Sanity: nothing set in this task before start
    with pytest.raises(RuntimeError):
        http_context.http_session()

    capturing_stt = _CapturingSTT()
    session = _make_session(stt=capturing_stt)

    await session.start(_NoopAgent())

    # The set in start() propagates to this task's context (start awaited here).
    sess = http_context.http_session()
    assert isinstance(sess, aiohttp.ClientSession)
    assert not sess.closed

    # The STT.stream() called during activity start sees the same session.
    assert capturing_stt.captured_session is sess

    await session.aclose()

    # After aclose the underlying session is closed and the contextvar is reset.
    assert sess.closed
    with pytest.raises(RuntimeError):
        http_context.http_session()


async def test_concurrent_sessions_in_separate_tasks_are_isolated() -> None:
    """Two AgentSessions started inside their own asyncio.Task each get their own
    http session. Closing one does not affect the other.
    """
    barrier = asyncio.Event()

    async def session_worker() -> tuple[aiohttp.ClientSession, aiohttp.ClientSession]:
        capturing_stt = _CapturingSTT()
        session = _make_session(stt=capturing_stt)

        await session.start(_NoopAgent())
        seen = http_context.http_session()
        # wait so both tasks are alive simultaneously — proves isolation
        await barrier.wait()
        # session is still live and accessible from this task's context
        still_seen = http_context.http_session()
        await session.aclose()
        return seen, still_seen

    task_a = asyncio.create_task(session_worker())
    task_b = asyncio.create_task(session_worker())

    # let both reach the barrier
    await asyncio.sleep(0.05)
    barrier.set()

    (a_first, a_second), (b_first, b_second) = await asyncio.gather(task_a, task_b)

    # each task sees a stable session before close
    assert a_first is a_second
    assert b_first is b_second

    # tasks see different sessions — not a single global one
    assert a_first is not b_first

    # both got closed independently
    assert a_first.closed
    assert b_first.closed


def _mock_job_ctx() -> MagicMock:
    """Build the minimal JobContext mock that AgentSession.start() reads from."""
    mock = MagicMock()
    mock.job.enable_recording = False
    mock.job.id = "test-job-id"
    mock.job.agent_name = "test-agent"
    mock.room.name = "test-room"
    mock._primary_agent_session = None
    mock.session_directory = Path("/tmp/test-session")
    return mock


async def test_session_does_not_own_http_ctx_inside_job_context(
    job_process: None,  # fixture sets up http_context for the test
) -> None:
    """When AgentSession runs inside a real job context, it must not overwrite or
    close the process-level http_context on aclose.
    """
    outer_session = http_context.http_session()
    assert not outer_session.closed

    session = _make_session()

    with patch(f"{_AGENT_SESSION_MOD}.get_job_context", return_value=_mock_job_ctx()):
        await session.start(_NoopAgent())

        # AgentSession reuses the existing context — same ClientSession surfaces.
        assert http_context.http_session() is outer_session

        await session.aclose()

    # The job-context session is still alive — only the job_process fixture closes it.
    assert not outer_session.closed
    assert http_context.http_session() is outer_session


async def test_nested_sessions_in_same_task_share_http_ctx() -> None:
    """A second AgentSession started inside a still-running outer session (same
    task) must reuse the outer's http session and not close it on aclose.
    """
    outer = _make_session()
    await outer.start(_NoopAgent())
    outer_session = http_context.http_session()
    assert outer._owned_http_session_ctx is True

    inner = _make_session()
    await inner.start(_NoopAgent())

    # inner sees the contextvar already set → does not take ownership
    assert inner._owned_http_session_ctx is False
    assert http_context.http_session() is outer_session

    await inner.aclose()

    # outer's session is unaffected by inner's close
    assert not outer_session.closed
    assert http_context.http_session() is outer_session

    await outer.aclose()
    assert outer_session.closed


async def test_start_failure_cleans_up_http_ctx() -> None:
    """If start() fails after setting up the http session, aclose() must still
    clean it up. Otherwise __aexit__ on the async-with would leak the factory.
    """
    session = _make_session()

    with patch.object(AgentSession, "_update_activity", side_effect=RuntimeError("boom")):
        with pytest.raises(BaseException):  # noqa: B017,PT011 - want any failure
            await session.start(_NoopAgent())

    # the session never reached _started=True, but the http_session ctx was set
    assert session._started is False
    assert session._owned_http_session_ctx is True

    await session.aclose()

    # aclose must clean up even when start failed
    assert session._owned_http_session_ctx is False
    with pytest.raises(RuntimeError):
        http_context.http_session()
