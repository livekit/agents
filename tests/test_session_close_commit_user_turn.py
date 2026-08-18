"""Regression tests for AgentSession close waiting on the trailing STT transcript.

On a clean (non-error) close, ``AgentSession._aclose_impl`` flushes any in-flight
STT recognition via ``audio_recognition._commit_user_turn(...)`` so the caller's
final utterance is committed before teardown. ``_commit_user_turn`` only *starts*
a background task and returns a future; the very next step, ``activity.aclose()``,
runs ``AudioRecognition._aclose()`` which cancels that same task. If the future is
not awaited first, the trailing transcript is silently discarded (issue #6889).

These tests pin the invariant that the commit future is awaited to completion
before ``activity.aclose()`` gets a chance to cancel it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_session import FakeActions, create_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def _aclose_session(session: object, activity: object) -> None:
    """Close the session and the transcription synchronizer that ``create_session``
    wires onto ``session.output.audio`` (``run_session`` does this cleanup for us
    normally, but these tests drive close by hand)."""
    audio_output = session.output.audio  # type: ignore[attr-defined]
    transcription_sync = (
        audio_output._synchronizer if isinstance(audio_output, _SyncedAudioOutput) else None
    )
    await session.aclose()  # type: ignore[attr-defined]
    if transcription_sync is not None:
        await transcription_sync.aclose()


async def _start_session_mid_utterance() -> tuple[object, AudioRecognition]:
    """Start a session and return (activity, audio_recognition) while the user is
    still being recognized (audio pushed, close not yet requested)."""
    actions = FakeActions()
    # A long trailing utterance whose final transcript lands well after we begin
    # closing (end_time 2.5s + stt_delay 3.0s), mirroring a SIP party that hangs
    # up right as they finish speaking.
    actions.add_user_speech(
        0.5,
        2.5,
        "the mailbox is full and cannot accept new messages at this time",
        stt_delay=3.0,
    )
    session = create_session(
        actions,
        extra_kwargs={"session_close_transcript_timeout": 2.0},
    )
    await session.start(Agent(instructions="You are a helpful assistant."))

    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)
    # let the activity + audio recognition spin up and begin recognizing
    await asyncio.sleep(0.1)

    activity = session._activity
    assert activity is not None
    audio_recognition = activity._audio_recognition
    assert audio_recognition is not None
    return activity, audio_recognition


@pytest.mark.asyncio
async def test_session_close_does_not_cancel_trailing_transcript_commit() -> None:
    """The commit future created on close must be awaited to completion, not
    cancelled by ``activity.aclose()`` racing ahead of it."""
    activity, audio_recognition = await _start_session_mid_utterance()
    session = activity._session

    committed_futs: list[asyncio.Future[str]] = []
    orig_commit = audio_recognition._commit_user_turn

    def _capturing_commit(**kwargs: object) -> asyncio.Future[str]:
        fut = orig_commit(**kwargs)  # type: ignore[arg-type]
        committed_futs.append(fut)
        return fut

    audio_recognition._commit_user_turn = _capturing_commit  # type: ignore[method-assign]

    await _aclose_session(session, activity)

    assert committed_futs, "_commit_user_turn was not called on clean session close"
    commit_fut = committed_futs[-1]
    assert commit_fut.done()
    assert not commit_fut.cancelled(), (
        "trailing user-transcript commit was cancelled by activity teardown instead "
        "of being awaited before activity.aclose() on session close (issue #6889)"
    )


@pytest.mark.asyncio
async def test_session_close_awaits_commit_before_activity_aclose() -> None:
    """``activity.aclose()`` (which cancels the commit task) must not run until the
    commit future returned by ``_commit_user_turn`` has resolved."""
    activity, audio_recognition = await _start_session_mid_utterance()
    session = activity._session

    order: list[str] = []
    commit_called = asyncio.Event()
    release = asyncio.Event()

    def _gated_commit(**kwargs: object) -> asyncio.Future[str]:
        order.append("commit_call")
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def _resolve() -> None:
            await release.wait()
            order.append("commit_resolved")
            fut.set_result("the mailbox is full and cannot accept new messages at this time")

        asyncio.ensure_future(_resolve())
        commit_called.set()
        return fut

    orig_activity_aclose = activity.aclose

    async def _spy_activity_aclose(**kwargs: object) -> None:
        order.append("activity_aclose")
        await orig_activity_aclose(**kwargs)  # type: ignore[arg-type]

    audio_recognition._commit_user_turn = _gated_commit  # type: ignore[method-assign]
    activity.aclose = _spy_activity_aclose  # type: ignore[method-assign]

    audio_output = session.output.audio
    transcription_sync = (
        audio_output._synchronizer if isinstance(audio_output, _SyncedAudioOutput) else None
    )

    close_task = asyncio.ensure_future(session.aclose())
    await asyncio.wait_for(commit_called.wait(), timeout=5.0)

    # _aclose_impl must now be parked awaiting the commit future: activity teardown
    # (which cancels the commit task) must not have started yet.
    assert "activity_aclose" not in order

    release.set()
    await asyncio.wait_for(close_task, timeout=5.0)
    if transcription_sync is not None:
        await transcription_sync.aclose()

    assert order == ["commit_call", "commit_resolved", "activity_aclose"]


async def _assert_teardown_completes_despite_flush(
    make_commit_fut: object,
) -> None:
    """Patch ``_commit_user_turn`` to return a doomed future and assert session
    teardown still runs to completion (``activity.aclose()`` runs, ``close`` is
    emitted, ``_started`` is cleared)."""
    activity, audio_recognition = await _start_session_mid_utterance()
    session = activity._session

    close_events: list[object] = []
    session.on("close", close_events.append)

    activity_aclose_ran = False
    orig_activity_aclose = activity.aclose

    async def _spy_activity_aclose(**kwargs: object) -> None:
        nonlocal activity_aclose_ran
        activity_aclose_ran = True
        await orig_activity_aclose(**kwargs)  # type: ignore[arg-type]

    audio_recognition._commit_user_turn = make_commit_fut  # type: ignore[assignment]
    activity.aclose = _spy_activity_aclose  # type: ignore[method-assign]

    # aclose() must not raise even though the flush future is doomed.
    await _aclose_session(session, activity)

    assert activity_aclose_ran, "teardown was abandoned before activity.aclose()"
    assert close_events, "session close event was never emitted"
    assert session._started is False
    assert session._activity is None


@pytest.mark.asyncio
async def test_session_close_survives_cancelled_flush_future() -> None:
    """A flush future cancelled by a racing ``commit_user_turn()`` must not abort
    session teardown (the mechanism flagged in Devin review of PR #6891)."""

    def _cancelled_commit(**kwargs: object) -> asyncio.Future[str]:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        fut.cancel()
        return fut

    await _assert_teardown_completes_despite_flush(_cancelled_commit)


@pytest.mark.asyncio
async def test_session_close_survives_failed_flush_future() -> None:
    """A flush future that resolves with an exception must be swallowed on close
    (logged) rather than aborting session teardown."""

    def _failing_commit(**kwargs: object) -> asyncio.Future[str]:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("end-of-turn detection blew up during flush"))
        return fut

    await _assert_teardown_completes_despite_flush(_failing_commit)
