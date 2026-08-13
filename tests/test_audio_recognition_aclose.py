"""
Tests to validate that AudioRecognition.aclose() handles pre-cancelled tasks gracefully.

Before the fix, if _commit_user_turn_atask or _end_of_turn_task were cancelled
before aclose() was called, awaiting them would raise CancelledError and
propagate up, causing cleanup to fail.

The fix wraps these awaits in try-except blocks to catch CancelledError.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class TestAudioRecognitionAclose:
    """Test cases for AudioRecognition.aclose() handling cancelled tasks."""

    def _create_audio_recognition(self) -> AudioRecognition:
        """Create an AudioRecognition instance with mocked dependencies."""
        with patch.object(AudioRecognition, "__init__", lambda self, *args, **kwargs: None):
            audio_recognition = AudioRecognition.__new__(AudioRecognition)

        # Initialize required attributes manually
        audio_recognition._session = MagicMock()
        audio_recognition._hooks = MagicMock()
        audio_recognition._closing = asyncio.Event()
        audio_recognition._tasks = set()
        audio_recognition._stt_consumer_atask = None
        audio_recognition._stt_pipeline = None
        audio_recognition._vad_atask = None
        audio_recognition._interruption_atask = None
        audio_recognition._turn_detector_stream = None
        audio_recognition._commit_user_turn_atask = None
        audio_recognition._session_close_commit_user_turn_atask = None
        audio_recognition._session_close_end_of_turn_atask = None
        audio_recognition._end_of_turn_task = None
        audio_recognition._vad_ch = None
        audio_recognition._interruption_ch = None
        audio_recognition._audio_input_atask = None
        audio_recognition._backchannel_boundary_timer = None
        audio_recognition._AudioRecognition__stt_context = None
        audio_recognition._user_turn_span = None
        audio_recognition._user_turn_start = None
        audio_recognition._transcription_timeout_handle = None

        return audio_recognition

    @pytest.mark.asyncio
    async def test_unprotected_await_blocks_subsequent_cleanup(self):
        """
        DEMONSTRATES THE BUG: Mimics the old aclose() pattern where CancelledError
        on the first task prevents the second task from ever being cleaned up.

        Old aclose() pattern:
            if self._commit_user_turn_atask is not None:
                await self._commit_user_turn_atask  # <-- raises CancelledError!
            # ... other cleanup ...
            if self._end_of_turn_task is not None:
                await self._end_of_turn_task  # <-- NEVER REACHED

        This leaves _end_of_turn_task orphaned and never awaited.
        """

        async def long_running_task():
            await asyncio.sleep(10)

        # Create two tasks like aclose() has
        commit_user_turn_atask = asyncio.create_task(long_running_task())
        end_of_turn_task = asyncio.create_task(long_running_task())

        # Cancel the first task (simulating external cancellation before aclose)
        commit_user_turn_atask.cancel()
        await asyncio.sleep(0)

        second_task_awaited = False

        async def old_aclose_pattern():
            """Mimics the OLD aclose() without try-except protection."""
            nonlocal second_task_awaited

            # First await - this raises CancelledError and exits
            if commit_user_turn_atask is not None:
                await commit_user_turn_atask

            # Second await - NEVER REACHED due to exception above
            if end_of_turn_task is not None:
                second_task_awaited = True
                end_of_turn_task.cancel()
                try:
                    await end_of_turn_task
                except asyncio.CancelledError:
                    pass

        # Run old_aclose_pattern with a timeout to prove it fails
        with pytest.raises(asyncio.CancelledError):
            await old_aclose_pattern()

        # The second task was never awaited - it's orphaned
        assert not second_task_awaited, "Second task cleanup was never reached"
        assert not end_of_turn_task.done(), "Second task is still running (orphaned)"

        # Cleanup the orphaned task
        end_of_turn_task.cancel()
        try:
            await end_of_turn_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_aclose_handles_precancelled_tasks_gracefully(self):
        """
        PROVES THE FIX: Both tasks are properly cleaned up even when pre-cancelled.

        Fixed aclose() pattern:
            if self._commit_user_turn_atask is not None:
                try:
                    await self._commit_user_turn_atask
                except asyncio.CancelledError:
                    pass  # <-- Catches the error, continues cleanup
            # ... other cleanup ...
            if self._end_of_turn_task is not None:
                try:
                    await self._end_of_turn_task
                except asyncio.CancelledError:
                    pass  # <-- This is now reached!
        """
        audio_recognition = self._create_audio_recognition()

        async def long_running_task():
            await asyncio.sleep(10)

        # Create and cancel both tasks before aclose()
        commit_task = asyncio.create_task(long_running_task())
        commit_task.cancel()

        end_of_turn_task = asyncio.create_task(long_running_task())
        end_of_turn_task.cancel()

        await asyncio.sleep(0)

        audio_recognition._commit_user_turn_atask = commit_task
        audio_recognition._end_of_turn_task = end_of_turn_task

        # With the fix, aclose() completes without raising
        await audio_recognition._aclose()

        # Verify cleanup completed
        assert audio_recognition._closing.is_set()
        # Both tasks are now done (not orphaned)
        assert commit_task.done()
        assert end_of_turn_task.done()

    @pytest.mark.asyncio
    async def test_aclose_propagates_outer_cancellation_after_finishing_cleanup(self) -> None:
        """Outer cancellation waits for the close flush and all remaining cleanup."""
        audio_recognition = self._create_audio_recognition()
        flush_started = asyncio.Event()
        release_flush = asyncio.Event()

        async def flush() -> None:
            flush_started.set()
            await release_flush.wait()

        flush_task = asyncio.create_task(flush())
        audio_recognition._commit_user_turn_atask = flush_task
        audio_recognition._session_close_commit_user_turn_atask = flush_task
        stt_pipeline = MagicMock()
        stt_pipeline.aclose = AsyncMock()
        audio_recognition._stt_pipeline = stt_pipeline
        turn_detector_stream = MagicMock()
        turn_detector_stream.aclose = AsyncMock()
        audio_recognition._turn_detector_stream = turn_detector_stream
        background_tasks = [asyncio.create_task(asyncio.Event().wait()) for _ in range(4)]
        audio_recognition._tasks = {background_tasks[0]}
        audio_recognition._stt_consumer_atask = background_tasks[1]
        audio_recognition._vad_atask = background_tasks[2]
        audio_recognition._interruption_atask = background_tasks[3]
        close_task = asyncio.create_task(audio_recognition._aclose())

        try:
            await flush_started.wait()
            close_task.cancel()
            await asyncio.sleep(0)

            assert not close_task.done()
            assert not flush_task.cancelled()

            release_flush.set()
            with pytest.raises(asyncio.CancelledError):
                await close_task

            assert flush_task.done()
            assert not flush_task.cancelled()
            stt_pipeline.aclose.assert_awaited_once()
            turn_detector_stream.aclose.assert_awaited_once()
            assert all(task.cancelled() for task in background_tasks)
        finally:
            release_flush.set()
            await asyncio.gather(close_task, flush_task, *background_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flush_task_attr", ["_commit_user_turn_atask", "_end_of_turn_task"])
    async def test_aclose_cancels_non_session_close_flushes(self, flush_task_attr: str) -> None:
        """Generic commit and EOU work retain the normal activity-teardown cancellation policy."""
        audio_recognition = self._create_audio_recognition()
        flush_task = asyncio.create_task(asyncio.Event().wait())
        setattr(audio_recognition, flush_task_attr, flush_task)

        await audio_recognition._aclose()

        assert flush_task.cancelled()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flush_task_attr",
        ["_session_close_commit_user_turn_atask", "_session_close_end_of_turn_atask"],
    )
    async def test_aclose_propagates_session_close_flush_errors_after_cleanup(
        self, flush_task_attr: str
    ) -> None:
        """A failed close-owned flush is observed only after the remaining teardown finishes."""
        audio_recognition = self._create_audio_recognition()

        async def fail_flush() -> None:
            raise RuntimeError("transcript flush failed")

        flush_task = asyncio.create_task(fail_flush())
        setattr(audio_recognition, flush_task_attr, flush_task)
        if flush_task_attr == "_session_close_commit_user_turn_atask":
            audio_recognition._commit_user_turn_atask = flush_task
        else:
            audio_recognition._end_of_turn_task = flush_task

        turn_detector_stream = MagicMock()
        turn_detector_stream.aclose = AsyncMock()
        audio_recognition._turn_detector_stream = turn_detector_stream

        with pytest.raises(RuntimeError, match="transcript flush failed"):
            await audio_recognition._aclose()

        turn_detector_stream.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("is_recording", "expected_end_count"), [(True, 1), (False, 0)])
    async def test_aclose_finalizes_user_turn_span(
        self, is_recording: bool, expected_end_count: int
    ):
        audio_recognition = self._create_audio_recognition()

        span = MagicMock()
        span.is_recording.return_value = is_recording
        audio_recognition._user_turn_span = span
        audio_recognition._user_turn_start = 123.0

        await audio_recognition._aclose()

        assert span.end.call_count == expected_end_count
        assert audio_recognition._user_turn_span is None
        assert audio_recognition._user_turn_start is None

    @pytest.mark.asyncio
    async def test_aclose_ends_user_turn_span_when_teardown_raises(self):
        audio_recognition = self._create_audio_recognition()

        span = MagicMock()
        span.is_recording.return_value = True
        audio_recognition._user_turn_span = span
        audio_recognition._user_turn_start = 123.0

        timeout_handle = MagicMock()
        audio_recognition._transcription_timeout_handle = timeout_handle

        stt_pipeline = MagicMock()
        stt_pipeline.aclose = AsyncMock(side_effect=RuntimeError("vendor stream teardown failed"))
        audio_recognition._stt_pipeline = stt_pipeline

        with pytest.raises(RuntimeError, match="vendor stream teardown failed"):
            await audio_recognition._aclose()

        span.end.assert_called_once()
        timeout_handle.cancel.assert_called_once()
        assert audio_recognition._user_turn_span is None
        assert audio_recognition._user_turn_start is None
