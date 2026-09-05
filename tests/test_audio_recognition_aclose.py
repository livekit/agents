import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class TestAudioRecognitionAclose:
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
        audio_recognition._end_of_turn_task = None
        audio_recognition._vad_ch = None
        audio_recognition._interruption_ch = None
        audio_recognition._audio_input_atask = None
        audio_recognition._backchannel_boundary_timer = None
        audio_recognition._AudioRecognition__stt_context = None
        audio_recognition._user_turn_span = None
        audio_recognition._user_turn_start = None
        audio_recognition._eot_wait_span = None
        audio_recognition._eot_wait_started_at = None
        audio_recognition._eot_wait_rearms = 0
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
    @pytest.mark.parametrize("task_attr", ["_commit_user_turn_atask", "_end_of_turn_task"])
    async def test_aclose_waits_for_pending_turn_task(self, task_attr: str) -> None:
        audio_recognition = self._create_audio_recognition()
        started = asyncio.Event()
        release = asyncio.Event()

        async def pending_task() -> None:
            started.set()
            await release.wait()

        task = asyncio.create_task(pending_task())
        await started.wait()
        setattr(audio_recognition, task_attr, task)

        close_task = asyncio.create_task(audio_recognition._aclose())
        await asyncio.sleep(0)

        assert not close_task.done()
        assert not task.cancelled()

        release.set()
        await close_task
        assert task.done()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_attr", ["_commit_user_turn_atask", "_end_of_turn_task"])
    async def test_aclose_propagates_cancellation_while_waiting_for_turn_task(
        self, task_attr: str
    ) -> None:
        audio_recognition = self._create_audio_recognition()
        started = asyncio.Event()

        async def pending_task() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(pending_task())
        await started.wait()
        setattr(audio_recognition, task_attr, task)

        close_task = asyncio.create_task(audio_recognition._aclose())
        await asyncio.sleep(0)
        close_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert task.cancelled()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("task_attr", "warning"),
        [
            (
                "_commit_user_turn_atask",
                "error while committing the final user turn on close: RuntimeError",
            ),
            (
                "_end_of_turn_task",
                "error while completing the final user turn on close: RuntimeError",
            ),
        ],
    )
    async def test_aclose_logs_failed_turn_task(
        self,
        task_attr: str,
        warning: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        audio_recognition = self._create_audio_recognition()

        async def failed_task() -> None:
            raise RuntimeError("turn task failed")

        setattr(audio_recognition, task_attr, asyncio.create_task(failed_task()))

        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            await audio_recognition._aclose()

        records = [record for record in caplog.records if record.getMessage() == warning]
        assert len(records) == 1
        assert records[0].exc_info is None

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
