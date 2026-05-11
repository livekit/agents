"""
Tests to validate that AudioRecognition.aclose() handles pre-cancelled tasks gracefully.

Before the fix, if _commit_user_turn_atask or _end_of_turn_task were cancelled
before aclose() was called, awaiting them would raise CancelledError and
propagate up, causing cleanup to fail.

The fix wraps these awaits in try-except blocks to catch CancelledError.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition


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
        audio_recognition._stt_atask = None
        audio_recognition._vad_atask = None
        audio_recognition._commit_user_turn_atask = None
        audio_recognition._end_of_turn_task = None

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
        await audio_recognition.aclose()

        # Verify cleanup completed
        assert audio_recognition._closing.is_set()
        # Both tasks are now done (not orphaned)
        assert commit_task.done()
        assert end_of_turn_task.done()
