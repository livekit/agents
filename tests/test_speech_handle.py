"""Tests for SpeechHandle

This test verifies that _wait_for_generation does not hang when a speech is
interrupted before the generation completes.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.voice.speech_handle import SpeechHandle


class TestSpeechHandleWaitForGeneration:
    """Test suite for SpeechHandle._wait_for_generation hang fix."""

    @pytest.mark.asyncio
    async def test_wait_for_generation_returns_when_interrupted(self) -> None:
        """Test that _wait_for_generation returns immediately when speech is interrupted.

        _wait_for_generation should not hang waiting for the generation future.
        """
        speech = SpeechHandle.create()

        # Authorize generation (creates the generation future)
        speech._authorize_generation()

        # Interrupt the speech before generation completes
        speech.interrupt()

        # _wait_for_generation should return immediately because the speech
        # is interrupted, even though the generation future is not resolved
        try:
            await asyncio.wait_for(speech._wait_for_generation(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "_wait_for_generation hung after interrupt"
            )

    @pytest.mark.asyncio
    async def test_wait_for_generation_returns_when_generation_done(self) -> None:
        """Test that _wait_for_generation returns when generation completes normally."""
        speech = SpeechHandle.create()

        # Authorize generation
        speech._authorize_generation()

        # Mark generation done in background
        async def mark_done_later():
            await asyncio.sleep(0.1)
            speech._mark_generation_done()

        asyncio.create_task(mark_done_later())

        # Should complete when generation is done
        try:
            await asyncio.wait_for(speech._wait_for_generation(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("_wait_for_generation did not return after generation was done")

    @pytest.mark.asyncio
    async def test_wait_for_generation_interrupt_during_wait(self) -> None:
        """Test that _wait_for_generation returns if interrupted while waiting."""
        speech = SpeechHandle.create()

        # Authorize generation
        speech._authorize_generation()

        # Interrupt after a short delay
        async def interrupt_later():
            await asyncio.sleep(0.1)
            speech.interrupt()

        asyncio.create_task(interrupt_later())

        # Should return when interrupt happens
        try:
            await asyncio.wait_for(speech._wait_for_generation(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("_wait_for_generation hung - did not respond to interrupt")

        assert speech.interrupted

    @pytest.mark.asyncio
    async def test_multiple_speeches_with_interrupts(self) -> None:
        """Test processing multiple speeches where some are interrupted.

        Simulates the mainTask queue processing scenario.
        """
        speeches = [SpeechHandle.create() for _ in range(3)]

        # Interrupt the middle speech before authorization
        speeches[1].interrupt()

        # Process all speeches (simulating mainTask)
        for speech in speeches:
            speech._authorize_generation()

            # For non-interrupted speeches, mark generation done
            if not speech.interrupted:
                speech._mark_generation_done()

            # This should not hang even for interrupted speeches
            try:
                await asyncio.wait_for(speech._wait_for_generation(), timeout=1.0)
            except asyncio.TimeoutError:
                pytest.fail(
                    f"_wait_for_generation hung for speech {speech.id} "
                    f"(interrupted={speech.interrupted})"
                )

    @pytest.mark.asyncio
    async def test_wait_for_generation_raises_without_authorization(self) -> None:
        """Test that _wait_for_generation raises if no generation is running."""
        speech = SpeechHandle.create()

        with pytest.raises(RuntimeError, match="no active generation is running"):
            await speech._wait_for_generation()

    @pytest.mark.asyncio
    async def test_scheduling_task_simulation(self) -> None:
        """Simulate the scheduling task flow that was hanging.

        This reproduces the exact scenario from agent_activity._scheduling_task.
        """
        # Create a queue of speeches
        speech_queue: list[tuple[int, int, SpeechHandle]] = []

        speech1 = SpeechHandle.create()
        speech2 = SpeechHandle.create()
        speech3 = SpeechHandle.create()

        # Interrupt speech2 before it's processed (simulating interrupt while in queue)
        speech2.interrupt()

        speech_queue.append((5, 1, speech1))
        speech_queue.append((5, 2, speech2))
        speech_queue.append((5, 3, speech3))

        processed_speeches: list[str] = []

        # Simulate scheduling_task loop
        async def scheduling_task():
            while speech_queue:
                _, _, speech = speech_queue.pop(0)

                if speech.done():
                    continue

                speech._authorize_generation()

                # For non-interrupted speeches, simulate generation completing
                if not speech.interrupted:
                    speech._mark_generation_done()

                # This is where the hang occurred
                await speech._wait_for_generation()

                processed_speeches.append(speech.id)

        try:
            await asyncio.wait_for(scheduling_task(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("scheduling_task simulation hung")

        # All speeches should have been processed without hanging
        assert len(processed_speeches) == 3


class TestSpeechHandleInterrupt:
    """Tests for SpeechHandle interrupt behavior."""

    @pytest.mark.asyncio
    async def test_interrupt_sets_interrupted_flag(self) -> None:
        """Test that interrupt() sets the interrupted property."""
        speech = SpeechHandle.create()

        assert not speech.interrupted
        speech.interrupt()
        assert speech.interrupted

    @pytest.mark.asyncio
    async def test_interrupt_disallowed_by_default(self) -> None:
        """Test that interrupt fails when allow_interruptions is False."""
        speech = SpeechHandle.create(allow_interruptions=False)

        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            speech.interrupt()

    @pytest.mark.asyncio
    async def test_force_interrupt(self) -> None:
        """Test that force=True overrides allow_interruptions."""
        speech = SpeechHandle.create(allow_interruptions=False)

        speech.interrupt(force=True)
        assert speech.interrupted

    @pytest.mark.asyncio
    async def test_wait_if_not_interrupted(self) -> None:
        """Test wait_if_not_interrupted returns when interrupted."""
        speech = SpeechHandle.create()

        never_done: asyncio.Future[None] = asyncio.Future()

        # Interrupt after a delay
        async def interrupt_later():
            await asyncio.sleep(0.1)
            speech.interrupt()

        asyncio.create_task(interrupt_later())

        # Should return when interrupted, not hang forever
        try:
            await asyncio.wait_for(
                speech.wait_if_not_interrupted([never_done]),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("wait_if_not_interrupted hung despite interrupt")

        assert speech.interrupted
