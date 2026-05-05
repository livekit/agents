"""Tests for the AMD classifier silence-timer state machine.

Focuses on the trigger-tagged silence timer logic: pre-baked HUMAN timers for
short greetings can be cancelled and replaced when a transcript arrives, while
long-speech timers (and postpone-termination timers) are left alone.
"""

from __future__ import annotations

import asyncio
import time

from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.amd.classifier import (
    AMDCategory,
    AMDPredictionEvent,
    _AMDClassifier,
)

from .fake_llm import FakeLLM, FakeLLMResponse


def _make_classifier(
    llm: FakeLLM | None = None,
    *,
    human_speech_threshold: float = 2.5,
    human_silence_threshold: float = 0.1,
    machine_silence_threshold: float = 0.3,
    no_speech_threshold: float = 10.0,
    timeout: float = 10.0,
) -> _AMDClassifier:
    return _AMDClassifier(
        llm or FakeLLM(),
        human_speech_threshold=human_speech_threshold,
        human_silence_threshold=human_silence_threshold,
        machine_silence_threshold=machine_silence_threshold,
        no_speech_threshold=no_speech_threshold,
        timeout=timeout,
    )


class TestAMDClassifier:
    """Tests for ``_AMDClassifier`` silence-timer behaviour."""

    async def test_short_greeting_no_transcript_emits_pre_baked_human(self) -> None:
        """Short utterance + no STT text => HUMAN/short_greeting verdict."""
        clf = _make_classifier(human_silence_threshold=0.1)
        clf.start()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer_trigger == "short_speech"
        assert clf._silence_timer is not None

        await asyncio.sleep(0.2)

        assert len(results) == 1
        assert results[0].category == AMDCategory.HUMAN
        assert results[0].reason == "short_greeting"
        assert clf._silence_timer is None
        assert clf._silence_timer_trigger is None
        assert clf._machine_silence_reached is True

        await clf.close()

    async def test_push_text_cancels_pre_baked_human_and_flips_trigger(self) -> None:
        """A transcript arriving during the short_speech window must cancel the
        pre-baked HUMAN timer and replace it with a long_speech timer anchored at
        speech_ended + machine_silence_threshold."""
        clf = _make_classifier(human_silence_threshold=0.1, machine_silence_threshold=0.3)
        clf.start()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer_trigger == "short_speech"

        clf.push_text("hello")
        assert clf._silence_timer_trigger == "long_speech"
        assert clf._silence_timer is not None

        # Past the would-be HUMAN deadline (0.1s), well before machine deadline (0.3s).
        await asyncio.sleep(0.18)
        assert results == [], "pre-baked HUMAN must not fire after a transcript arrives"
        assert clf._machine_silence_reached is False

        # Past the machine_silence deadline.
        await asyncio.sleep(0.2)
        assert clf._machine_silence_reached is True
        # No verdict was provided by the (empty) FakeLLM, so nothing emits yet.
        assert results == []

        await clf.close()

    async def test_push_text_replacement_timer_preserves_original_deadline(self) -> None:
        """The replacement timer fires near speech_ended + machine_silence_threshold,
        not push_text + machine_silence_threshold."""
        clf = _make_classifier(human_silence_threshold=0.05, machine_silence_threshold=0.3)
        clf.start()

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.on_user_speech_ended(silence_duration=0.0)
        t_end = clf._speech_ended_at
        assert t_end is not None

        push_delay = 0.04  # under human_silence_threshold so trigger is still short_speech
        await asyncio.sleep(push_delay)
        clf.push_text("hello")
        assert clf._silence_timer_trigger == "long_speech"

        expected_fire = t_end + 0.3
        deadline = expected_fire + 0.3
        while not clf._machine_silence_reached and time.time() < deadline:
            await asyncio.sleep(0.01)

        fired_at = time.time()
        assert clf._machine_silence_reached
        # Allow generous slack for event-loop jitter; the key assertion is that the
        # fire time is ~0.3s after t_end, not ~0.34s (which would mean we
        # re-armed for a full machine_silence_threshold from push_text).
        assert fired_at - t_end < 0.3 + 0.15, (
            f"timer fired at {fired_at - t_end:.3f}s after t_end; "
            f"expected ~0.30s, never ~0.34s+ (push_text-anchored)"
        )

        await clf.close()

    async def test_long_speech_push_text_does_not_replace_timer(self) -> None:
        """During the long_speech timer, push_text must leave the existing timer
        handle intact so the original 1.5s machine deadline is not extended."""
        clf = _make_classifier(
            human_speech_threshold=0.1,
            machine_silence_threshold=0.3,
        )
        clf.start()

        clf.on_user_speech_started()
        await asyncio.sleep(0.15)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer_trigger == "long_speech"
        handle_before = clf._silence_timer
        assert handle_before is not None

        clf.push_text("hello world")
        assert clf._silence_timer_trigger == "long_speech"
        assert clf._silence_timer is handle_before

        await clf.close()

    async def test_short_greeting_with_existing_transcript_uses_long_speech_trigger(
        self,
    ) -> None:
        """If a transcript is already present when speech ends (push_text before
        on_user_speech_ended), the short branch picks the long_speech trigger."""
        clf = _make_classifier(human_silence_threshold=0.1, machine_silence_threshold=0.3)
        clf.start()

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.push_text("hi")
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer_trigger == "long_speech"
        handle_before = clf._silence_timer
        assert handle_before is not None

        # A second transcript while in the long_speech window must not replace the timer.
        clf.push_text("there")
        assert clf._silence_timer is handle_before
        assert clf._silence_timer_trigger == "long_speech"

        await clf.close()

    async def test_on_user_speech_started_clears_trigger(self) -> None:
        """on_user_speech_started cancels the silence timer and nulls the trigger."""
        clf = _make_classifier(human_silence_threshold=1.0)
        clf.start()

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer is not None
        assert clf._silence_timer_trigger == "short_speech"

        clf.on_user_speech_started()
        assert clf._silence_timer is None
        assert clf._silence_timer_trigger is None

        await clf.close()

    async def test_silence_callback_clears_trigger_on_fire(self) -> None:
        """When the silence timer fires, both handle and trigger are nulled out."""
        clf = _make_classifier(human_silence_threshold=0.05)
        clf.start()

        clf.on_user_speech_started()
        await asyncio.sleep(0.02)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._silence_timer_trigger == "short_speech"

        await asyncio.sleep(0.12)

        assert clf._silence_timer is None
        assert clf._silence_timer_trigger is None

        await clf.close()

    async def test_short_greeting_transcript_emits_llm_verdict(self) -> None:
        """End-to-end: short greeting + transcript => LLM verdict emits at the
        machine_silence deadline (gated on both verdict and machine_silence_reached)."""
        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="hello",
                    content="",
                    ttft=0.0,
                    duration=0.05,
                    tool_calls=[
                        FunctionToolCall(
                            name="save_prediction",
                            arguments='{"label": "human"}',
                            call_id="c1",
                        )
                    ],
                )
            ]
        )
        clf = _make_classifier(llm=llm, human_silence_threshold=0.1, machine_silence_threshold=0.3)
        clf.start()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.05)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("hello")

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=2.0)

        assert len(results) == 1
        assert results[0].category == AMDCategory.HUMAN
        assert results[0].reason == "llm"
        assert results[0].transcript == "hello"

        await clf.close()
