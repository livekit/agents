"""Tests for the AMD classifier silence-timer state machine.

Focuses on the trigger-tagged silence timer logic: pre-baked HUMAN timers for
short greetings can be cancelled and replaced when a transcript arrives, while
long-speech timers (and postpone-termination timers) are left alone.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.amd.classifier import (
    AMDCategory,
    AMDPredictionEvent,
    _AMDClassifier,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _make_classifier(
    llm: FakeLLM | None = None,
    *,
    human_speech_threshold: float = 2.5,
    human_silence_threshold: float = 0.1,
    machine_silence_threshold: float = 0.3,
    no_speech_threshold: float = 10.0,
    timeout: float = 10.0,
    wait_until_finished: bool = False,
    max_endpointing_delay: float = 6.0,
    source: str = "stt",
) -> _AMDClassifier:
    return _AMDClassifier(
        llm or FakeLLM(),
        human_speech_threshold=human_speech_threshold,
        human_silence_threshold=human_silence_threshold,
        machine_silence_threshold=machine_silence_threshold,
        no_speech_threshold=no_speech_threshold,
        timeout=timeout,
        wait_until_finished=wait_until_finished,
        max_endpointing_delay=max_endpointing_delay,
        source=source,
    )


def _machine_vm_response(transcript: str = "voicemail greeting") -> FakeLLMResponse:
    return FakeLLMResponse(
        input=transcript,
        content="",
        ttft=0.0,
        duration=0.05,
        tool_calls=[
            FunctionToolCall(
                name="save_prediction",
                arguments='{"label": "machine-vm"}',
                call_id="c1",
            )
        ],
    )


class TestAMDClassifier:
    """Tests for ``_AMDClassifier`` silence-timer behaviour."""

    async def test_short_greeting_no_transcript_emits_pre_baked_human(self) -> None:
        """Short utterance + no STT text => HUMAN/short_greeting verdict."""
        clf = _make_classifier(human_silence_threshold=0.1)
        clf.start_listening()
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
        assert clf._silence_reached is True

        await clf.close()

    async def test_push_text_cancels_pre_baked_human_and_flips_trigger(self) -> None:
        """A transcript arriving during the short_speech window must cancel the
        pre-baked HUMAN timer and replace it with a long_speech timer anchored at
        speech_ended + machine_silence_threshold."""
        clf = _make_classifier(human_silence_threshold=0.1, machine_silence_threshold=0.3)
        clf.start_listening()
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
        assert clf._silence_reached is False

        # Past the machine_silence deadline.
        await asyncio.sleep(0.2)
        assert clf._silence_reached is True
        # No verdict was provided by the (empty) FakeLLM, so nothing emits yet.
        assert results == []

        await clf.close()

    async def test_push_text_replacement_timer_preserves_original_deadline(self) -> None:
        """The replacement timer fires near speech_ended + machine_silence_threshold,
        not push_text + machine_silence_threshold."""
        clf = _make_classifier(human_silence_threshold=0.05, machine_silence_threshold=0.3)
        clf.start_listening()

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
        while not clf._silence_reached and time.time() < deadline:
            await asyncio.sleep(0.01)

        fired_at = time.time()
        assert clf._silence_reached
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
        clf.start_listening()

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
        clf.start_listening()

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
        clf.start_listening()

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
        clf.start_listening()

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
        machine_silence deadline (gated on both verdict and silence_reached)."""
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
        clf.start_listening()
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

    async def test_machine_verdict_waits_for_eot(self) -> None:
        """Machine verdict is gated on BOTH silence_reached AND eot_reached."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail greeting")])
        clf = _make_classifier(llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)  # past human_speech_threshold → long_speech path
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail greeting")

        # silence timer fires (verdict already set by LLM); EOT has not.
        await asyncio.sleep(0.4)
        assert clf._silence_reached is True
        assert clf._eot_reached is False
        assert clf._verdict_result is not None
        assert clf._verdict_result.is_machine
        assert results == [], "machine verdict must wait for EOT"

        # EOT lands → emit.
        clf.on_end_of_turn()
        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_machine_verdict_eot_before_silence(self) -> None:
        """Order independence: EOT before silence still emits at silence."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail greeting")])
        clf = _make_classifier(llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail greeting")

        # EOT lands while silence timer is still running.
        await asyncio.sleep(0.05)
        clf.on_end_of_turn()
        assert clf._eot_reached is True
        assert clf._silence_reached is False
        assert results == [], "must still wait for silence timer"

        await asyncio.sleep(0.4)
        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_human_verdict_emits_without_eot(self) -> None:
        """Human/uncertain verdicts emit on silence alone (snappy)."""
        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="hello there",
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
        clf = _make_classifier(llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("hello there")

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=2.0)
        assert results[0].category == AMDCategory.HUMAN
        assert clf._eot_reached is False, "human must not require EOT"

        await clf.close()

    async def test_set_verdict_keeps_timers_armed(self) -> None:
        """Preemptive verdict must not cancel detection/no_speech timers; they
        still bound the overall AMD lifetime."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail")])
        clf = _make_classifier(
            llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3, timeout=5.0
        )
        clf.start_detection_timer()
        clf.start_listening()
        assert clf._detection_timeout_timer is not None
        assert clf._no_speech_timer is not None

        clf.on_user_speech_started()
        # speech-started already cancels no_speech_timer; we care about detection_timer
        assert clf._detection_timeout_timer is not None

        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail")
        # let LLM commit verdict
        await asyncio.sleep(0.2)
        assert clf._verdict_result is not None
        # detection_timeout must remain armed post-verdict
        assert clf._detection_timeout_timer is not None

        await clf.close()

    async def test_emit_cancels_timers(self) -> None:
        """Timers are cancelled at successful emission, not at verdict-set."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail")])
        clf = _make_classifier(
            llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3, timeout=5.0
        )
        clf.start_listening()
        clf.start_detection_timer()

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail")
        await asyncio.sleep(0.4)  # silence timer fires
        assert clf._detection_timeout_timer is not None  # still armed, EOT pending

        clf.on_end_of_turn()
        await asyncio.sleep(0)
        # emit happened → timer cancelled
        assert clf._detection_timeout_timer is None

        await clf.close()

    async def test_wait_until_finished_extends_detection_timeout(self) -> None:
        """With wait_until_finished=True and speech heard, detection_timeout
        does not force emission — AMD keeps waiting for EOT."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail")])
        clf = _make_classifier(
            llm=llm,
            human_speech_threshold=0.05,
            machine_silence_threshold=0.3,
            timeout=0.4,
            wait_until_finished=True,
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail")

        # well past detection_timeout — but speech was heard and EOT not yet
        await asyncio.sleep(0.7)
        assert results == [], "detection_timeout must not force emit when waiting"
        assert clf._verdict_result is not None
        assert clf._eot_reached is False

        clf.on_end_of_turn()
        assert len(results) == 1

        await clf.close()

    async def test_no_speech_timeout_always_forces_emit(self) -> None:
        """no_speech_timeout fires regardless of wait_until_finished — there
        is nothing to wait for if no audio was ever heard."""
        clf = _make_classifier(no_speech_threshold=0.2, wait_until_finished=True)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert len(results) == 1
        assert results[0].category == AMDCategory.UNCERTAIN
        assert results[0].reason == "no_speech_timeout"

        await clf.close()

    async def test_eot_backstop_emits_machine_without_turn_detector(self) -> None:
        """The synthetic end-of-turn backstop (max_endpointing_delay) lets a
        machine verdict emit even if on_end_of_turn() is never called."""
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail")])
        clf = _make_classifier(
            llm=llm,
            human_speech_threshold=0.05,
            machine_silence_threshold=0.2,
            timeout=5.0,
            wait_until_finished=True,
            max_endpointing_delay=0.4,
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail")

        # silence (0.2) elapses but eot has not yet → still gated
        await asyncio.sleep(0.3)
        assert clf._silence_reached is True
        assert clf._eot_reached is False
        assert results == []

        # the eot backstop (0.4) fires without any on_end_of_turn() call → emit
        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert clf._eot_reached is True
        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_detection_timeout_emits_uncertain_when_eot_reached_no_verdict(self) -> None:
        """wait_until_finished + speech but the LLM never commits a verdict:
        once eot is reached the greeting has ended, so the detection timeout
        must emit the uncertain fallback instead of deferring forever."""
        clf = _make_classifier(
            human_speech_threshold=0.05,
            machine_silence_threshold=0.2,
            timeout=0.4,
            wait_until_finished=True,
            max_endpointing_delay=6.0,  # backstop won't fire; eot comes from on_end_of_turn
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        # no transcript / no verdict produced; turn ends
        clf.on_end_of_turn()
        assert clf._eot_reached is True
        assert clf._verdict_result is None

        # detection timeout fires with eot already reached → emit uncertain fallback
        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert len(results) == 1
        assert results[0].category == AMDCategory.UNCERTAIN
        assert results[0].reason == "detection_timeout"

        await clf.close()

    async def test_speech_restart_cancels_eot_backstop(self) -> None:
        """on_user_speech_started cancels the eot backstop and resets the gate."""
        clf = _make_classifier(human_speech_threshold=0.05, max_endpointing_delay=0.3)
        clf.start_listening()

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        assert clf._eot_timer is not None

        # user speaks again before the backstop fires
        clf.on_user_speech_started()
        assert clf._eot_timer is None
        assert clf._eot_reached is False

        # well past the original backstop deadline → still not reached
        await asyncio.sleep(0.4)
        assert clf._eot_reached is False

        await clf.close()


class TestAMDClassifierReset:
    """reset() re-arms the classifier for the next internal screening turn."""

    async def test_reset_rearms_for_next_turn(self) -> None:
        llm = FakeLLM(
            fake_responses=[
                _machine_vm_response("voicemail greeting"),
                _machine_vm_response("second greeting"),
            ]
        )
        clf = _make_classifier(llm=llm, human_speech_threshold=0.05, machine_silence_threshold=0.3)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        # turn 1 → machine-vm
        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("voicemail greeting")
        await asyncio.sleep(0.4)
        clf.on_end_of_turn()
        assert len(results) == 1

        # reset re-arms a fresh turn
        await clf.reset()
        assert clf.listening is True
        assert clf._verdict_result is None
        assert clf._verdict_ready.is_set() is False
        assert clf._input_ch.closed is False
        assert clf._emitted is False
        assert clf._silence_reached is False
        assert clf._eot_reached is False
        assert clf._transcript == ""

        # turn 2 produces a second independent verdict
        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        clf.push_text("second greeting")
        await asyncio.sleep(0.4)
        clf.on_end_of_turn()
        assert len(results) == 2

        await clf.close()


class TestAMDClassifierSource:
    """Source filtering: race-based fallback and switch_source."""

    async def test_session_stt_wins_race_flips_source(self) -> None:
        """A session transcript before any amd_stt transcript flips the source one-way."""
        clf = _make_classifier(source="amd_stt")
        clf.start_listening()

        clf.push_text("hello", source="stt")
        assert clf._source == "stt"
        assert clf._transcript == "hello"

        await clf.close()

    async def test_amd_stt_first_keeps_source(self) -> None:
        """If amd_stt produced text first, a later session transcript is dropped."""
        clf = _make_classifier(source="amd_stt")
        clf.start_listening()

        clf.push_text("from amd", source="amd_stt")
        assert clf._source == "amd_stt"
        clf.push_text("from session", source="stt")
        assert clf._source == "amd_stt"
        assert "from session" not in clf._transcript

        await clf.close()

    async def test_switch_source_redirects_consumption(self) -> None:
        clf = _make_classifier(source="amd_stt")
        clf.start_listening()

        clf.switch_source("stt")
        clf.push_text("session text", source="stt")
        assert clf._transcript == "session text"
        clf.push_text("amd text", source="amd_stt")
        assert "amd text" not in clf._transcript

        await clf.close()


class TestAMDClassifierNoVAD:
    """Transcript-before-VAD synthesizes a quick utterance so the verdict emits promptly."""

    async def test_transcript_without_vad_emits_on_silence_timer(self) -> None:
        llm = FakeLLM(fake_responses=[_machine_vm_response("voicemail greeting")])
        clf = _make_classifier(
            llm=llm,
            machine_silence_threshold=0.3,
            timeout=10.0,
        )
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        # no on_user_speech_started — transcript arrives directly
        clf.push_text("voicemail greeting")
        assert clf._speech_started_at is not None  # synthesized
        assert clf._silence_timer is not None
        assert clf._eot_timer is not None

        # emits via the silence timer + eot backstop, well before the 10s detection timeout
        await asyncio.sleep(0.4)
        clf.on_end_of_turn()
        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_transcript_without_vad_human_uses_short_human_silence(self) -> None:
        """A VAD-missed utterance is treated as short: a human verdict releases on the
        human_silence window, not the longer machine_silence one."""
        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="hello",
                    content="",
                    ttft=0.0,
                    duration=0.02,
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
        # human_silence (0.1) << machine_silence (0.5): emitting before 0.5 proves the
        # short window is used.
        clf = _make_classifier(
            llm=llm, human_silence_threshold=0.1, machine_silence_threshold=0.5, timeout=10.0
        )
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.push_text("hello")  # no VAD event

        # past human_silence, before machine_silence — human releases on silence alone
        await asyncio.sleep(0.25)
        assert len(results) == 1
        assert results[0].category == AMDCategory.HUMAN

        await clf.close()
