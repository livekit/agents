from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from livekit import rtc
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.amd import detector as amd_detector
from livekit.agents.voice.amd.classifier import (
    AMDCategory,
    AMDPredictionEvent,
    _AMDClassifier,
)
from livekit.agents.voice.amd.detector import AMD

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


def _human_response(transcript: str = "hello") -> FakeLLMResponse:
    return FakeLLMResponse(
        input=transcript,
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


class TestAMDClassifier:
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
        """A transcript during the short-speech window swaps the human timer for a long-speech one."""
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
        """The replacement timer keeps the deadline anchored at speech end."""
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
        """A transcript during the long-speech timer does not extend its deadline."""
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
        """A transcript present at speech end picks the long-speech trigger."""
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
        """A short greeting with a transcript emits the LLM verdict at the machine-silence deadline."""
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
        """An early verdict leaves the detection and no-speech timers armed."""
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

    async def test_start_detection_timer_resets_existing_timer(self) -> None:
        clf = _make_classifier(timeout=0.4)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.start_detection_timer()
        first_timer = clf._detection_timeout_timer
        assert first_timer is not None

        await asyncio.sleep(0.25)
        clf.start_detection_timer()
        assert clf._detection_timeout_timer is not None
        assert clf._detection_timeout_timer is not first_timer

        await asyncio.sleep(0.25)
        assert results == []

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert results[0].reason == "detection_timeout"

        await clf.close()

    async def test_settled_classifier_rejects_new_timers_and_listening(self) -> None:
        clf = _make_classifier(no_speech_threshold=0.2)
        clf.start_listening()

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert clf._emitted is True
        assert clf._detection_timeout_timer is None
        assert clf.listening is False

        clf.start_detection_timer()
        clf.start_listening()

        assert clf._detection_timeout_timer is None
        assert clf.listening is False

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
        """With wait_until_finished and speech heard, the detection timeout waits for end of turn."""
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
        """The no-speech timeout emits even with wait_until_finished."""
        clf = _make_classifier(no_speech_threshold=0.2, wait_until_finished=True)
        clf.start_listening()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert len(results) == 1
        assert results[0].category == AMDCategory.UNCERTAIN
        assert results[0].reason == "no_speech_timeout"

        await clf.close()

    async def test_max_endpointing_delay_emits_machine_without_turn_detector(self) -> None:
        """The max-endpointing end of turn lets a machine verdict emit without on_end_of_turn()."""
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

        # the max endpointing delay (0.4) fires without any on_end_of_turn() call → emit
        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)
        assert clf._eot_reached is True
        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_max_endpointing_delay_accounts_for_elapsed_silence_on_eos(self) -> None:
        clf = _make_classifier(human_speech_threshold=0.05, max_endpointing_delay=0.4)
        clf.start_listening()

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.25)

        await asyncio.sleep(0.2)

        assert clf._eot_reached is True
        assert clf._eot_timer is None

        await clf.close()

    async def test_detection_timeout_emits_uncertain_when_eot_reached_no_verdict(self) -> None:
        """Without a verdict after end of turn, the detection timeout emits uncertain."""
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

    async def test_detection_timeout_preserves_transcript(self) -> None:
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
                            arguments='{"label": "uncertain"}',
                            call_id="c1",
                        )
                    ],
                )
            ]
        )
        clf = _make_classifier(
            llm=llm,
            timeout=0.4,
            wait_until_finished=True,
            max_endpointing_delay=0.2,
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.push_text("hello")

        await asyncio.wait_for(clf._verdict_ready.wait(), timeout=1.0)

        assert len(results) == 1
        assert results[0].category == AMDCategory.UNCERTAIN
        assert results[0].reason == "detection_timeout"
        assert results[0].transcript == "hello"

        await clf.close()

    async def test_speech_restart_cancels_eot_backstop(self) -> None:
        """on_user_speech_started cancels the EOT timer and resets the gate."""
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

    async def test_max_endpointing_delay_emits_human_without_vad_boundaries(self) -> None:
        """A final transcript can complete AMD without VAD boundaries."""
        clf = _make_classifier(
            llm=FakeLLM(fake_responses=[_human_response()]),
            timeout=5.0,
            max_endpointing_delay=0.3,
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.push_text("hello")
        assert clf._eot_timer is not None

        await asyncio.sleep(0.4)

        assert clf._eot_reached is True
        assert clf._eot_timer is None
        assert len(results) == 1
        assert results[0].category == AMDCategory.HUMAN
        assert results[0].reason == "llm"

        await clf.close()

    async def test_max_endpointing_delay_emits_machine_without_vad_boundaries(self) -> None:
        clf = _make_classifier(
            llm=FakeLLM(fake_responses=[_machine_vm_response()]),
            timeout=5.0,
            max_endpointing_delay=0.3,
        )
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.push_text("voicemail greeting")

        await asyncio.sleep(0.2)
        assert clf._verdict_result is not None
        assert results == []

        await asyncio.sleep(0.2)

        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_transcript_eot_waits_for_latest_chunk(self) -> None:
        llm = FakeLLM(
            fake_responses=[
                _human_response("hello"),
                _machine_vm_response("hello you've reached"),
            ]
        )
        clf = _make_classifier(llm=llm, timeout=5.0, max_endpointing_delay=0.3)
        clf.start_listening()
        clf.start_detection_timer()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.push_text("hello")
        await asyncio.sleep(0.2)
        assert clf._verdict_result is not None
        assert clf._verdict_result.category == AMDCategory.HUMAN
        assert results == []

        clf.push_text("you've reached")
        await asyncio.sleep(0.2)
        assert clf._verdict_result is not None
        assert clf._verdict_result.category == AMDCategory.MACHINE_VM
        assert results == []

        await asyncio.sleep(0.2)

        assert len(results) == 1
        assert results[0].category == AMDCategory.MACHINE_VM

        await clf.close()

    async def test_transcript_rearms_max_endpointing_delay_after_eos(self) -> None:
        """A post-EOS transcript re-arms the synthetic EOT timer."""
        clf = _make_classifier(human_speech_threshold=0.05, max_endpointing_delay=0.4)
        clf.start_listening()

        clf.on_user_speech_started()
        await asyncio.sleep(0.1)
        clf.on_user_speech_ended(silence_duration=0.0)
        first_timer = clf._eot_timer
        assert first_timer is not None

        await asyncio.sleep(0.25)
        clf.push_text("voicemail greeting")
        assert clf._eot_timer is not None
        assert clf._eot_timer is not first_timer

        await asyncio.sleep(0.25)
        assert clf._eot_reached is False

        await asyncio.sleep(0.25)
        assert clf._eot_reached is True

        await clf.close()

    async def test_settle_emits_uncertain_before_listening(self) -> None:
        clf = _make_classifier()
        results: list[AMDPredictionEvent] = []
        clf.on("amd_prediction", results.append)

        clf.settle(AMDCategory.UNCERTAIN, reason="participant_missing")

        assert len(results) == 1
        assert results[0].category == AMDCategory.UNCERTAIN
        assert results[0].reason == "participant_missing"
        assert clf._verdict_ready.is_set()

        await clf.close()


def _configure_cloud_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_detector, "is_cloud", lambda _: True)
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-that-is-at-least-32-bytes")


class _RecordingClassifier:
    def __init__(self) -> None:
        self.timer_calls = 0
        self.listening = False
        self.settled: list[tuple[AMDCategory, str]] = []

    def start_detection_timer(self) -> None:
        self.timer_calls += 1

    def start_listening(self) -> None:
        self.listening = True

    def settle(self, category: AMDCategory, reason: str) -> None:
        self.settled.append((category, reason))


class _FakeRoom:
    def __init__(self, remote_participants: dict[str, object]) -> None:
        self.remote_participants = remote_participants
        self._handlers: dict[str, list[object]] = {}

    def isconnected(self) -> bool:
        return True

    def on(self, event: str, callback: object) -> None:
        self._handlers.setdefault(event, []).append(callback)

    def off(self, event: str, callback: object) -> None:
        self._handlers.get(event, []).remove(callback)

    def emit(self, event: str, *args: object) -> None:
        for callback in list(self._handlers.get(event, [])):
            callback(*args)  # type: ignore[operator]


def _llm_verdict(category: AMDCategory) -> AMDPredictionEvent:
    return AMDPredictionEvent(
        speech_duration=3.0,
        category=category,
        reason="llm",
        transcript="greeting",
        delay=0.0,
    )


class _StartedAMD(NamedTuple):
    detector: AMD
    classifier: _AMDClassifier
    room: _FakeRoom
    publisher: SimpleNamespace
    # the session actions ``execute()`` takes on a verdict
    session_calls: list[str]


async def _start_detector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: object = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
) -> _StartedAMD:
    """Run AMD setup against a fake room, up to the point where a hangup can arrive."""

    async def fake_wait_for_track_publication(**_: object) -> SimpleNamespace:
        return SimpleNamespace(sid="track_sid")

    monkeypatch.setattr(
        amd_detector,
        "wait_for_track_publication",
        fake_wait_for_track_publication,
    )

    calls: list[str] = []

    async def interrupt(*, force: bool = False) -> None:
        calls.append("interrupt")

    async def start_ivr_detection(*, transcript: str | None = None) -> None:
        calls.append("start_ivr_detection")

    llm = FakeLLM()
    publisher = SimpleNamespace(
        kind=kind,
        identity="callee",
        attributes={},
        track_publications={"track_sid": object()},
    )
    room = _FakeRoom({"callee": publisher})
    session = SimpleNamespace(
        llm=llm,
        _activity=None,
        _session_host=None,
        _amd=None,
        _room_io=SimpleNamespace(room=room),
        interrupt=interrupt,
        _start_ivr_detection=start_ivr_detection,
    )
    detector = AMD(
        session,  # type: ignore[arg-type]
        llm=llm,
        suppress_compatibility_warning=True,
    )
    detector._stt = None
    classifier = detector._resolve_classifier(session)  # type: ignore[arg-type]
    assert classifier is not None
    classifier.on("amd_prediction", detector._on_amd_prediction)
    detector._classifier = classifier

    await detector._setup(session)  # type: ignore[arg-type]
    return _StartedAMD(detector, classifier, room, publisher, calls)


class TestAMDDetector:
    @pytest.mark.parametrize(
        ("detection_options", "activity_delay", "expected"),
        [
            ({"max_endpointing_delay": 0.25}, 9.0, 0.25),
            ({}, 1.25, 1.25),
            ({}, None, None),
        ],
    )
    def test_max_endpointing_delay_resolution(
        self, detection_options: dict, activity_delay: float | None, expected: float | None
    ) -> None:
        llm = FakeLLM()
        activity = SimpleNamespace(max_endpointing_delay=activity_delay) if activity_delay else None
        session = SimpleNamespace(llm=llm, _activity=activity)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            detection_options=detection_options,  # type: ignore[arg-type]
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        # with no override and no activity, the detector's own default applies
        assert clf._max_endpointing_delay == (
            expected if expected is not None else detector._opts["max_endpointing_delay"]
        )

    @pytest.mark.parametrize("wait_until_finished", [None, False])
    def test_wait_until_finished_defaults_on(self, wait_until_finished: bool | None) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        kwargs = {} if wait_until_finished is None else {"wait_until_finished": False}
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
            **kwargs,  # type: ignore[arg-type]
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._wait_until_finished is (wait_until_finished is None)

    def test_none_reuses_session_models_on_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure_cloud_inference(monkeypatch)
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=None,
            stt=None,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert detector._llm_config is None
        assert detector._stt is None
        assert clf is not None
        assert clf._llm is llm
        assert clf._source == "stt"

    def test_omitted_models_inherit_session_without_cloud_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(amd_detector, "is_cloud", lambda _: False)
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert detector._llm_config is None
        assert detector._stt is None
        assert clf is not None
        assert clf._llm is llm
        assert clf._source == "stt"

    def test_omitted_models_auto_select_cloud_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_cloud_inference(monkeypatch)
        session = SimpleNamespace(llm=FakeLLM(), _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            suppress_compatibility_warning=True,
        )

        assert detector._llm_config == detector._DEFAULT_LLM_MODEL
        assert detector._stt is not None
        assert detector._stt.model == detector._DEFAULT_STT_MODEL

    def test_none_llm_inherits_session_and_omitted_stt_uses_cloud_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_cloud_inference(monkeypatch)
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=None,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert detector._llm_config is None
        assert detector._stt is not None
        assert detector._stt.model == detector._DEFAULT_STT_MODEL
        assert clf is not None
        assert clf._llm is llm
        assert clf._source == "amd_stt"

    def test_omitted_llm_uses_cloud_default_and_none_stt_inherits_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_cloud_inference(monkeypatch)
        session = SimpleNamespace(llm=FakeLLM(), _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            stt=None,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert detector._llm_config == detector._DEFAULT_LLM_MODEL
        assert detector._stt is None
        assert clf is not None
        assert clf._llm.model == detector._DEFAULT_LLM_MODEL
        assert clf._source == "stt"

    def test_explicit_models_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure_cloud_inference(monkeypatch)
        llm = FakeLLM()
        session = SimpleNamespace(llm=FakeLLM(), _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            stt="deepgram/nova-3",
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert detector._llm_config is llm
        assert detector._stt is not None
        assert detector._stt.model == "deepgram/nova-3"
        assert clf is not None
        assert clf._llm is llm
        assert clf._source == "amd_stt"

    async def test_setup_arms_detection_timer_only_at_listening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        classifier = _RecordingClassifier()

        async def fake_wait_for_track_publication(**_: object) -> SimpleNamespace:
            assert classifier.timer_calls == 0, (
                "no detection timer must be armed while waiting for the track"
            )
            return SimpleNamespace(sid="track_sid")

        monkeypatch.setattr(
            amd_detector,
            "wait_for_track_publication",
            fake_wait_for_track_publication,
        )

        llm = FakeLLM()
        publisher = SimpleNamespace(
            kind=object(),
            identity="callee",
            track_publications={"track_sid": object()},
        )
        session = SimpleNamespace(
            llm=llm,
            _activity=None,
            _room_io=SimpleNamespace(room=_FakeRoom({"callee": publisher})),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )
        detector._stt = None
        detector._classifier = classifier  # type: ignore[assignment]

        await detector._setup(session)  # type: ignore[arg-type]

        assert classifier.timer_calls == 1
        assert classifier.listening is True
        assert classifier.settled == []

    async def test_execute_settles_when_participant_hangs_up_while_listening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        amd = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(amd.detector.execute())

        amd.classifier.on_user_speech_started()
        await asyncio.sleep(0)
        amd.room.emit("participant_disconnected", amd.publisher)
        # session teardown closes AMD right after the disconnect
        await amd.detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_disconnected"
        assert amd.room._handlers["participant_disconnected"] == []

    @pytest.mark.parametrize(
        "held",
        [
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_IVR,
            AMDCategory.UNCERTAIN,
        ],
    )
    async def test_hangup_overrides_held_verdict(
        self, monkeypatch: pytest.MonkeyPatch, held: AMDCategory
    ) -> None:
        amd = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(amd.detector.execute())

        # the verdict is stored mid-greeting, so the silence gate still holds it
        amd.classifier.on_user_speech_started()
        amd.classifier._set_verdict(_llm_verdict(held))
        await asyncio.sleep(0)
        assert amd.detector.pending
        amd.room.emit("participant_disconnected", amd.publisher)
        await amd.detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_disconnected"
        # nobody is left to interrupt for, or to navigate an IVR with
        assert amd.session_calls == []

    async def test_hangup_keeps_held_machine_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        amd = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(amd.detector.execute())

        amd.classifier.on_user_speech_started()
        amd.classifier._set_verdict(_llm_verdict(AMDCategory.MACHINE_UNAVAILABLE))
        await asyncio.sleep(0)
        assert amd.detector.pending
        amd.room.emit("participant_disconnected", amd.publisher)
        await amd.detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.MACHINE_UNAVAILABLE
        assert result.reason == "llm"

    async def test_hangup_after_verdict_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        amd = await _start_detector(monkeypatch)
        amd.classifier.settle(AMDCategory.HUMAN, reason="short_greeting")
        emitted = amd.detector._result
        assert emitted is not None

        amd.room.emit("participant_disconnected", amd.publisher)

        assert amd.detector._result is emitted
        await amd.detector.aclose()

    @pytest.mark.parametrize("yields_before_teardown", [0, 3])
    async def test_hangup_before_sip_answer_settles_participant_missing(
        self, monkeypatch: pytest.MonkeyPatch, yields_before_teardown: int
    ) -> None:
        amd = await _start_detector(monkeypatch, kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        settle = amd.classifier.settle
        settled: list[str] = []

        def recording_settle(category: AMDCategory, reason: str) -> None:
            settled.append(reason)
            settle(category, reason)

        monkeypatch.setattr(amd.classifier, "settle", recording_settle)
        execute_task = asyncio.create_task(amd.detector.execute())
        await asyncio.sleep(0)
        assert not amd.classifier.listening

        amd.room.emit("participant_disconnected", amd.publisher)
        # an immediate teardown cancels the SIP-answer wait before it can settle;
        # a later one lets that wait see the same hangup
        for _ in range(yields_before_teardown):
            await asyncio.sleep(0)
        await amd.detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_missing"
        assert settled == ["participant_missing"]

    async def test_other_participant_disconnect_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        amd = await _start_detector(monkeypatch)
        amd.classifier.on_user_speech_started()

        amd.room.emit("participant_disconnected", SimpleNamespace(identity="someone-else"))

        assert amd.detector.pending
        assert amd.classifier.listening
        await amd.detector.aclose()

    async def test_setup_settles_when_participant_disconnects_before_track(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        classifier = _RecordingClassifier()

        async def fake_wait_for_track_publication(**_: object) -> SimpleNamespace:
            raise RuntimeError("participant 'callee' disconnected while waiting")

        monkeypatch.setattr(
            amd_detector,
            "wait_for_track_publication",
            fake_wait_for_track_publication,
        )

        llm = FakeLLM()
        session = SimpleNamespace(
            llm=llm,
            _activity=None,
            _room_io=SimpleNamespace(room=SimpleNamespace(remote_participants={})),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )
        detector._stt = None
        detector._classifier = classifier  # type: ignore[assignment]

        await detector._setup(session)  # type: ignore[arg-type]

        assert classifier.settled == [(AMDCategory.UNCERTAIN, "participant_missing")]
        assert classifier.timer_calls == 0
        assert classifier.listening is False

    async def test_setup_settles_when_track_publication_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        classifier = _RecordingClassifier()

        async def fake_wait_for_track_publication(**_: object) -> SimpleNamespace:
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

        monkeypatch.setattr(
            amd_detector,
            "wait_for_track_publication",
            fake_wait_for_track_publication,
        )
        monkeypatch.setattr(amd_detector, "_TRACK_PUBLICATION_TIMEOUT", 0.1)

        llm = FakeLLM()
        session = SimpleNamespace(
            llm=llm,
            _activity=None,
            _room_io=SimpleNamespace(room=SimpleNamespace(remote_participants={})),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )
        detector._stt = None
        detector._classifier = classifier  # type: ignore[assignment]

        await detector._setup(session)  # type: ignore[arg-type]

        assert classifier.settled == [(AMDCategory.UNCERTAIN, "participant_missing")]
        assert classifier.timer_calls == 0
        assert classifier.listening is False

    async def test_setup_settles_when_participant_disappears_after_track(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        classifier = _RecordingClassifier()

        async def fake_wait_for_track_publication(**_: object) -> SimpleNamespace:
            return SimpleNamespace(sid="track_sid")

        monkeypatch.setattr(
            amd_detector,
            "wait_for_track_publication",
            fake_wait_for_track_publication,
        )

        llm = FakeLLM()
        session = SimpleNamespace(
            llm=llm,
            _activity=None,
            _room_io=SimpleNamespace(room=SimpleNamespace(remote_participants={})),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            participant_identity="callee",
            suppress_compatibility_warning=True,
        )
        detector._stt = None
        detector._classifier = classifier  # type: ignore[assignment]

        await detector._setup(session)  # type: ignore[arg-type]

        assert classifier.settled == [(AMDCategory.UNCERTAIN, "participant_missing")]
        assert classifier.timer_calls == 0
        assert classifier.listening is False

    async def test_sip_answer_failure_settles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        classifier = _RecordingClassifier()

        async def fake_wait_for_participant_attribute(*_: object, **__: object) -> None:
            raise RuntimeError("participant 'callee' disconnected while waiting for sip.callStatus")

        monkeypatch.setattr(
            amd_detector,
            "wait_for_participant_attribute",
            fake_wait_for_participant_attribute,
        )

        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )
        detector._classifier = classifier  # type: ignore[assignment]

        await detector._wait_for_sip_answer(SimpleNamespace(), "callee")  # type: ignore[arg-type]

        assert classifier.settled == [(AMDCategory.UNCERTAIN, "participant_missing")]
        assert classifier.timer_calls == 0
        assert classifier.listening is False
