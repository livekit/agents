"""Tests for the AMD detector lifecycle: when the detection budget is armed.

The budget bounds two different things at two different times:

- before any audio is published it bounds the wait for a publication, so AMD
  settles instead of hanging when a call never connects;
- once AMD starts listening it is the detection window for the greeting.

It must not run in between. For an outbound SIP call the first audio is usually
carrier early media that arrives while the phone is still ringing, so a budget
anchored at the publication expires before the callee ever answers.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.amd import detector as amd_detector
from livekit.agents.voice.amd.classifier import AMDCategory
from livekit.agents.voice.amd.detector import AMD

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

DETECTION_TIMEOUT = 20.0
NO_SPEECH_TIMEOUT = 10.0
HUMAN_SILENCE = 0.5


class _Clock:
    """Virtual seconds since the lifecycle started."""

    def __init__(self) -> None:
        self._t0 = time.time()

    def now(self) -> float:
        return round(time.time() - self._t0, 3)

    async def until(self, t: float) -> None:
        delay = t - self.now()
        if delay > 0:
            await asyncio.sleep(delay)


class _Activity:
    max_endpointing_delay = 3.0

    def _pause_authorization(self) -> None: ...

    def _resume_authorization(self) -> None: ...


class _Call:
    """One scripted call: the publication and the answer are driven by the test."""

    def __init__(self, *, sip: bool = True, identity: str = "callee") -> None:
        loop = asyncio.get_running_loop()
        self.published: asyncio.Future[object] = loop.create_future()
        self.answered: asyncio.Future[None] = loop.create_future()
        self.publisher = SimpleNamespace(
            kind=(
                rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                if sip
                else rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
            ),
            identity=identity,
            track_publications={f"{identity}_audio": object()},
        )
        self.room = SimpleNamespace(remote_participants={identity: self.publisher})
        self.session = SimpleNamespace(
            llm=FakeLLM(),
            options=SimpleNamespace(ivr_detection=False),
            _activity=_Activity(),
            _ivr_activity=None,
            _room_io=SimpleNamespace(room=self.room),
            _root_span_context=None,
            _session_host=None,
            _amd=None,
        )

    def publish_audio(self) -> None:
        """Carrier audio arrives (early media during ring, or the answered call)."""
        if not self.published.done():
            self.published.set_result(SimpleNamespace(sid=f"{self.publisher.identity}_audio"))

    def answer(self) -> None:
        """``sip.callStatus`` flips to ``active``."""
        if not self.answered.done():
            self.answered.set_result(None)

    def hangup_before_answer(self) -> None:
        if not self.answered.done():
            self.answered.set_exception(RuntimeError("participant disconnected"))


def _install(monkeypatch: pytest.MonkeyPatch, call: _Call) -> None:
    async def fake_wait_for_track_publication(**_: object) -> object:
        return await call.published

    async def fake_wait_for_participant_attribute(*_: object, **__: object) -> None:
        await call.answered

    monkeypatch.setattr(amd_detector, "wait_for_track_publication", fake_wait_for_track_publication)
    monkeypatch.setattr(
        amd_detector, "wait_for_participant_attribute", fake_wait_for_participant_attribute
    )


def _amd(call: _Call) -> AMD:
    detector = AMD(
        call.session,  # type: ignore[arg-type]
        llm=call.session.llm,
        participant_identity=call.publisher.identity,
        suppress_compatibility_warning=True,
    )
    detector._stt = NOT_GIVEN
    return detector


def _budget_armed(detector: AMD) -> bool:
    assert detector._classifier is not None
    return detector._classifier._detection_timeout_timer is not None


async def _short_greeting(detector: AMD, clock: _Clock, *, at: float) -> None:
    """A 0.5s greeting with no transcript, which pre-bakes a human verdict."""
    await clock.until(at)
    detector._on_user_speech_started()
    await clock.until(at + 0.5)
    detector._on_user_speech_ended(0.0)


class TestAMDDetectionBudget:
    async def test_budget_bounds_the_wait_for_a_publication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A call that never publishes audio must settle, not hang."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )
            await asyncio.sleep(0)
            assert _budget_armed(detector), "budget must bound the publication wait"

            await clock.until(DETECTION_TIMEOUT + 5.0)

        assert verdicts == [(DETECTION_TIMEOUT, AMDCategory.UNCERTAIN, "detection_timeout")]

    async def test_early_media_does_not_start_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ringing case from #6187: audio at t=1, answer well past the budget.

        The publication is carrier early media, so it must not anchor the budget
        and no verdict may be emitted while the phone is still ringing.
        """
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await asyncio.sleep(0)
            assert not _budget_armed(detector), "early media must not arm the budget"

            await clock.until(DETECTION_TIMEOUT + 12.0)
            assert verdicts == [], "no verdict may be reached before the call is answered"
            assert detector._classifier is not None
            assert not detector._classifier.listening

            call.answer()
            await asyncio.sleep(0)
            assert _budget_armed(detector), "the budget starts when the call is answered"

            await _short_greeting(detector, clock, at=DETECTION_TIMEOUT + 12.5)
            await clock.until(DETECTION_TIMEOUT + 20.0)

        assert verdicts == [
            (DETECTION_TIMEOUT + 13.5, AMDCategory.HUMAN, "short_greeting"),
        ]

    async def test_budget_runs_from_the_answer_not_from_the_publication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent answered call times out one full budget after the answer."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await clock.until(15.0)
            call.answer()

            await clock.until(15.0 + DETECTION_TIMEOUT + 5.0)

        assert verdicts == [(15.0 + NO_SPEECH_TIMEOUT, AMDCategory.UNCERTAIN, "no_speech_timeout")]

    async def test_ordinary_sip_call_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Answered quickly: same verdict and timing as before the change."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await clock.until(2.0)
            call.answer()
            await _short_greeting(detector, clock, at=2.5)
            await clock.until(10.0)

        assert verdicts == [(3.0 + HUMAN_SILENCE, AMDCategory.HUMAN, "short_greeting")]

    async def test_non_sip_participant_listens_at_track_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-SIP publisher has no answer signal: listening starts at track-up."""
        call = _Call(sip=False, identity="peer")
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await asyncio.sleep(0)
            assert detector._classifier is not None
            assert detector._classifier.listening
            assert _budget_armed(detector)

            await _short_greeting(detector, clock, at=1.5)
            await clock.until(10.0)

        assert verdicts == [(2.0 + HUMAN_SILENCE, AMDCategory.HUMAN, "short_greeting")]

    async def test_sip_disconnect_before_answer_still_settles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A callee that hangs up while ringing settles on the no-speech timer."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await clock.until(5.0)
            call.hangup_before_answer()

            await clock.until(5.0 + NO_SPEECH_TIMEOUT + 5.0)

        assert verdicts == [(5.0 + NO_SPEECH_TIMEOUT, AMDCategory.UNCERTAIN, "no_speech_timeout")]

    async def test_two_successive_lifecycles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second AMD lifecycle settles exactly like the first."""
        clock = _Clock()
        results: list[list[tuple[float, AMDCategory, str]]] = []

        for index in range(2):
            call = _Call()
            _install(monkeypatch, call)
            verdicts: list[tuple[float, AMDCategory, str]] = []
            base = index * 20.0

            async with _amd(call) as detector:
                detector.on(
                    "amd_prediction",
                    lambda ev, v=verdicts: v.append((clock.now(), ev.category, ev.reason)),
                )

                await clock.until(base + 1.0)
                call.publish_audio()
                await clock.until(base + 2.0)
                call.answer()
                await _short_greeting(detector, clock, at=base + 2.5)
                await clock.until(base + 10.0)

            results.append(verdicts)

        assert results == [
            [(3.5, AMDCategory.HUMAN, "short_greeting")],
            [(23.5, AMDCategory.HUMAN, "short_greeting")],
        ]

    async def test_unanswered_call_is_bounded_by_the_caller_dial_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's dial request is what bounds a call that is never answered."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        dial_timeout = 32.0

        async def dial() -> None:
            await clock.until(dial_timeout)
            raise TimeoutError("create_sip_participant timed out waiting for an answer")

        execute_task: asyncio.Task[object] | None = None
        with pytest.raises(TimeoutError):
            async with _amd(call) as detector:
                execute_task = asyncio.create_task(detector.execute())
                await clock.until(1.0)
                call.publish_audio()
                await dial()

        assert execute_task is not None
        with pytest.raises(RuntimeError, match="amd closed before a result was available"):
            await asyncio.wait_for(execute_task, timeout=5.0)
        assert clock.now() == pytest.approx(dial_timeout, abs=0.01)

    async def test_unanswered_call_has_no_amd_side_bound_after_early_media(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the documented caller bound, AMD itself waits for the answer."""
        call = _Call()
        _install(monkeypatch, call)
        clock = _Clock()
        verdicts: list[tuple[float, AMDCategory, str]] = []

        async with _amd(call) as detector:
            detector.on(
                "amd_prediction",
                lambda ev: verdicts.append((clock.now(), ev.category, ev.reason)),
            )

            await clock.until(1.0)
            call.publish_audio()
            await clock.until(10 * DETECTION_TIMEOUT)

            assert verdicts == []
            assert not _budget_armed(detector)
