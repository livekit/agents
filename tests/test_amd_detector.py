"""Tests for the AMD detector's internal screening loop and message playback.

Drives the detector's ``_detection_loop`` with a stub classifier + stub session so we can
assert the per-category routing (screening / ivr / terminal), the predefined-message
playout, and how ``screening_detected`` / ``message_playback`` land on the terminal verdict.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.amd.classifier import AMDCategory, AMDPredictionEvent
from livekit.agents.voice.amd.detector import AMD

pytestmark = [pytest.mark.unit]


class _FakeHandle:
    def __init__(self, *, interrupted: bool = False) -> None:
        self._interrupted = interrupted

    async def wait_for_playout(self) -> None:
        return None

    @property
    def interrupted(self) -> bool:
        return self._interrupted


class _FakeActivity:
    def __init__(self) -> None:
        self.resumed = 0
        self.paused = 0

    def _resume_authorization(self) -> None:
        self.resumed += 1

    def _pause_authorization(self) -> None:
        self.paused += 1


class _FakeSession:
    def __init__(self) -> None:
        self._activity = _FakeActivity()
        self._session_host = None
        self.said: list[str] = []
        self.interrupts = 0
        self.ivr_calls: list[str] = []
        self._next_handle = _FakeHandle()

    async def interrupt(self, *, force: bool = False) -> None:
        self.interrupts += 1

    def say(self, text: str, **kwargs: object) -> _FakeHandle:
        self.said.append(text)
        return self._next_handle

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        self.ivr_calls.append(transcript or "")


class _FakeClassifier:
    def __init__(self) -> None:
        self._verdict_ready = asyncio.Event()
        self._verdict_result: AMDPredictionEvent | None = None
        self.reset_count = 0
        self.ended = False

    async def reset(self) -> None:
        self.reset_count += 1
        self._verdict_result = None
        self._verdict_ready = asyncio.Event()

    def end_input(self) -> None:
        self.ended = True


def _verdict(category: AMDCategory, transcript: str = "greeting") -> AMDPredictionEvent:
    return AMDPredictionEvent(
        speech_duration=1.0,
        category=category,
        reason="test",
        transcript=transcript,
        delay=0.0,
    )


def _make_detector(
    session: _FakeSession,
    *,
    screening_message: object = "this is alex",
    voicemail_message: object = "leave a message",
) -> AMD:
    detector = AMD(
        session,  # type: ignore[arg-type]
        screening_message=screening_message,  # type: ignore[arg-type]
        voicemail_message=voicemail_message,  # type: ignore[arg-type]
        suppress_compatibility_warning=True,
    )
    return detector


async def _drive(
    detector: AMD, clf: _FakeClassifier, verdicts: list[AMDPredictionEvent]
) -> AMDPredictionEvent:
    detector._classifier = clf  # type: ignore[assignment]
    emitted: list[AMDPredictionEvent] = []
    detector.on("amd_prediction", emitted.append)

    loop = asyncio.create_task(detector._detection_loop())
    for v in verdicts:
        clf._verdict_result = v
        clf._verdict_ready.set()
        await asyncio.sleep(0.05)

    await asyncio.wait_for(detector._terminal_ready.wait(), timeout=1.0)
    await asyncio.wait_for(loop, timeout=1.0)
    assert len(emitted) == 1
    return emitted[0]


async def test_screening_then_human_reports_screening_playback() -> None:
    session = _FakeSession()
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(
        detector,
        clf,
        [_verdict(AMDCategory.MACHINE_SCREENING), _verdict(AMDCategory.HUMAN)],
    )

    assert session.said == ["this is alex"]  # only the screening message played
    assert clf.reset_count == 1  # reset once after screening
    assert result.category == AMDCategory.HUMAN
    assert result.screening_detected is True
    assert result.message_playback == "played"


async def test_screening_then_voicemail_plays_both() -> None:
    session = _FakeSession()
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(
        detector,
        clf,
        [_verdict(AMDCategory.MACHINE_SCREENING), _verdict(AMDCategory.MACHINE_VM)],
    )

    assert session.said == ["this is alex", "leave a message"]
    assert result.category == AMDCategory.MACHINE_VM
    assert result.screening_detected is True
    assert result.message_playback == "played"  # reflects the voicemail message


async def test_direct_human_plays_nothing() -> None:
    session = _FakeSession()
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(detector, clf, [_verdict(AMDCategory.HUMAN)])

    assert session.said == []
    assert result.screening_detected is False
    assert result.message_playback == "not_played"


async def test_direct_voicemail_plays_voicemail() -> None:
    session = _FakeSession()
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(detector, clf, [_verdict(AMDCategory.MACHINE_VM)])

    assert session.said == ["leave a message"]
    assert result.screening_detected is False
    assert result.message_playback == "played"


async def test_screening_message_interrupted_is_reported() -> None:
    session = _FakeSession()
    session._next_handle = _FakeHandle(interrupted=True)
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(
        detector,
        clf,
        [_verdict(AMDCategory.MACHINE_SCREENING), _verdict(AMDCategory.HUMAN)],
    )

    assert result.message_playback == "interrupted"


async def test_ivr_navigates_and_continues() -> None:
    session = _FakeSession()
    detector = _make_detector(session)
    clf = _FakeClassifier()

    result = await _drive(
        detector,
        clf,
        [_verdict(AMDCategory.MACHINE_IVR, "press 1"), _verdict(AMDCategory.HUMAN)],
    )

    assert session.ivr_calls == ["press 1"]
    assert session.said == []  # no predefined message for ivr
    assert result.category == AMDCategory.HUMAN


async def test_screening_without_message_is_terminal() -> None:
    """No screening_message → screening can't be advanced, so surface it as terminal."""
    session = _FakeSession()
    detector = _make_detector(session, screening_message=NOT_GIVEN)
    clf = _FakeClassifier()

    result = await _drive(detector, clf, [_verdict(AMDCategory.MACHINE_SCREENING)])

    assert session.said == []
    assert clf.reset_count == 0  # did not loop into another turn
    assert result.category == AMDCategory.MACHINE_SCREENING
    assert result.screening_detected is True
    assert result.message_playback == "not_played"


async def test_loop_failure_releases_execute() -> None:
    """An unexpected loop failure must still unblock execute() (no deadlock)."""
    session = _FakeSession()

    async def _boom(*, force: bool = False) -> None:
        raise RuntimeError("interrupt blew up")

    session.interrupt = _boom  # type: ignore[method-assign]
    detector = _make_detector(session)
    clf = _FakeClassifier()
    detector._classifier = clf  # type: ignore[assignment]

    loop = asyncio.create_task(detector._detection_loop())
    clf._verdict_result = _verdict(AMDCategory.MACHINE_VM)
    clf._verdict_ready.set()

    # execute() must return rather than hang forever
    result = await asyncio.wait_for(detector.execute(), timeout=1.0)
    assert result is not None
    await asyncio.wait_for(loop, timeout=1.0)


async def test_omitted_messages_not_played() -> None:
    session = _FakeSession()
    detector = _make_detector(session, screening_message=NOT_GIVEN, voicemail_message=NOT_GIVEN)
    clf = _FakeClassifier()

    result = await _drive(detector, clf, [_verdict(AMDCategory.MACHINE_VM)])

    assert session.said == []
    assert result.message_playback == "not_played"
