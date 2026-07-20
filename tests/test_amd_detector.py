"""Tests for SIP answer timing in the AMD detector."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, DetectionOptions as PublicDetectionOptions
from livekit.agents.voice.amd import (
    DetectionOptions as AMDDetectionOptions,
    detector as detector_module,
)
from livekit.agents.voice.amd.classifier import AMDCategory, _AMDClassifier
from livekit.agents.voice.amd.detector import AMD

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _make_detector(
    *,
    sip_answer_timeout: float,
    no_speech_threshold: float = 10.0,
) -> tuple[AMD, _AMDClassifier]:
    classifier = _AMDClassifier(
        FakeLLM(),
        no_speech_threshold=no_speech_threshold,
        timeout=20.0,
    )
    detector = object.__new__(AMD)
    detector._closed = False
    detector._classifier = classifier
    detector._opts = {"sip_answer_timeout": sip_answer_timeout}
    return detector, classifier


def test_detection_options_are_exported() -> None:
    assert PublicDetectionOptions is AMDDetectionOptions


def test_default_sip_answer_timeout_is_60_seconds() -> None:
    assert detector_module._DEFAULT_DETECTION_OPTIONS["sip_answer_timeout"] == 60.0


async def test_sip_active_starts_detection_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wait_for_active(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        detector_module,
        "wait_for_participant_attribute",
        wait_for_active,
    )
    detector, classifier = _make_detector(sip_answer_timeout=30.0)

    await detector._wait_for_sip_answer(object(), "callee")

    assert classifier.listening
    assert classifier._detection_timeout_timer is not None
    await classifier.close()


async def test_start_listening_does_not_restart_detection_timer() -> None:
    detector, classifier = _make_detector(sip_answer_timeout=30.0)

    detector._start_listening()
    timer = classifier._detection_timeout_timer
    detector._start_listening()

    assert classifier.listening
    assert timer is not None
    assert classifier._detection_timeout_timer is timer
    await classifier.close()


async def test_non_sip_track_starts_detection_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = SimpleNamespace(sid="track")
    participant = SimpleNamespace(
        identity="callee",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        track_publications={publication.sid: publication},
    )
    room = SimpleNamespace(remote_participants={"callee": participant})

    async def wait_for_track(*_args: Any, **_kwargs: Any) -> Any:
        return publication

    monkeypatch.setattr(detector_module, "wait_for_track_publication", wait_for_track)
    detector, classifier = _make_detector(sip_answer_timeout=30.0)
    detector._participant_identity = "callee"
    detector._stt = NOT_GIVEN
    detector._sip_answer_task = None
    session = SimpleNamespace(_room_io=SimpleNamespace(room=room))

    await detector._setup(session)

    assert classifier.listening
    assert classifier._detection_timeout_timer is not None
    await classifier.close()


async def test_close_cancels_pending_setup_and_answer_tasks() -> None:
    async def wait_forever() -> None:
        await asyncio.Event().wait()

    setup_task = asyncio.create_task(wait_forever())
    answer_task = asyncio.create_task(wait_forever())
    detector = object.__new__(AMD)
    detector._closed = False
    detector._setup_task = setup_task
    detector._sip_answer_task = answer_task
    detector._audio_ch = None
    detector._classifier = None
    detector._span = None
    detector._session = SimpleNamespace(_activity=None, _amd=detector)
    await asyncio.sleep(0)

    await detector.aclose()

    assert setup_task.cancelled()
    assert answer_task.cancelled()
    assert detector._setup_task is None
    assert detector._sip_answer_task is None
    assert detector._session._amd is None


@pytest.mark.parametrize("sip_answer_timeout", [35.0, 45.0])
async def test_sip_answer_timeout_completes_uncertain_verdict(
    monkeypatch: pytest.MonkeyPatch,
    sip_answer_timeout: float,
) -> None:
    async def wait_forever(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        detector_module,
        "wait_for_participant_attribute",
        wait_forever,
    )
    detector, classifier = _make_detector(sip_answer_timeout=sip_answer_timeout)
    started_at = asyncio.get_running_loop().time()

    await detector._wait_for_sip_answer(object(), "callee")

    assert asyncio.get_running_loop().time() - started_at == pytest.approx(sip_answer_timeout)
    assert classifier._verdict_ready.is_set()
    assert classifier._verdict_result is not None
    assert classifier._verdict_result.category == AMDCategory.UNCERTAIN
    assert classifier._verdict_result.reason == "sip_answer_timeout"
    assert not classifier.listening
    assert classifier._detection_timeout_timer is None
    await classifier.close()


async def test_sip_wait_failure_preserves_listening_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_wait(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("participant disconnected")

    monkeypatch.setattr(
        detector_module,
        "wait_for_participant_attribute",
        fail_wait,
    )
    detector, classifier = _make_detector(
        sip_answer_timeout=30.0,
        no_speech_threshold=30.0,
    )

    await detector._wait_for_sip_answer(object(), "callee")

    assert classifier.listening
    assert classifier._detection_timeout_timer is not None
    await classifier._verdict_ready.wait()
    assert classifier._verdict_result is not None
    assert classifier._verdict_result.category == AMDCategory.UNCERTAIN
    assert classifier._verdict_result.reason == "detection_timeout"
    await classifier.close()
