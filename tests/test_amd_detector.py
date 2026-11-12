"""Tests for AMD configuration and detector lifecycle behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents.voice.amd import detector as amd_detector
from livekit.agents.voice.amd.classifier import (
    AMDCategory,
    AMDPredictionEvent,
    _AMDClassifier,
)
from livekit.agents.voice.amd.detector import AMD

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


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


async def _start_detector(
    monkeypatch: pytest.MonkeyPatch, *, kind: object | None = None
) -> tuple[AMD, _AMDClassifier, _FakeRoom, SimpleNamespace, list[str]]:
    """Run AMD setup against a fake room, up to the point where a hangup can arrive.

    The returned list records the session actions ``execute()`` takes on a verdict.
    """

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
        kind=kind if kind is not None else object(),
        identity="callee",
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
    detector._session = session  # type: ignore[assignment]

    await detector._setup(session)  # type: ignore[arg-type]
    return detector, classifier, room, publisher, calls


class TestAMDDetector:
    def test_detection_option_max_endpointing_delay_overrides_activity(self) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(
            llm=llm,
            _activity=SimpleNamespace(max_endpointing_delay=9.0),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            detection_options={"max_endpointing_delay": 0.25},
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._max_endpointing_delay == 0.25

    def test_detection_option_max_endpointing_delay_falls_back_to_activity(self) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(
            llm=llm,
            _activity=SimpleNamespace(max_endpointing_delay=1.25),
        )
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._max_endpointing_delay == 1.25

    def test_detection_option_max_endpointing_delay_falls_back_to_default(self) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._max_endpointing_delay == detector._opts["max_endpointing_delay"]

    def test_amd_defaults_to_wait_until_finished(self) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._wait_until_finished is True

    def test_amd_wait_until_finished_can_be_disabled(self) -> None:
        llm = FakeLLM()
        session = SimpleNamespace(llm=llm, _activity=None)
        detector = AMD(
            session,  # type: ignore[arg-type]
            llm=llm,
            wait_until_finished=False,
            suppress_compatibility_warning=True,
        )

        clf = detector._resolve_classifier(session)  # type: ignore[arg-type]

        assert clf is not None
        assert clf._wait_until_finished is False

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
        detector, classifier, room, publisher, _ = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(detector.execute())

        # the callee starts their greeting, then hangs up before a verdict
        classifier.on_user_speech_started()
        await asyncio.sleep(0)
        room.emit("participant_disconnected", publisher)
        # session teardown closes AMD right after the disconnect
        await detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_disconnected"
        assert room._handlers["participant_disconnected"] == []

    @pytest.mark.parametrize(
        "held", [AMDCategory.HUMAN, AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR]
    )
    async def test_hangup_overrides_held_verdict(
        self, monkeypatch: pytest.MonkeyPatch, held: AMDCategory
    ) -> None:
        detector, classifier, room, publisher, calls = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(detector.execute())

        # the verdict is stored mid-greeting, so the silence gate still holds it
        classifier.on_user_speech_started()
        classifier._set_verdict(_llm_verdict(held))
        await asyncio.sleep(0)
        assert detector.pending
        room.emit("participant_disconnected", publisher)
        await detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_disconnected"
        # nobody is left to interrupt for, or to navigate an IVR with
        assert calls == []

    async def test_hangup_keeps_held_machine_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector, classifier, room, publisher, _ = await _start_detector(monkeypatch)
        execute_task = asyncio.create_task(detector.execute())

        # the carrier says the mailbox is not set up, then hangs up
        classifier.on_user_speech_started()
        classifier._set_verdict(_llm_verdict(AMDCategory.MACHINE_UNAVAILABLE))
        await asyncio.sleep(0)
        assert detector.pending
        room.emit("participant_disconnected", publisher)
        await detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.MACHINE_UNAVAILABLE
        assert result.reason == "llm"

    async def test_hangup_before_sip_answer_settles_participant_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def never_answered(*_: object, **__: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(amd_detector, "wait_for_participant_attribute", never_answered)
        detector, classifier, room, publisher, _ = await _start_detector(
            monkeypatch, kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        )
        execute_task = asyncio.create_task(detector.execute())
        await asyncio.sleep(0)
        assert not classifier.listening

        room.emit("participant_disconnected", publisher)
        await detector.aclose()

        result = await execute_task
        assert result.category == AMDCategory.UNCERTAIN
        assert result.reason == "participant_missing"

    async def test_other_participant_disconnect_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector, classifier, room, _, _ = await _start_detector(monkeypatch)
        classifier.on_user_speech_started()

        room.emit("participant_disconnected", SimpleNamespace(identity="someone-else"))

        assert detector.pending
        assert classifier.listening
        await detector.aclose()

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
