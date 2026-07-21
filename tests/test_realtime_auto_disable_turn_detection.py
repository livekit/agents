from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions, inference
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.agents.voice.agent_activity import AgentActivity

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _livekit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # inference.TurnDetector() and the adaptive detector read these; keep construction hermetic
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")


def _activity(session: AgentSession) -> AgentActivity:
    return AgentActivity(Agent(instructions="test"), session)


# --------------------------------------------------------------------------- #
# reconcile matrix: client config x model capabilities -> effective server-side TD
# --------------------------------------------------------------------------- #


def test_turn_detector_disables_server_side_td() -> None:
    # explicit client-side EoT + a model that defaulted to (and can disable) server VAD
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_vad_mode_disables_server_side_td() -> None:
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_adaptive_interruption_alone_disables_server_side_td() -> None:
    # no explicit turn_detection, but adaptive interruption is the explicit trigger
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"mode": "adaptive"}),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_manual_disables_server_side_td_without_vad() -> None:
    # manual turn commit drives turns on the client and needs no VAD
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=None,
        turn_handling=TurnHandlingOptions(turn_detection="manual"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_vad_interruption_disables_server_side_td() -> None:
    # any explicit interruption mode (not just adaptive) hands interruption to the client
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"mode": "vad"}),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_stt_mode_keeps_server_side_td() -> None:
    # "stt" turn detection is rejected for realtime models, so it never disables server-side TD
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True


def test_conflict_keeps_server_side_td() -> None:
    # user pinned turn_detection on the model (can_disable=False) AND set client-side -> server wins
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(turn_detection=True, can_disable_turn_detection=False)
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True


def test_explicit_none_on_model_is_already_off() -> None:
    # model turn_detection explicitly None: nothing to disable, ours is honored, no server TD
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(turn_detection=False, can_disable_turn_detection=False)
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


def test_no_client_trigger_keeps_server_side_td() -> None:
    # nothing client-side configured (NOT_GIVEN) -> server-side TD wins, unchanged
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True


def test_realtime_llm_mode_is_not_a_trigger() -> None:
    # turn_detection="realtime_llm" explicitly wants server-side turn-taking
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="realtime_llm"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True


def test_no_vad_does_not_disable() -> None:
    # with VAD explicitly off, we must not disable into a no-turn-detection state
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=None,
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True


def test_non_realtime_llm_reports_no_server_side_td() -> None:
    session = AgentSession(
        llm=FakeLLM(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)

    assert activity._rt_turn_detection_enabled is False


# --------------------------------------------------------------------------- #
# fallback adapter: disable propagates to every underlying session; support is
# the conservative AND across models
# --------------------------------------------------------------------------- #


def test_fallback_adapter_propagates_disable_to_underlying_session() -> None:
    m1 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    m2 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    adapter = RealtimeModelFallbackAdapter([m1, m2])

    assert adapter.capabilities.can_disable_turn_detection is True

    adapter.session(turn_detection_disabled=True)
    assert m1.created_sessions[-1].turn_detection_disabled is True


def test_fallback_adapter_support_is_conservative_and() -> None:
    m1 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    m2 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=False))
    adapter = RealtimeModelFallbackAdapter([m1, m2])

    assert adapter.capabilities.can_disable_turn_detection is False
