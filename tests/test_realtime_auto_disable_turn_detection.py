from __future__ import annotations

import logging

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions, inference
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.turn import TurnDetectionMode

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


def _rt_resync_warned(caplog: pytest.LogCaptureFixture) -> bool:
    # match the flip warning specifically, not the generic _validate_turn_detection warnings
    return any(
        r.levelno == logging.WARNING and "resolved at session start" in r.getMessage()
        for r in caplog.records
    )


def _change_turn_detection(
    session: AgentSession, activity: AgentActivity, turn_detection: TurnDetectionMode | None
) -> None:
    # drive the real entry point: session.update_options updates the session-owned turn detection
    # state, then notifies the activity (which re-resolves)
    session._activity = activity
    session.update_options(turn_detection=turn_detection)


def test_update_options_warns_when_runtime_change_would_disable_server_td(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # server-side TD is on at start; switching to a client-side mode would turn it off, but the
    # realtime session is resolved at session start and won't re-sync — warn (PR #6495)
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
    )
    activity = _activity(session)
    assert activity._rt_turn_detection_enabled is True

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _change_turn_detection(session, activity, "manual")

    assert _rt_resync_warned(caplog)


def test_update_options_warns_when_runtime_change_would_enable_server_td(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # server-side TD is off at start (client-side VAD); switching to realtime_llm would turn it
    # back on, which also won't re-sync at runtime — warn (PR #6495)
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)
    assert activity._rt_turn_detection_enabled is False

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _change_turn_detection(session, activity, "realtime_llm")

    assert _rt_resync_warned(caplog)


def test_update_options_no_warn_when_reverting_to_automatic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # reverting to None re-selects automatically rather than pinning an explicit client mode; even
    # though automatic could re-enable server-side TD, we don't warn for it (PR #6495)
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)
    assert activity._rt_turn_detection_enabled is False

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _change_turn_detection(session, activity, None)

    assert not _rt_resync_warned(caplog)


def test_update_options_no_warn_when_server_td_state_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # server-side TD is off at start; switching between two client-side modes keeps it off, so
    # there's nothing to re-sync — no warning
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )
    activity = _activity(session)
    assert activity._rt_turn_detection_enabled is False

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _change_turn_detection(session, activity, "manual")

    assert not _rt_resync_warned(caplog)


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


# --------------------------------------------------------------------------- #
# handoff: reuse a realtime session only when the new agent resolves server TD the same
# --------------------------------------------------------------------------- #


async def test_rt_session_reuse_respects_turn_detection_resolution() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))

    def _act(**handling: object) -> AgentActivity:
        session = AgentSession(
            llm=model,
            vad=FakeVAD(fake_user_speeches=[]),
            turn_handling=TurnHandlingOptions(**handling),  # type: ignore[typeddict-item]
        )
        return _activity(session)

    server_on = _act()  # no client trigger -> server-side TD on
    client_a = _act(turn_detection="vad")  # client vad -> server-side TD off
    client_b = _act(turn_detection="vad")
    assert server_on._rt_turn_detection_enabled is True
    assert client_a._rt_turn_detection_enabled is False

    # differing resolution -> refuse reuse, fall back to a fresh session
    server_on._rt_session = model.session()
    assert (await server_on._detach_reusable_resources(client_a)).rt_session is None

    # same resolution -> reuse
    client_a._rt_session = model.session()
    assert (await client_a._detach_reusable_resources(client_b)).rt_session is not None
