import logging

from livekit.agents.voice.interruptions import FillerAwareInterruptionGuard


def _make_guard(enabled: bool = True) -> FillerAwareInterruptionGuard:
    return FillerAwareInterruptionGuard(
        ignored_words=("uh", "umm", "hmm", "haan"),
        min_confidence=0.6,
        enabled=enabled,
        logger=logging.getLogger("tests.interruption_guard"),
    )


def test_filler_ignored_while_agent_speaks() -> None:
    guard = _make_guard()
    guard.start_turn(agent_speaking=True)

    decision = guard.observe_transcript(
        "uh", confidence=0.9, agent_speaking=True, stage="interim"
    )
    assert decision.reason == "filler_agent_speaking"
    assert decision.allow is False
    assert guard.should_interrupt(transcript="uh", agent_speaking=True) is False
    assert guard.should_process_turn("uh") is False


def test_filler_counts_when_agent_silent() -> None:
    guard = _make_guard()
    guard.start_turn(agent_speaking=False)

    decision = guard.observe_transcript(
        "umm", confidence=0.9, agent_speaking=False, stage="interim"
    )
    assert decision.allow is True
    assert guard.should_process_turn("umm") is True


def test_low_confidence_filler_is_dropped() -> None:
    guard = _make_guard()
    guard.start_turn(agent_speaking=False)

    decision = guard.observe_transcript(
        "uh", confidence=0.2, agent_speaking=False, stage="final"
    )
    assert decision.reason == "low_confidence"
    assert decision.allow is False


def test_real_command_breaks_through() -> None:
    guard = _make_guard()
    guard.start_turn(agent_speaking=True)

    guard.observe_transcript("umm", confidence=0.9, agent_speaking=True, stage="interim")
    decision = guard.observe_transcript(
        "umm okay stop", confidence=0.95, agent_speaking=True, stage="final"
    )

    assert decision.reason == "content"
    assert decision.allow is True
    assert guard.should_interrupt(transcript="umm okay stop", agent_speaking=True) is True
    assert guard.should_process_turn("umm okay stop") is True

