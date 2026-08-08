import pytest

from livekit.agents.evals import (
    EvaluationResult,
    ReliabilityObserver,
    ReliabilityTrace,
)
from livekit.agents.evals.judge import JudgmentResult

pytestmark = pytest.mark.evals


def test_reliability_trace_defaults():
    trace = ReliabilityTrace(session_id="test-1")
    assert trace.session_id == "test-1"
    assert trace.turn_count == 0
    assert trace.interruptions == 0
    assert trace.session_complete is False
    assert trace.transcript_integrity == 1.0
    assert trace.tool_reliability == 1.0
    assert trace.response_latency_ms == []
    assert trace.provider_errors == []


def test_reliability_trace_overall_score_perfect():
    trace = ReliabilityTrace(
        session_id="test-perfect",
        turn_count=5,
        interruptions=0,
        session_complete=True,
        transcript_integrity=1.0,
        tool_reliability=1.0,
        response_latency_ms=[500, 800, 1200],
    )
    assert trace.turn_handling_score == 1.0
    assert trace.transcript_integrity_score == 1.0
    assert trace.tool_reliability_score == 1.0
    assert trace.response_latency_score == 1.0
    assert trace.overall_score == 1.0


def test_reliability_trace_overall_score_poor():
    trace = ReliabilityTrace(
        session_id="test-poor",
        turn_count=5,
        interruptions=3,
        session_complete=False,
        transcript_integrity=0.5,
        tool_reliability=0.5,
        response_latency_ms=[6000, 7000, 8000],
    )
    assert trace.turn_handling_score < 0.5
    assert trace.transcript_integrity_score == 0.5
    assert trace.tool_reliability_score == 0.5
    assert trace.response_latency_score == 0.0
    assert trace.overall_score < 0.4


def test_reliability_trace_latency_thresholds():
    trace = ReliabilityTrace(response_latency_ms=[1000])
    assert trace.response_latency_score == 1.0

    trace = ReliabilityTrace(response_latency_ms=[2000])
    assert trace.response_latency_score == 1.0

    trace = ReliabilityTrace(response_latency_ms=[3500])
    assert 0.4 < trace.response_latency_score < 0.6

    trace = ReliabilityTrace(response_latency_ms=[5000])
    assert trace.response_latency_score == 0.0

    trace = ReliabilityTrace(response_latency_ms=[10000])
    assert trace.response_latency_score == 0.0


def test_reliability_trace_to_dict_metadata_only():
    trace = ReliabilityTrace(
        session_id="test-dict",
        turn_count=3,
        interruptions=1,
        session_complete=True,
    )
    d = trace.to_dict(include_transcript=False)
    assert d["session_id"] == "test-dict"
    assert d["turn_count"] == 3
    assert d["interruptions"] == 1
    assert d["transcript_exported"] is False
    assert "scores" in d
    assert "overall" in d["scores"]


def test_reliability_trace_to_dict_with_transcript_opt_in():
    trace = ReliabilityTrace(session_id="test-opt-in")
    d = trace.to_dict(include_transcript=True)
    assert d["transcript_exported"] is True


def test_reliability_trace_to_dict_with_evaluation():
    judgments = {
        "accuracy": JudgmentResult(verdict="pass", reasoning="accurate"),
        "safety": JudgmentResult(verdict="fail", reasoning="unsafe advice"),
    }
    eval_result = EvaluationResult(judgments=judgments)
    trace = ReliabilityTrace(session_id="test-eval", evaluation=eval_result)
    d = trace.to_dict()
    assert "evaluation" in d
    assert d["evaluation"]["score"] == 0.5
    assert d["evaluation"]["all_passed"] is False
    assert "accuracy" in d["evaluation"]["judgments"]
    assert d["evaluation"]["judgments"]["accuracy"]["verdict"] == "pass"


def test_reliability_observer_record_turn():
    observer = ReliabilityObserver(session_id="obs-1")
    observer.record_turn(latency_ms=800)
    observer.record_turn(latency_ms=1200)
    assert observer.trace.turn_count == 2
    assert observer.trace.response_latency_ms == [800, 1200]


def test_reliability_observer_record_interruption():
    observer = ReliabilityObserver(session_id="obs-2")
    observer.record_interruption()
    observer.record_interruption()
    assert observer.trace.interruptions == 2


def test_reliability_observer_mark_complete():
    observer = ReliabilityObserver(session_id="obs-3")
    observer.mark_complete()
    assert observer.trace.session_complete is True
    assert observer.trace.ended_at is not None


def test_reliability_observer_set_evaluation():
    observer = ReliabilityObserver(session_id="obs-4")
    eval_result = EvaluationResult(
        judgments={"accuracy": JudgmentResult(verdict="pass", reasoning="ok")}
    )
    observer.set_evaluation(eval_result)
    assert observer.trace.evaluation is not None
    assert observer.trace.evaluation.score == 1.0


def test_reliability_observer_to_dict_respects_privacy_default():
    observer = ReliabilityObserver(session_id="obs-5")
    d = observer.to_dict()
    assert d["transcript_exported"] is False


def test_reliability_observer_to_dict_respects_opt_in():
    observer = ReliabilityObserver(session_id="obs-6", include_transcript=True)
    d = observer.to_dict()
    assert d["transcript_exported"] is True


@pytest.mark.asyncio
async def test_reliability_observer_flush_idempotent():
    observer = ReliabilityObserver(session_id="obs-flush")
    await observer.flush()
    first_ended = observer.trace.ended_at
    await observer.flush()
    assert observer.trace.ended_at == first_ended


@pytest.mark.asyncio
async def test_reliability_observer_flush_with_null_reporter():
    observer = ReliabilityObserver(session_id="obs-null")
    observer.record_turn(latency_ms=500)
    observer.mark_complete()
    await observer.flush()
    assert observer.trace.session_complete is True
