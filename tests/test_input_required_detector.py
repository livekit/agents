"""``InputRequiredDetector``: which of an expert's open questions a delegation answers."""

from __future__ import annotations

import pytest

from livekit.agents.delegation import A2ADelegate, InputRequiredCheck, InputRequiredDetector
from livekit.agents.llm import AskedQuestion, ChatContext
from livekit.agents.voice.run_result import RunUsage

from .a2a_driver import Reply, ScriptedLLM, call

pytestmark = pytest.mark.unit

DATE = AskedQuestion("Which date?", "AST_1", 1.0)
SEAT = AskedQuestion("Which seat?", "AST_2", 2.0)


def check() -> InputRequiredCheck:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="friday")
    return InputRequiredCheck("friday", [DATE, SEAT], chat_ctx)


@pytest.mark.parametrize(
    ("reported", "verdict"), [("Q2", SEAT), ("new", "new"), ("unclear", "unclear")]
)
async def test_the_detector_returns_what_the_model_reported(reported: str, verdict: object) -> None:
    model = ScriptedLLM(Reply(tool_calls=[call("report", "r1", verdict=reported)]))
    assert await InputRequiredDetector(model).detect(check()) == verdict
    assert "- Q1: Which date?" in str(model.requests[0].items[-1].content)


async def test_the_detector_reports_its_models_usage() -> None:
    model = ScriptedLLM(Reply(tool_calls=[call("report", "r1", verdict="new")]))
    detector = InputRequiredDetector(model)
    collected: list[object] = []
    detector.on("metrics_collected", collected.append)
    judged = check()
    judged.usage = RunUsage()
    await detector.detect(judged)
    assert collected, "on a call, through the activity"
    assert judged.usage.requests == 1, "in a run, on its usage"


async def test_a_verdict_that_names_no_question_raises() -> None:
    model = ScriptedLLM(Reply(tool_calls=[call("report", "r1", verdict="Q9")]))
    with pytest.raises(ValueError):
        await InputRequiredDetector(model).detect(check())


def test_a_delegate_with_answer_detector_true_runs_the_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret" * 8)
    delegate = A2ADelegate("http://experts.test/backend", answer_detector=True)
    assert isinstance(delegate._answer_detector, InputRequiredDetector)
