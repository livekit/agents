"""AMD's Python host: options and entry errors, model selection, events, completion and the
DTMF tool, over a session the core runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from livekit.agents import (
    AMD,
    Agent,
    AMDCategory,
    AMDCompletedEvent,
    AMDLifecycle,
    AMDPredictionEvent,
    AMDReason,
    VoiceSession,
    _core,
    llm,
)
from livekit.agents.voice.amd import Choice
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    tool_call,
    until,
)

TURN_HANDLING: Any = {"turn_detection": None, "endpointing": {"min_delay": 0.1}}


def _classifier(category: str, reply: str = "Hi, this is Ada.") -> FakeLLM:
    """Classifies every turn as ``category`` and answers every reply with ``reply``."""

    def answer(chat_ctx: llm.ChatContext) -> Any:
        system = [i.text_content for i in chat_ctx.items if i.type == "message"]
        if any("record_result" in (text or "") or "stage" in (text or "") for text in system[:2]):
            return tool_call("record_result", category=category)
        return reply

    return FakeLLM(answer)


def _session(model: llm.LLM, phrases: list[str]) -> tuple[VoiceSession[Any], FakeAudioInput]:
    session: VoiceSession[Any] = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT(phrases),
        llm=model,
        tts=FakeTTS(),
        turn_handling=TURN_HANDLING,
        aec_warmup_duration=None,
    )
    audio = FakeAudioInput()
    session.input.audio = audio
    session.output.audio = FakeAudioOutput()
    return session, audio


@pytest.fixture(autouse=True)
def _no_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# Catches: an option `main` refuses reaching the core, a negative silence threshold included.
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timeout": 0}, "amd timeouts and max_uncertain_turns must be positive"),
        ({"idle_timeout": -1}, "amd timeouts and max_uncertain_turns must be positive"),
        ({"max_uncertain_turns": 0}, "amd timeouts and max_uncertain_turns must be positive"),
        ({"max_inference_timeouts": 0}, "max_inference_timeouts must be positive"),
        ({"machine_silence_threshold": -0.1}, "machine_silence_threshold must be non-negative"),
        (
            {"opening_text": "hi", "opening_instructions": "greet"},
            "set either opening_text or opening_instructions, not both",
        ),
        ({"opening_text": " "}, "opening_text must not be empty"),
        ({"opening_instructions": ""}, "opening_instructions must not be empty"),
    ],
)
def test_the_constructor_refuses_invalid_options(options: dict[str, Any], message: str) -> None:
    session: VoiceSession[Any] = VoiceSession(llm=FakeLLM())
    with pytest.raises(ValueError, match=message):
        AMD(session, llm=None, stt=None, **options)


# Catches: a supplied STT that cannot stream accepted, where `main` raises at construction.
def test_the_constructor_refuses_a_non_streaming_stt() -> None:
    session: VoiceSession[Any] = VoiceSession(llm=FakeLLM())
    with pytest.raises(ValueError, match="amd requires a streaming STT"):
        AMD(session, llm=None, stt=FakeSTT(streaming=False))


# Catches: AMD picking Inference models without LiveKit Cloud credentials.
def test_omitted_models_fall_back_to_the_session_without_credentials() -> None:
    detector = AMD(VoiceSession(llm=FakeLLM()))
    assert (detector._llm, detector._llm_model, detector._stt, detector._stt_model) == (
        None,
        None,
        None,
        None,
    )
    assert detector._model_sources == ("session", "session")
    assert detector.lifecycle is AMDLifecycle.INITIALIZED


# Catches: the auto selection ignoring Cloud credentials.
def test_omitted_models_are_the_inference_defaults_with_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://project.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    assert _core._core_amd_auto_models()
    detector = AMD(VoiceSession(llm=FakeLLM()))
    assert detector._llm_model == "google/gemma-4-31b-it"
    assert detector._stt_model == "cartesia/ink-whisper"
    assert detector._model_sources == ("default", "default")
    kept = AMD(VoiceSession(llm=FakeLLM()), llm=None, stt=None)
    assert kept._model_sources == ("session", "session")


def test_event_models_keep_mains_shape() -> None:
    event = AMDPredictionEvent(
        category=AMDCategory.WAIT,
        reason=AMDReason.PREDICTION,
        transcript="hi",
        speech_duration=0,
        delay=0,
    )
    assert event.stage is AMDCategory.UNCERTAIN
    assert json.loads(event.model_dump_json())["stage"] == "uncertain"
    assert not event.is_human and not event.is_machine
    assert AMDCompletedEvent(category=AMDCategory.UNCERTAIN, reason=AMDReason.CANCELLED).is_error
    assert not AMDCompletedEvent(category=AMDCategory.HUMAN, reason=AMDReason.TIMEOUT).is_error


# Catches: entering AMD before the session runs.
async def test_entering_before_start_raises() -> None:
    session: VoiceSession[Any] = VoiceSession(llm=FakeLLM())
    with pytest.raises(RuntimeError):
        async with AMD(session, llm=None, stt=None):
            pass


# Catches: a human answer not completing the run once, with its prediction first, or the
# agent's reply to that turn held after completion.
async def test_a_human_answer_completes_once_and_the_agent_replies() -> None:
    model = _classifier("human")
    session, audio = _session(model, ["hello who is this"])
    seen: list[object] = []
    async with session:
        await session.start(Agent(instructions="You are calling Bob."))
        detector = AMD(session, llm=None, stt=None)
        detector.on("amd_prediction", seen.append)
        detector.on("amd_completed", seen.append)
        async with detector:
            assert session.amd is detector
            with pytest.raises(RuntimeError, match="already active"):
                async with AMD(session, llm=None, stt=None):
                    pass
            audio.speak(0.4)
            result = await detector.execute()
            assert await detector.execute() is result
        assert session.amd is None
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert result.category is AMDCategory.HUMAN
        assert result.reason is AMDReason.FINISHED
        assert result.last_prediction is not None and result.last_prediction.turn_id == 1
        assert [type(e) for e in seen] == [AMDPredictionEvent, AMDCompletedEvent]
        assert seen[-1] == result
        await until(
            lambda: any(
                i.type == "message" and i.role == "assistant" for i in session.history.items
            ),
            "the human turn got its reply",
        )
        with pytest.raises(RuntimeError, match="use a new AMD instance"):
            await detector.__aenter__()


# Catches: a raising prediction listener not ending the run with `internal_error`, or an
# `amd_completed` listener's error escaping cleanup.
async def test_a_failing_prediction_listener_ends_the_run_with_an_internal_error() -> None:
    model = _classifier("uncertain")
    session, audio = _session(model, ["hello"])
    async with session:
        await session.start(Agent(instructions="Help."))
        detector = AMD(session, llm=None, stt=None)

        def fail(_: AMDPredictionEvent) -> None:
            raise RuntimeError("listener")

        def fail_completed(_: AMDCompletedEvent) -> None:
            raise RuntimeError("completed listener")

        detector.on("amd_prediction", fail)
        detector.on("amd_completed", fail_completed)
        async with detector:
            audio.speak(0.4)
            result = await detector.execute()
        assert result.reason is AMDReason.INTERNAL_ERROR
        assert result.is_error


# Catches: `aclose()` not completing a listening run with `cancelled`.
async def test_aclose_cancels_a_running_detection() -> None:
    session, _ = _session(_classifier("human"), [])
    async with session:
        await session.start(Agent(instructions="Help."))
        detector = AMD(session, llm=None, stt=None)
        async with detector:
            await until(lambda: detector.lifecycle is AMDLifecycle.ACTIVE, "AMD listens")
        result = await detector.execute()
        assert result.reason is AMDReason.CANCELLED and result.category is AMDCategory.UNCERTAIN


class _Decisions:
    """A decision model: one ``Choice`` per call, answered with ``category``."""

    def __init__(self, category: str, error: BaseException | None = None) -> None:
        self.category = category
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []

    class capabilities:  # noqa: N801
        decision_kinds = ["choice"]

    model = "decider"
    provider = "fake"

    async def evaluate(self, *, input: str, decisions: Mapping[str, Any], **kwargs: Any) -> Any:
        self.calls.append((input, decisions, kwargs))
        if self.error is not None:
            raise self.error

        class _Result:
            kind = "choice"
            value = self.category

        class _Response:
            results = {name: _Result() for name in decisions}

        return _Response()


# Catches: a decision model asked for anything but one `choice` over the seven categories.
async def test_a_decision_model_classifies_with_one_choice() -> None:
    decider = _Decisions("human")
    session, audio = _session(FakeLLM(["Hi."]), ["hello"])
    async with session:
        await session.start(Agent(instructions="Help."))
        async with AMD(session, llm=decider, stt=None) as detector:
            audio.speak(0.4)
            result = await detector.execute()
    assert result.category is AMDCategory.HUMAN
    assert len(decider.calls) == 1
    _, decisions, kwargs = decider.calls[0]
    [(name, choice)] = decisions.items()
    assert isinstance(choice, Choice)
    assert list(choice.options) == [c.value for c in AMDCategory]
    assert "extra_headers" not in kwargs, "only Inference models carry AMD's feature header"


# Catches: a model's `ValueError` ending the run instead of an `inference_error` prediction.
async def test_a_value_error_from_the_model_is_an_inference_error() -> None:
    decider = _Decisions("human", error=ValueError("bad output"))
    session, audio = _session(FakeLLM(["Hi."]), ["hello"])
    predictions: list[AMDPredictionEvent] = []
    async with session:
        await session.start(Agent(instructions="Help."))
        async with AMD(session, llm=decider, stt=None) as detector:
            detector.on("amd_prediction", predictions.append)
            audio.speak(0.4)
            await until(lambda: predictions, "the turn got a prediction")
    assert predictions[0].reason is AMDReason.INFERENCE_ERROR


# Catches: a decision model without `choice` accepted.
async def test_a_decision_model_without_choice_is_refused() -> None:
    decider = _Decisions("human")
    decider.capabilities = type("caps", (), {"decision_kinds": ["probability"]})  # type: ignore[assignment]
    session, _ = _session(FakeLLM(), [])
    async with session:
        await session.start(Agent(instructions="Help."))
        with pytest.raises(ValueError, match="supports choice decisions"):
            async with AMD(session, llm=decider, stt=None):
                pass


# Catches: the model calling `send_dtmf_events` in an IVR reply finding no tool, or no policy
# for it, so the reply fails before the body runs.
async def test_an_ivr_reply_runs_the_dtmf_tool() -> None:
    def answer(chat_ctx: llm.ChatContext) -> Any:
        texts = [i.text_content or "" for i in chat_ctx.items if i.type == "message"]
        if any("stage" in t for t in texts[:2]):
            return tool_call("record_result", category="machine-ivr")
        if any(i.type == "function_call_output" for i in chat_ctx.items):
            return "Done."
        return tool_call("send_dtmf_events", events=["1"])

    model = FakeLLM(answer)
    session, audio = _session(model, ["press 1 for sales"])
    executed: list[Any] = []
    session.on("function_tools_executed", executed.append)
    async with session:
        await session.start(Agent(instructions="Reach sales."))
        async with AMD(session, llm=None, stt=None, machine_silence_threshold=0) as detector:
            audio.speak(0.4)
            await until(lambda: executed, "the DTMF call ran")
            assert detector.lifecycle is AMDLifecycle.ACTIVE, "an IVR stage keeps listening"
        offered = [[t.id for t in tools] for tools in model.tools]
        # the IVR reply and its tool follow-up; never the classifier, menu or preemptive requests
        assert sum("send_dtmf_events" in ids for ids in offered) == 2
        [call] = executed[0].function_calls
        assert call.name == "send_dtmf_events"
        # the room-less session has no room to publish on: the body ran and raised
        [output] = executed[0].function_call_outputs
        assert output is not None and "unknown tool" not in output.output
