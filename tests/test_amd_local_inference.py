from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents import AMD, NOT_GIVEN, Agent, AgentSession, llm
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.amd import AMDCategory, _inference

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeRecognizeStream, FakeSTT
from .test_amd_detector import ClassifierLLM, commit, end_of_turn, eventually, running

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class DrainingStream(FakeRecognizeStream):
    def __init__(self, model: DrainingSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=model, conn_options=conn_options)
        self.input_ended = asyncio.Event()
        self.release = asyncio.Event()
        self.frames: list[rtc.AudioFrame] = []

    async def _run(self) -> None:
        async for frame in self._input_ch:
            if isinstance(frame, rtc.AudioFrame):
                self.frames.append(frame)
        self.input_ended.set()
        await self.release.wait()


class DrainingSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self.streams: list[DrainingStream] = []

    def stream(self, *, conn_options: APIConnectOptions, **kwargs: Any) -> DrainingStream:
        stream = DrainingStream(self, conn_options)
        self.streams.append(stream)
        return stream


def push_audio(detector: AMD, stt: DrainingSTT) -> DrainingStream:
    detector.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    return stt.streams[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_stt", [None, NOT_GIVEN])
async def test_default_uses_session_model_and_no_extra_stt_or_amd_credentials(
    monkeypatch: pytest.MonkeyPatch, extra_stt: Any
) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    model = ClassifierLLM()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session, stt=extra_stt) as detector:
            await eventually(lambda: detector.started)
            assert detector._llm is model
            await commit(detector, session, model)
            model.prediction(1, AMDCategory.HUMAN)
            assert (await detector.execute()).category == AMDCategory.HUMAN
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_missing_llm_does_not_install_a_reply_guard() -> None:
    session = AgentSession(turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        with pytest.raises(ValueError, match="requires an LLM"):
            await AMD(session).__aenter__()
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_stt_model_string_uses_the_standard_factory_and_closes_owned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    model = DrainingSTT()
    model.aclose = AsyncMock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(inference.STT, "from_model_string", factory)
    async with running(stt="cartesia/ink-2") as (detector, _, _, _):
        stream = push_audio(detector, model)
        factory.assert_called_once_with("cartesia/ink-2")
        await detector.aclose()
        assert stream._task.done()
    model.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["session", "amd"])
async def test_first_final_wins_for_amd_without_changing_agent_transcript(winner: str) -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, session, classifier, reply_model):
        reply_model.fake_response_map["session transcript"] = FakeLLMResponse(
            input="session transcript", content="Hello.", ttft=0, duration=0
        )
        stream = push_audio(detector, stt)
        if winner == "session":
            detector._on_transcript("session transcript")
        stream.send_fake_transcript("AMD transcript")
        await eventually(lambda: detector._transcript._texts["amd"] == "AMD transcript")
        if winner == "amd":
            detector._on_transcript("session transcript")

        info = end_of_turn("session transcript")
        session._activity.on_end_of_turn(info)
        request = await classifier.request()
        assert request.transcript == f"{winner if winner == 'session' else 'AMD'} transcript"
        assert request.transcript_source == winner
        await asyncio.wait_for(stream.input_ended.wait(), 2)
        classifier.prediction(1, AMDCategory.UNCERTAIN)
        reply = await asyncio.wait_for(reply_model.calls.get(), 2)
        texts = [m.text_content for m in reply["chat_ctx"].items if m.type == "message"]
        assert "session transcript" in texts
        assert "AMD transcript" not in texts
        assert info.new_transcript == "session transcript"
        await session._activity._user_turn_completed_atask
        await eventually(lambda: session._activity._no_pending_speech)
        assert any(
            m.type == "message" and m.text_content == "session transcript"
            for m in session.current_agent.chat_ctx.items
        )


@pytest.mark.asyncio
async def test_interim_and_empty_results_do_not_win_and_final_segments_accumulate() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("partial", is_final=False)
        stream.send_fake_transcript("")
        await asyncio.sleep(0)
        assert detector._transcript.winner is None
        stream.send_fake_transcript("Please leave")
        stream.send_fake_transcript("a message.")
        await eventually(lambda: detector._transcript.text == "Please leave a message.")
        detector._on_end_of_turn(end_of_turn("Leave a message."))
        request = await classifier.request()
        assert request.transcript == "Please leave a message."
        assert not hasattr(request, "alternative_transcript")
        assert detector._history[0]["alternative_transcript"] == "Leave a message."


@pytest.mark.asyncio
async def test_waits_for_first_nonempty_stt_result_after_eot() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        detector._on_end_of_turn(end_of_turn(""))
        await asyncio.wait_for(stream.input_ended.wait(), 2)
        assert classifier.requests.empty()
        stream.send_fake_transcript("Hello, can you hear me?")
        request = await classifier.request()
        assert request.transcript == "Hello, can you hear me?"
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await detector.execute()).transcript == request.transcript


@pytest.mark.asyncio
async def test_waiting_transcript_can_finish_after_history_eviction() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        detector.notify_dtmf_sent("1")
        detector._on_end_of_turn(end_of_turn(""))
        for _ in range(20):
            detector._on_end_of_turn(end_of_turn(""))
        assert detector._history[0]["turn_id"] == 2
        stream.send_fake_transcript("Hello, can you hear me?")
        request = await classifier.request()
        assert request.turn_id == 1
        assert request.transcript == "Hello, can you hear me?"
        assert request.dtmf_digits == "1"
        assert request.earlier_turns == []
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_late_winner_text_stays_on_original_turn_and_reaches_next_inference() -> None:
    stt = DrainingSTT()
    async with running(stt=stt, inference_timeout=0.02) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        detector._on_end_of_turn(end_of_turn(""))
        await eventually(lambda: detector._turns[1].decision.done())
        assert detector._turns[1].decision.result().reason == "reused"
        stream.send_fake_transcript("Hello, can you hear me?")
        await eventually(lambda: 1 in detector._updated_turn_ids)
        assert classifier.requests.empty()
        assert detector._turns[1].transcript == ""
        detector._on_end_of_turn(end_of_turn(""))
        request = await classifier.request()
        assert request.turn_id == 2
        assert request.transcript == ""
        assert request.updated_turn_ids == [1]
        assert request.earlier_turns[0]["turn_id"] == 1
        assert request.earlier_turns[0]["transcript"] == "Hello, can you hear me?"
        classifier.prediction(2, AMDCategory.UNCERTAIN)
        await eventually(lambda: detector._turns[2].decision.done())
        detector._on_end_of_turn(end_of_turn(""))
        assert detector._turns[3].decision.result().reason == "reused"
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_late_losing_stt_is_same_turn_evidence_and_cannot_mutate_inflight_context() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, session, classifier, _):
        old_stream = push_audio(detector, stt)
        await commit(detector, session, classifier)
        old_request = detector._classifier_task
        next_stream = push_audio(detector, stt)
        assert next_stream is not old_stream
        old_stream.send_fake_transcript("Hello, can you hear me?")
        await eventually(lambda: 1 in detector._updated_turn_ids)
        assert detector._turns[1].transcript == "hello"
        assert detector._classifier_task is old_request
        assert detector._transcript.text == ""

        detector._on_end_of_turn(end_of_turn("Newer speech"))
        request = await classifier.request()
        assert request.transcript == "Newer speech"
        assert request.updated_turn_ids == [1]
        assert request.earlier_turns[0]["transcript"] == "hello"
        assert request.earlier_turns[0]["alternative_transcript"] == "Hello, can you hear me?"
        old_stream.send_fake_transcript("More old speech.")
        await eventually(
            lambda: detector._history[0]["alternative_transcript"].endswith("More old speech.")
        )
        assert request.earlier_turns[0]["alternative_transcript"] == "Hello, can you hear me?"


@pytest.mark.asyncio
async def test_superseded_inference_keeps_the_participant_transcript() -> None:
    async with running() as (detector, _, classifier, _):
        detector._on_end_of_turn(end_of_turn("Hello, can you hear me?"))
        await classifier.request()
        detector._on_end_of_turn(end_of_turn("Yes, let's schedule that."))
        request = await classifier.request()
        assert request.earlier_turns[0]["transcript"] == "Hello, can you hear me?"
        assert detector._turns[1].decision.result().reason == "superseded"
        assert classifier.responses[1].cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_turns_do_not_cancel_or_count_against_pending_inference(failed: bool) -> None:
    async with running() as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        detector._on_end_of_turn(end_of_turn(""))
        detector._on_end_of_turn(end_of_turn(""))
        assert not classifier.responses[1].done()
        assert not detector._turns[2].decision.done()
        classifier.respond(1, "invalid" if failed else '{"category":"uncertain"}')
        await eventually(lambda: detector._turns[3].decision.done())
        assert [(e.turn_id, e.reason) for e in events] == [
            (1, "inference_error" if failed else "prediction"),
            (2, "reused"),
            (3, "reused"),
        ]
        assert detector._uncertain_turns == (0 if failed else 1)
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_dtmf_on_an_empty_turn_is_context_not_an_inference_trigger() -> None:
    async with running() as (detector, _, classifier, _):
        detector.notify_dtmf_sent("1")
        detector._on_end_of_turn(end_of_turn(""))
        assert classifier.requests.empty()
        detector._on_end_of_turn(end_of_turn("Hello."))
        request = await classifier.request()
        assert request.dtmf_digits == ""
        assert request.earlier_turns[0]["dtmf_digits"] == "1"


@pytest.mark.asyncio
async def test_new_turn_carries_late_evidence_from_a_superseded_request() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, session, classifier, _):
        stream = push_audio(detector, stt)
        await commit(detector, session, classifier)
        stream.send_fake_transcript("Hello, can you hear me?")
        await eventually(lambda: 1 in detector._updated_turn_ids)
        detector._on_end_of_turn(end_of_turn(""))
        assert (await classifier.request()).updated_turn_ids == [1]
        detector._on_end_of_turn(end_of_turn("Yes."))
        assert (await classifier.request()).updated_turn_ids == [1]


@pytest.mark.asyncio
async def test_invalid_transition_falls_back_and_uncertain_preserves_stage() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await detector._prepare_reply(first, llm.ChatContext())
        for turn_id, category, reason in (
            (2, AMDCategory.UNCERTAIN, "prediction"),
            (3, AMDCategory.MACHINE_SCREENING, "inference_error"),
        ):
            info = await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            await detector._prepare_reply(info, llm.ChatContext())
            assert detector._latest.reason == reason
            assert detector._category == AMDCategory.MACHINE_VM
            assert not detector._latest.state_changed


@pytest.mark.asyncio
async def test_slow_menu_does_not_block_classification_or_next_turn() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await detector._prepare_reply(first, llm.ChatContext())
        menu_response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        assert not menu_response.done()
        await commit(detector, session, classifier)
        assert menu_response.cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_cleanup_cancels_both_stt_turns_and_does_not_close_supplied_stt() -> None:
    stt = DrainingSTT()
    stt.aclose = AsyncMock()
    async with running(stt=stt) as (detector, session, classifier, _):
        old_stream = push_audio(detector, stt)
        await commit(detector, session, classifier)
        current_stream = push_audio(detector, stt)
        await asyncio.sleep(0)
        await detector.aclose()
        assert old_stream._task.done()
        assert current_stream._task.done()
        assert not detector._tasks
        assert session.amd is None
        stt.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_stt_stream_failure_leaves_session_transcript_available() -> None:
    stt = DrainingSTT()
    stt.stream = Mock(side_effect=RuntimeError("failed to open"))
    async with running(stt=stt) as (detector, session, classifier, _):
        detector.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await detector.execute()).transcript == "hello"


@pytest.mark.asyncio
async def test_owned_model_cleanup_failure_still_detaches_and_releases_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    stt = DrainingSTT()
    stt.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    monkeypatch.setattr(inference.STT, "from_model_string", Mock(return_value=stt))
    async with running(stt="cartesia/ink-2") as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        await asyncio.wait_for(detector.aclose(), 2)
        assert (await detector.execute()).reason == "cancelled"
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()


@pytest.mark.asyncio
async def test_screening_prediction_is_forwarded_to_session_observability() -> None:
    async with running() as (detector, session, classifier, _):
        host = Mock()
        session._session_host = host
        try:
            info = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await detector._prepare_reply(info, llm.ChatContext())
            assert (
                host._on_amd_prediction.call_args.args[0].category == AMDCategory.MACHINE_SCREENING
            )
        finally:
            session._session_host = None


@pytest.mark.asyncio
async def test_split_screening_rollover_retains_all_turns() -> None:
    async with running() as (detector, _, classifier, _):
        history = [
            ("Please state your name and why you are calling.", AMDCategory.MACHINE_SCREENING),
            ("Okay.", AMDCategory.UNCERTAIN),
            ("They can't take the call.", AMDCategory.UNCERTAIN),
            ("Feel free to leave a message.", AMDCategory.MACHINE_VM),
        ]
        for index, (text, category) in enumerate(history, start=1):
            detector._on_end_of_turn(end_of_turn(text))
            request = await classifier.request()
            assert [e["transcript"] for e in request.earlier_turns] == [
                prior[0] for prior in history[: index - 1]
            ]
            classifier.prediction(index, category)
            await eventually(lambda turn_id=index: detector._turns[turn_id].decision.done())
        assert detector._category == AMDCategory.MACHINE_VM
        assert detector._previous_stage == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
async def test_nonstreaming_stt_receives_the_turn_audio() -> None:
    stt = FakeSTT(fake_transcript="Hello, can you hear me?")
    stt._capabilities.streaming = False
    async with running(stt=stt) as (detector, _, classifier, _):
        detector.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        detector._on_end_of_turn(end_of_turn(""))
        request = await classifier.request()
        assert request.transcript == "Hello, can you hear me?"
        assert request.transcript_source == "amd"


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not JSON", "[]", '{"category":"unknown"}', "x" * 8193])
async def test_classifier_rejects_invalid_and_oversized_provider_output(content: str) -> None:
    model = FakeLLM(
        fake_responses=[FakeLLMResponse(input="input", content=content, ttft=0, duration=0)]
    )
    context = llm.ChatContext()
    context.add_message(role="user", content="input")
    with pytest.raises(ValueError):
        await _inference.classify(model, context)


@pytest.mark.asyncio
async def test_classifier_accepts_fenced_json() -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input", content='```json\n{"category":"human"}\n```', ttft=0, duration=0
            )
        ]
    )
    context = llm.ChatContext()
    context.add_message(role="user", content="input")
    assert (await _inference.classify(model, context)).category == AMDCategory.HUMAN
