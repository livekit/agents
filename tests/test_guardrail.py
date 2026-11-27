from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import utils
from livekit.agents.beta import Block, DecisionEvaluated, DecisionGuardrail, Steer
from livekit.agents.beta.guardrail import (
    _combine,
)
from livekit.agents.llm import ChatContext, ChatMessage, decisions
from livekit.agents.llm.decisions import DecisionResponse, ProbabilityResult
from livekit.agents.types import APIConnectOptions, FlushSentinel
from livekit.agents.voice.events import ConversationItemAddedEvent, UserInputTranscribedEvent

pytestmark = pytest.mark.unit

ABUSIVE = {"abusive": decisions.Probability("The user is abusive.")}
OFF_POLICY = {"off_policy": decisions.Probability("The last sentence promises a refund.")}


class ScriptedModel(decisions.DecisionModel):
    """Scores each decision with ``score(input)``, optionally after a delay."""

    def __init__(self, score: Callable[[str], float], *, delay: float = 0.0) -> None:
        super().__init__(
            capabilities=decisions.DecisionCapabilities(decision_kinds=["probability"])
        )
        self._score = score
        self._delay = delay
        self.inputs: list[str] = []

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, decisions.Decision],
        conn_options: APIConnectOptions,
        extra_headers: Mapping[str, str],
    ) -> DecisionResponse:
        self.inputs.append(input)
        if self._delay:
            await asyncio.sleep(self._delay)
        value = self._score(input)
        return DecisionResponse(
            results={name: ProbabilityResult(value=value) for name in decisions}
        )


class FailingModel(ScriptedModel):
    async def _evaluate_impl(self, **kwargs: Any) -> DecisionResponse:  # type: ignore[override]
        raise RuntimeError("provider down")


class StubSession(utils.EventEmitter[Any]):
    """The parts of AgentSession a guardrail uses."""

    def __init__(self) -> None:
        super().__init__()
        self._guardrail: DecisionGuardrail | None = None
        self.history = ChatContext()
        self.agent_chat_ctx: ChatContext | None = None
        self._activity: Any = None
        self.current_speech: Any = None

    @property
    def current_agent(self) -> Any:
        if self.agent_chat_ctx is None:
            raise RuntimeError("VoiceAgent isn't running")
        return SimpleNamespace(chat_ctx=self.agent_chat_ctx)


def _contains(word: str) -> Callable[[str], float]:
    return lambda text: 1.0 if word in text.lower() else 0.0


def _user_message(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=[text])


async def _check_turn(
    guardrail: DecisionGuardrail | None, *, chat_ctx: ChatContext, user_message: ChatMessage
) -> Block | Steer | None:
    # as AgentActivity does: take the turn's eager state when it ends, cancel what's unused
    assert guardrail is not None
    eager = guardrail._take_turn()
    try:
        return await guardrail._run_user_turn_checks(
            chat_ctx=chat_ctx, user_message=user_message, eager=eager
        )
    finally:
        for evaluation in eager.values():
            evaluation.task.cancel()


async def _run_user_turn(session: StubSession, text: str) -> Block | Steer | None:
    return await _check_turn(
        session._guardrail, chat_ctx=ChatContext(), user_message=_user_message(text)
    )


def test_combine_first_block_wins_then_steers_join() -> None:
    assert _combine([None, Steer("a"), Block(say="x"), Block(say="y")]) == Block(say="x")
    assert _combine([Steer("a"), None, Steer("b")]) == Steer(instructions="a\nb")
    assert _combine([None, None]) is None


async def test_user_turn_block_and_steer() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(_contains("idiot")))  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, eager=False)
    def before_reply(ev: DecisionEvaluated) -> Block | Steer | None:
        assert ev.hook == "user_turn"
        if ev.get("abusive") > 0.5:
            return Block(say="Goodbye.")
        return Steer(instructions="Be brief.")

    assert await _run_user_turn(session, "you idiot") == Block(say="Goodbye.")
    assert await _run_user_turn(session, "hello there") == Steer(instructions="Be brief.")
    await guardrail.aclose()
    assert session._guardrail is None


async def test_user_turn_history_input_includes_conversation() -> None:
    session = StubSession()
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, input="history", eager=False)
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content="How can I help?")
    await _check_turn(session._guardrail, chat_ctx=chat_ctx, user_message=_user_message("my order"))
    assert model.inputs == ["assistant: How can I help?\nuser: my order"]
    await guardrail.aclose()


@pytest.mark.parametrize("on_error, expected", [("allow", None), ("block", Block())])
async def test_user_turn_failure_follows_on_error(on_error: Any, expected: Any) -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=FailingModel(lambda _: 0.0))  # type: ignore[arg-type]
    called = False

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error=on_error, eager=False)
    def before_reply(ev: DecisionEvaluated) -> None:
        nonlocal called
        called = True

    assert await _run_user_turn(session, "hello") == expected
    assert not called
    await guardrail.aclose()


async def test_user_turn_timeout_and_handler_error_follow_on_error() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0, delay=1.0))  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error="block", timeout=0.05, eager=False)
    def slow(ev: DecisionEvaluated) -> None:
        raise AssertionError("not reached")

    assert await _run_user_turn(session, "hello") == Block()
    await guardrail.aclose()

    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error="block", eager=False)
    def broken(ev: DecisionEvaluated) -> None:
        raise ValueError("bug")

    assert await _run_user_turn(session, "hello") == Block()
    await guardrail.aclose()


async def test_slow_async_handler_counts_against_the_timeout() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error="block", timeout=0.05, eager=False)
    async def stalls(ev: DecisionEvaluated) -> None:
        await asyncio.Event().wait()

    assert await asyncio.wait_for(_run_user_turn(session, "hello"), timeout=1.0) == Block()
    await guardrail.aclose()


class PartialModel(ScriptedModel):
    async def _evaluate_impl(self, **kwargs: Any) -> DecisionResponse:  # type: ignore[override]
        return DecisionResponse(results={}, errors=dict.fromkeys(kwargs["decisions"], "no answer"))


@pytest.mark.parametrize("on_error, expected", [("allow", None), ("block", Block())])
async def test_failed_decision_fails_the_check(on_error: Any, expected: Any) -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=PartialModel(lambda _: 0.0))  # type: ignore[arg-type]
    called = False

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error=on_error, eager=False)
    def before_reply(ev: DecisionEvaluated) -> None:
        nonlocal called
        called = True

    assert await _run_user_turn(session, "hello") == expected
    assert not called
    await guardrail.aclose()


async def test_eager_history_is_reevaluated_when_the_context_changed() -> None:
    session = StubSession()
    session.agent_chat_ctx = ChatContext()
    session.agent_chat_ctx.add_message(role="assistant", content="Your order number?")
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, input="history")
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="it is 42", is_final=True)
    )
    await asyncio.sleep(0)

    # unchanged context: the eager evaluation is reused
    await _check_turn(
        session._guardrail,
        chat_ctx=session.agent_chat_ctx.copy(),
        user_message=_user_message("it is 42"),
    )
    assert len(model.inputs) == 1

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="it is 42", is_final=True)
    )
    await asyncio.sleep(0)

    # on_user_turn_completed added a message: same transcript, different input
    edited = session.agent_chat_ctx.copy()
    edited.add_message(role="user", content="(order 42 was corrected to 43)")
    await _check_turn(session._guardrail, chat_ctx=edited, user_message=_user_message("it is 42"))
    assert len(model.inputs) == 3
    assert "corrected to 43" in model.inputs[-1]
    await guardrail.aclose()


async def test_user_turn_ignores_results_its_hook_does_not_support() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 1.0))  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY)
    def sentence(ev: DecisionEvaluated) -> Any:
        return Steer(instructions="not allowed here")

    async def text() -> AsyncIterator[str | FlushSentinel]:
        yield "This is the first sentence of the reply."

    out = [
        c
        async for c in session._guardrail._gate_agent_sentences(
            text(), chat_ctx=ChatContext(), speech_id="s", on_block=_no_block
        )
    ]
    assert out == ["This is the first sentence of the reply."]
    await guardrail.aclose()


async def test_eager_evaluation_is_reused_when_transcript_matches() -> None:
    session = StubSession()
    model = ScriptedModel(_contains("idiot"))
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE)
    def before_reply(ev: DecisionEvaluated) -> Block | None:
        return Block() if ev.get("abusive") > 0.5 else None

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="you", is_final=True)
    )
    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="idiot", is_final=True)
    )
    # interim transcripts don't start evaluations
    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="you id", is_final=False)
    )
    await asyncio.sleep(0)

    assert await _run_user_turn(session, "you  idiot") == Block()
    # the first chunk's request was cancelled; the second one was reused at end of turn
    assert model.inputs == ["you idiot"]

    # a transcript the eager evaluation didn't see is evaluated again
    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="hello", is_final=True)
    )
    await asyncio.sleep(0)
    assert await _run_user_turn(session, "hello idiot") == Block()
    assert model.inputs[-2:] == ["hello", "hello idiot"]
    await guardrail.aclose()


RESOLVED = {"resolved": decisions.Probability("The caller's issue is resolved.")}


async def test_handler_can_switch_the_decisions_it_checks() -> None:
    session = StubSession()
    model = ScriptedModel(_contains("idiot"))
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]
    seen: list[list[str]] = []

    @guardrail.on_user_turn(decisions=ABUSIVE, eager=False)
    def before_reply(ev: DecisionEvaluated) -> Block | None:
        seen.append(sorted(ev.response.results))
        if ev.get("abusive", 0.0) > 0.5:
            # warned once: from now on only check whether the issue got resolved
            guardrail.update_decisions(before_reply, decisions=RESOLVED)
            return Block(say="Please keep it civil.")
        return None

    assert await _run_user_turn(session, "you idiot") == Block(say="Please keep it civil.")
    assert await _run_user_turn(session, "you idiot again") is None
    assert seen == [["abusive"], ["resolved"]]
    await guardrail.aclose()


async def test_eager_evaluation_of_replaced_decisions_is_not_reused() -> None:
    session = StubSession()
    model = ScriptedModel(_contains("idiot"))
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]
    seen: list[list[str]] = []

    @guardrail.on_user_turn(decisions=ABUSIVE)
    def before_reply(ev: DecisionEvaluated) -> None:
        seen.append(sorted(ev.response.results))

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="hello", is_final=True)
    )
    await asyncio.sleep(0)
    guardrail.update_decisions(before_reply, decisions=RESOLVED)

    await _run_user_turn(session, "hello")
    # the eager result covered the same transcript but the old decisions: evaluated again
    assert model.inputs == ["hello", "hello"]
    assert seen == [["resolved"]]
    await guardrail.aclose()


def test_update_decisions_needs_a_registered_handler_and_decisions() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE)
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    with pytest.raises(ValueError, match="at least one decision"):
        guardrail.update_decisions(before_reply, decisions={})
    with pytest.raises(ValueError, match="no guardrail check"):
        guardrail.update_decisions(lambda ev: None, decisions=RESOLVED)


async def _no_block(block: Block) -> None:
    raise AssertionError("unexpected block")


async def _stream(*chunks: str | FlushSentinel) -> AsyncIterator[str | FlushSentinel]:
    for chunk in chunks:
        yield chunk
        await asyncio.sleep(0)


async def test_sentence_gate_releases_the_reply_unchanged() -> None:
    session = StubSession()
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY)
    def during_reply(ev: DecisionEvaluated) -> None:
        assert ev.speech_id == "speech_1"

    reply = "Sure, I can help with that.  Your order shipped yesterday!\nAnything else today?"
    chunks = [reply[i : i + 7] for i in range(0, len(reply), 7)]
    out = [
        c
        async for c in session._guardrail._gate_agent_sentences(
            _stream(*chunks),
            chat_ctx=ChatContext(),
            speech_id="speech_1",
            on_block=_no_block,
        )
    ]
    assert "".join(c for c in out if isinstance(c, str)) == reply
    # each check sees the reply so far, ending with the sentence being checked
    assert model.inputs[-1] == reply
    assert all(reply.startswith(i) for i in model.inputs)
    assert len(model.inputs) >= 2
    await guardrail.aclose()


async def test_sentence_gate_blocks_from_the_failing_sentence() -> None:
    session = StubSession()
    model = ScriptedModel(_contains("refund"))
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY)
    def during_reply(ev: DecisionEvaluated) -> Block | None:
        return Block(say="Let me check.") if ev.get("off_policy") > 0.5 else None

    blocked: list[Block] = []

    async def on_block(block: Block) -> None:
        blocked.append(block)

    out = [
        c
        async for c in session._guardrail._gate_agent_sentences(
            _stream(
                "Thanks for waiting on the line. ",
                FlushSentinel(),
                "You will get a full refund today. ",
                "It should arrive within three days.",
            ),
            chat_ctx=ChatContext(),
            speech_id="s",
            on_block=on_block,
        )
    ]
    # everything before the blocked sentence is released, nothing from it onwards
    assert isinstance(out[-1], FlushSentinel)
    assert "".join(c for c in out[:-1] if isinstance(c, str)) == "Thanks for waiting on the line. "
    assert blocked == [Block(say="Let me check.")]
    await guardrail.aclose()


async def test_sentence_gate_on_error_block() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=FailingModel(lambda _: 0.0))  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY, on_error="block")
    def during_reply(ev: DecisionEvaluated) -> None:
        return None

    blocked: list[Block] = []

    async def on_block(block: Block) -> None:
        blocked.append(block)

    out = [
        c
        async for c in session._guardrail._gate_agent_sentences(
            _stream("Nothing gets through when the check fails."),
            chat_ctx=ChatContext(),
            speech_id="s",
            on_block=on_block,
        )
    ]
    assert out == []
    assert blocked == [Block()]
    await guardrail.aclose()


async def test_agent_turn_runs_on_assistant_messages() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(_contains("sorry")))  # type: ignore[arg-type]
    seen: list[tuple[str, float]] = []
    done = asyncio.Event()

    @guardrail.on_agent_turn(decisions={"empathetic": decisions.Probability("Apologized.")})
    async def after_reply(ev: DecisionEvaluated) -> None:
        seen.append((ev.input_text, ev.get("empathetic")))
        done.set()

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(item=ChatMessage(role="user", content=["hi"])),
    )
    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(
            item=ChatMessage(role="assistant", content=["Sorry for the wait."])
        ),
    )
    await asyncio.wait_for(done.wait(), timeout=1.0)
    assert seen == [("Sorry for the wait.", 1.0)]
    await guardrail.aclose()


def test_a_check_needs_decisions() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):

        @guardrail.on_user_turn(decisions={})
        def before_reply(ev: DecisionEvaluated) -> None:
            return None


async def test_one_guardrail_per_session() -> None:
    session = StubSession()
    first = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    assert session._guardrail is first
    with pytest.raises(RuntimeError, match="already has a DecisionGuardrail"):
        DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    assert session._guardrail is first

    # closing frees the slot for a new one
    await first.aclose()
    second = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    assert session._guardrail is second
    await second.aclose()


async def test_eager_state_is_cleared_when_a_turn_ends_before_the_check() -> None:
    session = StubSession()
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE)
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="first", is_final=True)
    )
    await asyncio.sleep(0)
    # the turn ends without reaching the check (StopResponse, an error, a cancelled turn)
    for evaluation in guardrail._take_turn().values():
        evaluation.task.cancel()

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="second", is_final=True)
    )
    await asyncio.sleep(0)
    await _run_user_turn(session, "second")
    # the next turn's eager input doesn't carry the old text, so its result is reused
    assert model.inputs == ["first", "second"]
    await guardrail.aclose()


async def test_failed_eager_evaluation_is_evaluated_again() -> None:
    class FlakyModel(ScriptedModel):
        async def _evaluate_impl(self, **kwargs: Any) -> DecisionResponse:  # type: ignore[override]
            if not self.inputs:
                self.inputs.append(kwargs["input"])
                raise RuntimeError("provider down")
            return await super()._evaluate_impl(**kwargs)

    session = StubSession()
    model = FlakyModel(_contains("idiot"))
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error="block")
    def before_reply(ev: DecisionEvaluated) -> Block | None:
        return Block(say="Please keep it civil.") if ev.get("abusive") > 0.5 else None

    session.emit(
        "user_input_transcribed", UserInputTranscribedEvent(transcript="hello", is_final=True)
    )
    await asyncio.wait([e.task for e in guardrail._eager.values()])
    # the eager request failed: on_error="block" would block a harmless turn
    assert await _run_user_turn(session, "hello") is None
    assert model.inputs == ["hello", "hello"]
    await guardrail.aclose()


def test_decisions_are_validated_against_the_model_capabilities() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    topic = {"topic": decisions.Choice("The topic.", options={"a": "A", "b": "B"})}

    with pytest.raises(ValueError, match="does not support 'choice'"):

        @guardrail.on_user_turn(decisions=topic)
        def before_reply(ev: DecisionEvaluated) -> None:
            return None

    @guardrail.on_agent_turn(decisions=ABUSIVE)
    def after_reply(ev: DecisionEvaluated) -> None:
        return None

    with pytest.raises(ValueError, match="does not support 'choice'"):
        guardrail.update_decisions(after_reply, decisions=topic)


async def test_eager_checks_skip_realtime_server_turn_detection() -> None:
    session = StubSession()
    # the realtime model ends turns itself: on_user_turn checks never run
    session._activity = SimpleNamespace(_rt_turn_detection_enabled=True)
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE)
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    for text in ("first turn", "second turn"):
        session.emit(
            "user_input_transcribed", UserInputTranscribedEvent(transcript=text, is_final=True)
        )
    await asyncio.sleep(0)
    assert model.inputs == []
    assert guardrail._turn_text == ""
    await guardrail.aclose()


async def test_agent_turn_reports_the_message_and_its_speech() -> None:
    session = StubSession()
    session.current_speech = SimpleNamespace(id="speech_1")
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]
    seen: list[tuple[str | None, str | None]] = []
    done = asyncio.Event()

    @guardrail.on_agent_turn(decisions=ABUSIVE)
    def after_reply(ev: DecisionEvaluated) -> None:
        seen.append((ev.speech_id, ev.item_id))
        done.set()

    session.emit(
        "conversation_item_added",
        ConversationItemAddedEvent(
            item=ChatMessage(id="item_1", role="assistant", content=["Hello."])
        ),
    )
    await asyncio.wait_for(done.wait(), timeout=1.0)
    assert seen == [("speech_1", "item_1")]
    await guardrail.aclose()


async def test_sentence_gate_checks_short_sentences_on_their_own() -> None:
    session = StubSession()
    guardrail = DecisionGuardrail(session, model=ScriptedModel(lambda _: 0.0))  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY)
    def during_reply(ev: DecisionEvaluated) -> Block | None:
        # about the last sentence only, as the decisions are written
        last = ev.input_text.rstrip().rsplit(". ", 1)[-1]
        return Block() if last.startswith("Refund") else None

    blocked: list[Block] = []

    async def on_block(block: Block) -> None:
        blocked.append(block)

    out = [
        c
        async for c in guardrail._gate_agent_sentences(
            _stream("Refund approved. Is there anything else I can help you with today?"),
            chat_ctx=ChatContext(),
            speech_id="s",
            on_block=on_block,
        )
    ]
    # the short first sentence is checked by itself, so nothing is released
    assert out == []
    assert blocked == [Block()]
    await guardrail.aclose()


@pytest.mark.parametrize("on_error, expected", [("allow", None), ("block", Block())])
async def test_user_turn_without_a_transcript_follows_on_error(
    on_error: Any, expected: Any
) -> None:
    session = StubSession()
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_user_turn(decisions=ABUSIVE, on_error=on_error, eager=False)
    def before_reply(ev: DecisionEvaluated) -> None:
        return None

    # a realtime model without an STT: the turn reaches the check with no text
    assert await _run_user_turn(session, "") == expected
    assert model.inputs == []
    await guardrail.aclose()


async def test_sentence_gate_splits_streamed_abbreviations_like_the_whole_text() -> None:
    reply = "Okay. U.S. refunds are not permitted. Sorry."
    seen: list[list[str]] = []
    for chunks in ([reply], ["Okay. U", ".S", ". refunds are not permitted. Sorry."]):
        session = StubSession()
        model = ScriptedModel(lambda _: 0.0)
        guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

        @guardrail.on_agent_sentence(decisions=OFF_POLICY)
        def during_reply(ev: DecisionEvaluated) -> None:
            return None

        async for _ in guardrail._gate_agent_sentences(
            _stream(*chunks), chat_ctx=ChatContext(), speech_id="s", on_block=_no_block
        ):
            pass
        seen.append(sorted(model.inputs))
        await guardrail.aclose()
    # "Okay. U" must not release "Okay." alone before "U.S." is known not to end it
    assert seen[0] == seen[1]


async def test_sentence_gate_releases_unspaced_sentences_while_streaming() -> None:
    session = StubSession()
    model = ScriptedModel(lambda _: 0.0)
    guardrail = DecisionGuardrail(session, model=model)  # type: ignore[arg-type]

    @guardrail.on_agent_sentence(decisions=OFF_POLICY)
    def during_reply(ev: DecisionEvaluated) -> None:
        return None

    more = asyncio.Event()

    async def text() -> AsyncIterator[str | FlushSentinel]:
        for chunk in ["你好！", "LiveKit是", "一个平台。", "今天"]:
            yield chunk
        await more.wait()  # the LLM is still generating
        yield "很好。"

    out: list[str] = []

    async def consume() -> None:
        async for c in guardrail._gate_agent_sentences(
            text(), chat_ctx=ChatContext(), speech_id="s", on_block=_no_block
        ):
            if isinstance(c, str):
                out.append(c)

    task = asyncio.create_task(consume())
    for _ in range(20):
        await asyncio.sleep(0)
    # Chinese has no spaces between words: complete sentences don't wait for the stream
    assert "".join(out) == "你好！LiveKit是一个平台。"
    more.set()
    await task
    assert "".join(out) == "你好！LiveKit是一个平台。今天很好。"
    await guardrail.aclose()
