"""Answering-machine detection through its Python host API: ``AMD(session, ...)``, its options,
model selection, lifecycle, events and listeners, over a live ``VoiceSession``.

Each test ports the ``tests/test_amd.py`` test on ``main`` it is named after (spec: ``spec/amd.md``
on ``main``, cited as §N; design: the multi-turn AMD proposal). ``main``'s tests drive the detector's
private hooks; these drive a call instead: the person speaks into the session's microphone, the
session's STT commits each utterance as a turn, and AMD's classifier is a scripted model keyed by
the turn's words. The machine's own rules, the reply gate and their orderings are tested in Rust
and the DST; what is here only a host can show.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from multidict import CIMultiDict
from pydantic import TypeAdapter

from livekit.agents import (
    AMD,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    Agent,
    AMDCategory,
    AMDCompletedEvent,
    AMDLifecycle,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    AMDReason,
    APIConnectOptions,
    APIError,
    Completed,
    RunContext,
    VoiceSession,
    VoiceTask,
    _core,
    beta,
    inference,
    llm,
)
from livekit.agents.beta.workflows import DtmfEvent
from livekit.agents.types import NotGivenOr
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.amd import AMDOptions
from livekit.agents.voice.io import SpeechHandle
from tests.fakes import (
    AgentRoom,
    FakeAudioInput,
    FakeAudioOutput,
    FakeLLM,
    FakeRealtime,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    Hang,
    Reply,
    tool_call,
    until,
)

# `voice/amd/detector.py` on `main`: the built-in stage instructions (§4.1).
SCREENING_INSTRUCTIONS = (
    "Call state: automated call screening. Answer the screening assistant's latest "
    "prompt briefly. Use what you know about who you are and why you are calling. "
    "Then wait for its next prompt."
)
VOICEMAIL_INSTRUCTIONS = (
    "Call state: voicemail. Deliver one concise, self-contained message. "
    "State who you are, why you are calling, and the next step for the recipient."
)
IVR_INSTRUCTIONS = (
    "Call state: an automated phone menu. Use the participant's actual prompt and "
    "the purpose of this call to choose the next step. Use send_dtmf_events for an "
    "explicit keypad choice, or give a short spoken answer when requested. "
    "Do not invent a menu option. After the action, wait for the next prompt."
)
HUMAN_INSTRUCTIONS = (
    "Call state: a human has answered. The latest participant turn is from "
    "the human, not the automated system. Earlier automated prompts no longer "
    "apply. Respond to the human's latest message and continue the call normally. "
    "Do not resume a response to an earlier automated prompt."
)
# `_inference.py` on `main`: the first words of the classifier's and the menu extraction's
# system messages.
CLASSIFY_PROMPT = "Classify the call participant for answering-machine detection."
MENU_PROMPT = "Extract observed IVR menu from current turn's transcript."
# The header LiveKit Inference reads a request's client feature from, and AMD's tag (§4.1).
FEATURE_HEADER = "X-LiveKit-Client-Feature"
DEFAULT_LLM = "google/gemma-4-31b-it"
DEFAULT_STT = "cartesia/ink-whisper"

AGENT_INSTRUCTIONS = "You are calling about an appointment."
TURNS = {"turn_detection": None, "endpointing": {"min_delay": 0.1}}


# -- the call


def record_result(**arguments: object) -> llm.FunctionToolCall:
    return tool_call("record_result", **arguments)


def last_user_words(chat_ctx: llm.ChatContext) -> str:
    """The latest turn's words: with AMD's own STT, the classifier sees both transcripts as
    ``{"transcript_1": session, "transcript_2": amd}`` (§8.1); these are the session's."""
    text = next(
        (m.text_content or "" for m in reversed(chat_ctx.messages()) if m.role == "user"), ""
    )
    try:
        dual = json.loads(text)
    except ValueError:
        return text
    return str(dual["transcript_1"]) if isinstance(dual, dict) else text


def message_texts(chat_ctx: llm.ChatContext) -> list[str]:
    return [m.text_content or "" for m in chat_ctx.messages()]


def tool_names(tools: list[llm.Tool]) -> list[str]:
    def name(tool: Any) -> str:
        if getattr(tool, "id", None):
            return str(tool.id)
        info = getattr(tool, "info", None)
        return str(info.name if info is not None else tool.name)

    return [name(tool) for tool in tools]


class Classifier(FakeLLM):
    """AMD's classifier: answers each classification with ``answers[the turn's words]``, a
    category or any ``FakeLLM`` reply, and each menu extraction with ``menu`` (an empty menu,
    never observed, when not given); as the session's model too, it says ``Okay.`` to the
    agent's requests. Keeps each classification's chat context in
    ``classified``, every call's connection options and extra keyword arguments in
    ``conn_options`` and ``extras``, counts menu requests in ``menus`` and closes in
    ``closes``."""

    def __init__(self, answers: Mapping[str, Reply] | None = None, *, menu: Reply | None = None):
        super().__init__(self._answer)
        self.answers: dict[str, Reply] = dict(answers or {})
        self.menu = menu
        self.classified: list[llm.ChatContext] = []
        self.menus = 0
        self.conn_options: list[APIConnectOptions] = []
        self.extras: list[dict[str, Any]] = []
        self.closes = 0

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: object,
    ) -> llm.LLMStream:
        self.conn_options.append(conn_options)
        self.extras.append(dict(kwargs))
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options, **kwargs)

    def _answer(self, chat_ctx: llm.ChatContext) -> Reply:
        texts = message_texts(chat_ctx)
        if texts and texts[0].startswith(MENU_PROMPT):
            self.menus += 1
            return self.menu if self.menu is not None else record_result(menu="")
        if not texts or not texts[0].startswith(CLASSIFY_PROMPT):
            return "Okay."
        self.classified.append(chat_ctx.copy())
        answer = self.answers[last_user_words(chat_ctx)]
        return record_result(category=answer) if isinstance(answer, str) else answer

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


class AgentModel(FakeLLM):
    """The agent's own model: says ``Okay.`` to every request and counts its closes."""

    def __init__(self) -> None:
        super().__init__(lambda _: "Okay.")
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


class AmdSTT(FakeSTT):
    """AMD's own streaming STT: hears what the person says; counts its streams and closes."""

    def __init__(self, phrases: list[str] | None = None, *, streaming: bool = True) -> None:
        super().__init__(phrases or [], streaming=streaming)
        self.streams: list[stt_stream] = []
        self.closes = 0

    def stream(self, **kwargs: Any) -> stt_stream:
        self.streams.append(super().stream(**kwargs))
        return self.streams[-1]

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


stt_stream = Any  # FakeSTT's stream type is private to tests.fakes


@dataclass
class Call:
    """One outbound call under AMD: what the person says, what AMD and the agent did."""

    session: VoiceSession
    amd: AMD
    classifier: Classifier
    agent_model: AgentModel
    microphone: FakeAudioInput
    predictions: list[AMDPredictionEvent] = field(default_factory=list)
    completions: list[AMDCompletedEvent] = field(default_factory=list)
    menus: list[AMDMenuObservedEvent] = field(default_factory=list)

    def speak(self) -> None:
        """The person says their next phrase; the session commits it as a turn."""
        self.microphone.speak(0.3)

    async def predicted(self, count: int) -> AMDPredictionEvent:
        await until(lambda: len(self.predictions) >= count, f"no prediction {count}")
        return self.predictions[count - 1]

    async def replied(self, count: int) -> llm.ChatContext:
        """The agent's ``count``-th reply request."""
        await until(lambda: len(self.agent_model.requests) >= count, f"no reply {count}")
        return self.agent_model.requests[count - 1]

    async def result(self) -> AMDCompletedEvent:
        return await asyncio.wait_for(self.amd.execute(), 2)

    async def idle_until_done(self) -> AMDCompletedEvent:
        """Skips host time a second at a time until AMD completes: its deadlines fire however
        late it armed them."""
        done = asyncio.ensure_future(self.amd.execute())
        try:
            async with asyncio.timeout(2):
                while not done.done():
                    _core.skip_host_time(1)
                    await asyncio.sleep(0.01)
        finally:
            done.cancel()
        return await self.result()

    def replies(self) -> list[str]:
        return [
            m.text_content or "" for m in self.session.history.messages() if m.role == "assistant"
        ]


def voice_session(
    phrases: list[str],
    *,
    llm_model: llm.LLM | None | NotGivenOr[Any] = NOT_GIVEN,
    tts: bool = True,
    tts_seconds: float = 0.1,
    **options: Any,
) -> tuple[VoiceSession, FakeAudioInput]:
    microphone = FakeAudioInput()
    options = {
        "turn_handling": {**TURNS, "preemptive_generation": {"enabled": False}},
        "aec_warmup_duration": None,
        # one attempt per request: a scripted failure is the classification's answer
        "conn_options": SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0)),
        **options,
    }
    session = VoiceSession(
        llm=AgentModel() if llm_model is NOT_GIVEN else llm_model,  # type: ignore[arg-type]
        vad=FakeVAD(),
        stt=FakeSTT(phrases),
        tts=FakeTTS(seconds=tts_seconds) if tts else NOT_GIVEN,
        **options,
    )
    session.input.audio = microphone
    session.output.audio = FakeAudioOutput()
    return session, microphone


@asynccontextmanager
async def calling(
    phrases: list[str],
    answers: Mapping[str, Reply] | None = None,
    *,
    classifier: Classifier | None = None,
    menu: Reply | None = None,
    agent: Agent | None = None,
    session_options: dict[str, Any] | None = None,
    prepare: Callable[[VoiceSession], None] | None = None,
    **options: Any,
) -> AsyncIterator[Call]:
    """A started session under an entered ``AMD``, listening: no room, so AMD listens at once
    (§5.2). Like ``main``'s ``running()``, AMD has no STT of its own and no silence hold unless
    the test asks."""
    agent_model = AgentModel()
    session, microphone = voice_session(phrases, llm_model=agent_model, **(session_options or {}))
    classifier = classifier or Classifier(answers, menu=menu)
    if prepare is not None:
        prepare(session)
    async with session:
        await session.start(agent or Agent(instructions=AGENT_INSTRUCTIONS))
        amd = AMD(
            session, llm=classifier, **{"stt": None, "machine_silence_threshold": 0, **options}
        )
        call = Call(session, amd, classifier, agent_model, microphone)
        amd.on("amd_prediction", call.predictions.append)
        amd.on("amd_completed", call.completions.append)
        amd.on("amd_menu_observed", call.menus.append)
        async with amd:
            await until(lambda: amd.lifecycle is AMDLifecycle.ACTIVE, "AMD never listened")
            yield call


@contextmanager
def refused(kind: type[Exception], match: str | None = None) -> Iterator[None]:
    """AMD refuses with ``kind``, for its own reason: a missing implementation is no refusal."""
    with pytest.raises(kind, match=match) as caught:
        yield
    assert not isinstance(caught.value, NotImplementedError), "AMD is not implemented"


def handed_to(session: VoiceSession, agent: Agent) -> bool:
    """The session's history records a landed handoff to ``agent``."""
    return any(
        item.type == "agent_handoff" and item.new_agent_id == agent.name
        for item in session.history.items
    )


def dump(event: object) -> dict[str, Any]:
    """An event as JSON, the way an application serializes it."""
    dumped: dict[str, Any] = TypeAdapter(type(event)).dump_python(event, mode="json")
    return dumped


# -- options (§4.1)


async def test_machine_silence_default_and_validation() -> None:
    """A negative silence threshold is refused; zero disables the hold; the default is 1.5 s."""
    session = VoiceSession()
    with refused(ValueError, "machine_silence_threshold must be non-negative"):
        AMD(session, llm=None, stt=None, machine_silence_threshold=-0.1)
    AMD(session, llm=None, stt=None, machine_silence_threshold=0)
    # v2.0 declares each default once, on the options record `AMD(**options)` takes, so the
    # signature has no per-keyword default; `main` reads its keyword's default
    assert AMDOptions().machine_silence_threshold == 1.5


@pytest.mark.parametrize("limit", [0, -1])
async def test_max_inference_timeouts_must_be_positive(limit: int) -> None:
    with refused(ValueError, "max_inference_timeouts must be positive"):
        AMD(VoiceSession(), llm=None, stt=None, max_inference_timeouts=limit)


# `main` checks `voicemail_idle_timeout`; §4.1 refuses every timeout that is not positive and a
# `max_uncertain_turns` below 1 with the same error.
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("voicemail_idle_timeout", 0),
        ("voicemail_idle_timeout", -1),
        ("idle_timeout", 0),
        ("timeout", -1),
        ("inference_timeout", 0),
        ("max_uncertain_turns", 0),
    ],
)
async def test_nonpositive_voicemail_idle_timeout_is_rejected(option: str, value: float) -> None:
    with refused(ValueError, "amd timeouts and max_uncertain_turns must be positive"):
        AMD(VoiceSession(), llm=None, stt=None, **{option: value})


async def test_opening_takes_text_or_instructions_but_not_both() -> None:
    session = VoiceSession()
    with refused(ValueError, "either opening_text or opening_instructions"):
        AMD(session, llm=None, stt=None, opening_text="Hi.", opening_instructions="Greet.")
    with refused(ValueError, "opening_text must not be empty"):
        AMD(session, llm=None, stt=None, opening_text=" ")
    with refused(ValueError, "opening_instructions must not be empty"):
        AMD(session, llm=None, stt=None, opening_instructions="")


async def test_nonstreaming_stt_is_rejected() -> None:
    with refused(ValueError, "amd requires a streaming STT"):
        AMD(VoiceSession(), llm=None, stt=AmdSTT(streaming=False))


# -- events (§4.3)


def test_prediction_event_carries_its_stage() -> None:
    """A prediction's stage defaults to ``uncertain`` and serializes as its wire value."""
    event = AMDPredictionEvent(
        category=AMDCategory.WAIT,
        reason=AMDReason.PREDICTION,
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.stage is AMDCategory.UNCERTAIN
    assert event.turn_id == 0
    assert event.inference_duration is None
    assert not event.is_human and not event.is_machine
    dumped = dump(event)
    assert dumped["type"] == "amd_prediction"
    assert dumped["stage"] == "uncertain"
    assert dumped["category"] == "wait"

    event = AMDPredictionEvent(
        category=AMDCategory.MACHINE_VM,
        stage=AMDCategory.MACHINE_VM,
        reason=AMDReason.PREDICTION,
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.is_machine
    assert dump(event)["stage"] == "machine-vm"


@pytest.mark.parametrize("reason", list(AMDReason))
def test_completed_event_is_error_only_when_amd_stopped_without_detecting(
    reason: AMDReason,
) -> None:
    event = AMDCompletedEvent(category=AMDCategory.UNCERTAIN, reason=reason)
    assert event.is_error is (
        reason
        in {
            AMDReason.INTERNAL_ERROR,
            AMDReason.PARTICIPANT_MISSING,
            AMDReason.AGENT_CHANGED,
            AMDReason.CANCELLED,
        }
    )
    assert event.voicemail_message_played is False
    assert event.last_prediction is None
    # a property, so the wire format is unchanged
    dumped = dump(event)
    assert "is_error" not in dumped
    assert dumped["type"] == "amd_completed"


# -- entry (§5.1)


@pytest.mark.parametrize("started", [False, True])
async def test_rejected_entry_does_not_create_owned_models(
    model_factories: Factories, started: bool
) -> None:
    """An entry that is refused never creates the models it names by string: before the
    session starts, and while another AMD runs (``main`` uses session-level ``ivr_detection``,
    which v2.0 does not have)."""
    session, _ = voice_session([])
    async with session:
        first: AMD | None = None
        if started:
            await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
            first = AMD(session, llm=Classifier(), stt=None)
            await first.__aenter__()
        detector = AMD(session, llm="google/gemini-3.1-flash-lite", stt="cartesia/ink-whisper")
        with refused(RuntimeError):
            await detector.__aenter__()
        assert model_factories.llm_calls == [] and model_factories.stt_calls == []
        assert session.amd is first
        await detector.aclose()
        if first is not None:
            await first.aclose()
            assert session.amd is None


async def test_amd_cannot_enter_while_an_agent_task_runs() -> None:
    """Entering while a ``VoiceTask`` holds the floor is refused: the task's return to its caller
    is a handoff, which AMD rejects (§5.1)."""
    outcome: list[BaseException | None] = []
    collector = Agent(
        instructions="Collect.",
        name="collector",
        output_type=str,
        llm=FakeLLM(lambda _: tool_call("collect")),
    )

    @collector.tool
    async def collect(ctx: RunContext) -> Completed[str]:
        """Collect the reference."""
        session = ctx.live_session
        assert session is not None
        try:
            await AMD(session, llm=Classifier(), stt=None).__aenter__()
            outcome.append(None)
        except BaseException as error:  # noqa: BLE001 - the refusal is the outcome under test
            outcome.append(error)
        return Completed("ref-7")

    front = Agent(instructions="Help.", name="front")

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Find the reference."""
        return await VoiceTask(agent=collector).run()

    model = FakeLLM({"hello": tool_call("lookup"), "ref-7": "Found it."})
    async with VoiceSession(llm=model, vad=None, turn_handling={"turn_detection": "manual"}) as s:
        await s.start(front)
        await s.generate_reply(user_input="hello")
        await until(lambda: outcome, "the task never tried to enter AMD")
        assert isinstance(outcome[0], RuntimeError)
        assert not isinstance(outcome[0], NotImplementedError), "AMD is not implemented"
        assert s.amd is None


@pytest.mark.parametrize(
    ("configuration", "error"),
    [
        ("server_turn_detection", "client-side turn detection"),
        ("auto_tool_reply", "client-controlled tool replies"),
        ("session_tools_only", "per-response tools"),
        ("missing_stt", "session STT"),
    ],
)
async def test_unsupported_configuration_does_not_install_amd(
    configuration: str, error: str
) -> None:
    """A realtime model AMD cannot gate is refused at entry, in ``detector.py``'s order, and
    leaves nothing installed (§5.1)."""
    model = FakeRealtime()
    model._capabilities = replace(
        model.capabilities,
        turn_detection=configuration == "server_turn_detection",
        auto_tool_reply_generation=configuration == "auto_tool_reply",
        per_response_tool_choice=configuration != "session_tools_only",
    )
    async with VoiceSession(
        llm=model,
        stt=NOT_GIVEN if configuration == "missing_stt" else FakeSTT(),
        vad=None,
        turn_handling={"turn_detection": "manual"},
    ) as session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        detector = AMD(session, llm=Classifier(), stt=None)
        with refused(ValueError, error):
            await detector.__aenter__()
        assert session.amd is None
        assert detector.lifecycle is AMDLifecycle.INITIALIZED


async def test_opening_text_needs_a_tts_or_a_realtime_model_that_can_say() -> None:
    """``opening_text`` needs a TTS (or a realtime ``say()``); ``opening_instructions`` an LLM."""
    session, _ = voice_session([], tts=False)
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        with refused(ValueError, "opening_text requires a TTS"):
            await AMD(session, llm=Classifier(), stt=None, opening_text="Hi.").__aenter__()
        assert session.amd is None
    session, _ = voice_session([], llm_model=None)
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        detector = AMD(session, llm=Classifier(), stt=None, opening_instructions="Greet them.")
        with refused(ValueError, "opening_instructions requires an LLM"):
            await detector.__aenter__()
        assert session.amd is None


async def test_missing_llm_does_not_install_turn_hooks() -> None:
    """With no classifier anywhere, entry is refused and leaves the session ungated: a later
    AMD with a classifier enters."""
    session, _ = voice_session([], llm_model=None)
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        with refused(ValueError, "requires an LLM"):
            await AMD(session, llm=None, stt=None).__aenter__()
        assert session.amd is None
        async with AMD(session, llm=Classifier(), stt=None) as detector:
            assert session.amd is detector


async def test_completed_amd_cannot_be_entered_again() -> None:
    async with calling([]) as call:
        await call.amd.aclose()
        result = await call.result()
        with refused(RuntimeError, "new AMD instance"):
            await call.amd.__aenter__()
        assert await call.result() == result
        assert call.session.amd is None


@pytest.mark.parametrize("draining", [False, True])
async def test_amd_cannot_enter_during_an_agent_handoff(draining: bool) -> None:
    """Entry is refused while an agent handoff is in flight, before or while the old agent
    exits; the same instance enters once the handoff landed (§5.1)."""
    exiting, release = asyncio.Event(), asyncio.Event()
    original = Agent(instructions=AGENT_INSTRUCTIONS, name="original")

    @original.on_exit
    async def leave(session: VoiceSession) -> None:
        exiting.set()
        await release.wait()

    replacement = Agent(instructions="Continue the call.", name="replacement")
    session, _ = voice_session([])
    async with session:
        await session.start(original)
        session.update_agent(replacement)
        try:
            detector = AMD(session, llm=Classifier(), stt=None)
            if draining:
                await until(exiting.is_set, "the original agent never began to exit")
            with refused(RuntimeError):
                await detector.__aenter__()
            assert session.amd is None
            assert detector.lifecycle is AMDLifecycle.INITIALIZED
        finally:
            release.set()
        # `main` awaits the handoff's task; v2.0's `current_agent` names the arriving agent at once,
        # so the landed handoff is its record in the history
        await until(lambda: handed_to(session, replacement), "the handoff never landed")
        async with detector:
            assert session.current_agent is replacement
            assert session.amd is detector


# -- lifecycle and completion (§4.2, §12)


@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_terminal_result_completes_once_and_releases_turn_hooks(
    category: AMDCategory,
) -> None:
    """A terminal prediction completes AMD once: every ``execute()`` waiter gets the same result,
    ``amd_completed`` fires once, ``session.amd`` clears and later turns are not gated. A human
    gets the agent's reply under its own instructions, an unreachable call none (§10.1, §10.2)."""
    async with calling(
        ["Hello, who is this?", "Are you there?"], {"Hello, who is this?": category}
    ) as call:
        amd = call.amd
        assert call.session.amd is amd
        first = asyncio.ensure_future(amd.execute())
        second = asyncio.ensure_future(amd.execute())
        call.speak()
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert a == b
        assert a.category == category
        assert a.reason == AMDReason.FINISHED
        assert not a.is_error
        assert a.last_prediction is not None and a.last_prediction.turn_id == 1
        assert a.last_prediction.category == category
        assert call.completions == [a]
        assert amd.lifecycle is AMDLifecycle.FINISHED
        assert call.session.amd is None
        if category == AMDCategory.HUMAN:
            reply = await call.replied(1)
            # a direct human answer replies under the agent's own instructions (§10.2)
            assert not any("Call state:" in text for text in message_texts(reply))
        # a later turn is the session's own: it replies without AMD
        call.speak()
        await call.replied(2 if category == AMDCategory.HUMAN else 1)
        assert len(call.classifier.classified) == 1
        await amd.aclose()
        assert len(call.completions) == 1


async def test_cancelled_execute_waiter_does_not_cancel_detection() -> None:
    async with calling(["Hello?"], {"Hello?": "human"}) as call:
        waiter = asyncio.ensure_future(call.amd.execute())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        call.speak()
        assert (await call.result()).category == AMDCategory.HUMAN


@pytest.mark.parametrize("reason", ["cancelled"])
async def test_nonterminal_finish_releases_a_waiter_without_a_prediction(reason: str) -> None:
    """``aclose()`` while a turn's classification is pending completes with ``cancelled`` in
    the current stage, with no prediction for the turn; the turn is then decided by the "AMD
    completed" row of §10.1, which allows it outside ``machine-unavailable`` and the stages the
    application handles, under the agent's own instructions. ``main`` also reaches
    ``participant_disconnected`` and ``timeout`` through private hooks; the Rust executor tests
    cover those."""
    pending = Hang()
    async with calling(["Hello?"], {"Hello?": [pending]}, inference_timeout=60) as call:
        call.speak()
        await until(pending.started.is_set, "the turn was never classified")
        await call.amd.aclose()
        result = await call.result()
        assert result.reason == reason
        assert result.category == AMDCategory.UNCERTAIN
        assert result.last_prediction is None
        assert result.is_error
        assert call.predictions == []
        await until(pending.ended.is_set, "the pending classification was not cancelled")
        reply = await call.replied(1)
        assert not any("Call state:" in text for text in message_texts(reply))


async def test_session_close_completes_amd_and_releases_resources() -> None:
    async with calling([]) as call:
        await call.session.aclose()
        result = await call.result()
        assert result.reason == AMDReason.CANCELLED
        assert call.amd.lifecycle is AMDLifecycle.FINISHED
        assert call.session.amd is None


async def test_failing_completion_listener_does_not_block_session_close() -> None:
    """An ``amd_completed`` listener that raises is logged; the session still closes (§12.3)."""
    async with calling([]) as call:

        def broken(event: AMDCompletedEvent) -> None:
            raise TypeError("listener bug")

        call.amd.on("amd_completed", broken)
        await asyncio.wait_for(call.session.aclose(), 2)
        result = await call.result()
        assert result.reason == AMDReason.CANCELLED
        assert call.completions == [result]
        assert call.session.amd is None


# -- listeners (§12.1 `internal_error`)


async def test_menu_listener_error_ends_detection_with_internal_error() -> None:
    menu = record_result(menu="Choose a department", options=[{"label": "Sales", "dtmf": "1"}])
    async with calling(
        ["For sales, press 1."], {"For sales, press 1.": "machine-ivr"}, menu=menu
    ) as call:

        def fail_listener(_event: object) -> None:
            raise TypeError("listener bug")

        call.amd.on("amd_menu_observed", fail_listener)
        call.speak()
        result = await call.result()
        assert result.reason == AMDReason.INTERNAL_ERROR
        assert result.category == AMDCategory.MACHINE_IVR
        assert result.is_error


async def test_prediction_listener_failure_cleans_up_a_deferred_release() -> None:
    """A prediction listener that raises ends the run with ``internal_error``; the screening
    reply held for silence is released, not stranded (§10.3: completion frees a held reply)."""
    async with calling(
        ["Please state your name."],
        {"Please state your name.": "machine-screening"},
        machine_silence_threshold=1.5,
    ) as call:

        def on_prediction(_event: object) -> None:
            raise TypeError("listener failed")

        call.amd.on("amd_prediction", on_prediction)
        call.speak()
        result = await call.result()
        assert result.reason == AMDReason.INTERNAL_ERROR
        assert result.category == AMDCategory.MACHINE_SCREENING
        assert call.session.amd is None
        await call.replied(1)


async def test_prediction_listener_cannot_mutate_saved_decision() -> None:
    async with calling(["Leave a message."], {"Leave a message.": "machine-vm"}) as call:

        def mutate(event: AMDPredictionEvent) -> None:
            event.category = AMDCategory.HUMAN
            event.transcript = "changed by listener"

        call.amd.on("amd_prediction", mutate)
        call.speak()
        await call.predicted(1)
        await call.amd.aclose()
        result = await call.result()
        assert result.reason == AMDReason.CANCELLED
        assert result.category == AMDCategory.MACHINE_VM
        assert result.last_prediction is not None
        assert result.last_prediction.category == AMDCategory.MACHINE_VM
        assert result.last_prediction.transcript == "Leave a message."


# -- classification failures (§8.4, §12.1)


@pytest.mark.parametrize(
    "failure",
    [
        APIError("the provider failed", kind="invalid_request"),
        # `detector.py` on `main` treats a model's `ValueError` and `asyncio.TimeoutError` as an
        # inference error too; the host maps them, since the core sees only a class name.
        ValueError("unparseable output"),
        TimeoutError(),
    ],
    ids=["api_error", "value_error", "timeout_error"],
)
async def test_inference_error_falls_back_and_uncertain_keeps_the_stage(
    failure: BaseException,
) -> None:
    """A failed classification is an ``inference_error`` prediction: ``uncertain`` in the stage
    it had, and AMD keeps listening."""
    async with calling(
        ["Leave a message.", "Hello?", "Hi, this is Sam."],
        {"Leave a message.": "machine-vm", "Hello?": failure, "Hi, this is Sam.": "human"},
    ) as call:
        call.speak()
        await call.predicted(1)
        call.speak()
        prediction = await call.predicted(2)
        assert prediction.reason == AMDReason.INFERENCE_ERROR
        assert prediction.category == AMDCategory.UNCERTAIN
        assert prediction.stage == AMDCategory.MACHINE_VM
        assert prediction.inference_duration is None
        assert call.amd.lifecycle is AMDLifecycle.ACTIVE
        call.speak()
        assert (await call.result()).category == AMDCategory.HUMAN


@pytest.mark.parametrize("source", ["classification", "listener"])
async def test_unexpected_inference_errors_finish_with_internal_error(
    caplog: pytest.LogCaptureFixture, source: str
) -> None:
    """A classifier failure outside the provider taxonomy, or a failing prediction listener,
    ends the run with ``internal_error``, and the logs never carry the error's text (``main``
    also fails the machine's transition, a Python internal)."""
    error = (RuntimeError if source == "classification" else TypeError)("private customer content")
    error.__cause__ = ValueError("private provider payload")
    answer: Reply = error if source == "classification" else "machine-ivr"
    async with calling(["Hello?"], {"Hello?": answer}) as call:
        if source == "listener":

            def fail(_event: AMDPredictionEvent) -> None:
                raise error

            call.amd.on("amd_prediction", fail)
        call.speak()
        result = await call.result()
        assert result.reason == AMDReason.INTERNAL_ERROR
    _core.flush_logs()
    failures = [record for record in caplog.records if record.levelname == "ERROR"]
    assert failures
    assert all(record.exc_info is None for record in failures)
    assert all(record.error_type == type(error).__name__ for record in failures)
    assert "private customer content" not in caplog.text
    assert "private provider payload" not in caplog.text


# -- decision models (§4.1, §8.1, §11.2)


def _choice_model(*kinds: llm.DecisionKind) -> Any:
    """A decision model answering each request with the next queued category; ``None`` answers
    with an error for the choice, an exception raises it. Built on ``main``'s
    ``llm.DecisionModel``."""

    class ChoiceModel(llm.DecisionModel):
        def __init__(self) -> None:
            super().__init__(capabilities=llm.DecisionCapabilities(decision_kinds=list(kinds)))
            self.requests: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []
            self.answers: asyncio.Queue[str | BaseException | None] = asyncio.Queue()

        async def _evaluate_impl(
            self,
            *,
            input: str,
            decisions: Mapping[str, Any],
            conn_options: APIConnectOptions,
            extra_headers: Mapping[str, str],
        ) -> Any:
            self.requests.append((input, decisions, dict(extra_headers)))
            category = await self.answers.get()
            if isinstance(category, BaseException):
                raise category
            if category is None:
                return llm.DecisionResponse(results={}, errors={"category": "refused"})
            return llm.DecisionResponse(results={"category": llm.ChoiceResult(value=category)})

    return ChoiceModel()


async def test_decision_model_classifies_with_one_choice_and_skips_menus() -> None:
    model = _choice_model("choice")
    first, second = "Thanks for calling. For billing, press 1.", "For billing, press 1."
    session, microphone = voice_session([first, second])
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        menus: list[AMDMenuObservedEvent] = []
        async with AMD(session, llm=model, stt=None, machine_silence_threshold=0) as detector:
            detector.on("amd_menu_observed", menus.append)
            microphone.speak(0.3)
            await until(lambda: model.requests, "the decision model was never asked")
            text, decisions, headers = model.requests[0]
            assert list(decisions) == ["category"]
            choice = decisions["category"]
            assert type(choice).__name__ == "Choice"
            assert choice.prompt.startswith(CLASSIFY_PROMPT)
            assert list(choice.options) == [category.value for category in AMDCategory]
            assert not any(choice.options.values())
            assert text.startswith('Request: {"stage": "uncertain"')
            assert text.endswith(f"\nTranscript:\nuser: {first}")
            assert FEATURE_HEADER not in headers, "a plugin decision model is not tagged"
            model.answers.put_nowait("machine-ivr")

            microphone.speak(0.3)
            await until(lambda: len(model.requests) == 2, "the second turn was never classified")
            text, _, _ = model.requests[1]
            assert text.startswith('Request: {"stage": "machine-ivr"')
            # the agent's IVR reply may sit between the two turns
            assert text.index(f"user: {first}") < text.index(f"user: {second}")
            model.answers.put_nowait("machine-unavailable")
            result = await asyncio.wait_for(detector.execute(), 2)
            assert result.category == AMDCategory.MACHINE_UNAVAILABLE
        assert menus == []


# The implementation's dev test (`test_a_value_error_from_the_model_is_an_inference_error`) adds
# the raised `ValueError`, which `detector.py` on `main` also counts as an inference error.
@pytest.mark.parametrize(
    "failure", [None, ValueError("unparseable output")], ids=["choice_error", "value_error"]
)
async def test_decision_model_error_is_an_inference_error(failure: BaseException | None) -> None:
    model = _choice_model("choice")
    session, microphone = voice_session(["Hello?"])
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        predictions: list[AMDPredictionEvent] = []
        async with AMD(session, llm=model, stt=None, machine_silence_threshold=0) as detector:
            detector.on("amd_prediction", predictions.append)
            microphone.speak(0.3)
            await until(lambda: model.requests, "the decision model was never asked")
            model.answers.put_nowait(failure)
            await until(lambda: predictions, "no prediction")
            assert predictions[0].reason == AMDReason.INFERENCE_ERROR
            assert predictions[0].category == AMDCategory.UNCERTAIN


async def test_decision_model_without_choice_is_rejected() -> None:
    session, _ = voice_session([])
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS))
        with refused(ValueError, "choice decisions"):
            await AMD(session, llm=_choice_model("probability"), stt=None).__aenter__()
        assert session.amd is None


# -- model selection and ownership (§4.1, §14 invariant 11)


@dataclass
class Factories:
    """The LiveKit Inference factories AMD builds models from strings with, faked: each call
    is recorded and answered with the model below."""

    llm_model: Classifier
    stt_model: AmdSTT
    llm_calls: list[str] = field(default_factory=list)
    stt_calls: list[str] = field(default_factory=list)
    stt_error: Exception | None = None


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> Factories:
    factories = Factories(Classifier({"Hello?": "human"}), AmdSTT())

    def make_llm(model: str) -> Classifier:
        factories.llm_calls.append(model)
        return factories.llm_model

    def make_stt(model: str) -> AmdSTT:
        factories.stt_calls.append(model)
        if factories.stt_error is not None:
            raise factories.stt_error
        return factories.stt_model

    monkeypatch.setattr(inference.LLM, "from_model_string", make_llm)
    monkeypatch.setattr(inference.STT, "from_model_string", make_stt)
    return factories


@pytest.fixture
def cloud_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")


@asynccontextmanager
async def started(
    phrases: list[str], *, llm_model: Any = NOT_GIVEN, agent_llm: Any = NOT_GIVEN
) -> AsyncIterator[tuple[VoiceSession, FakeAudioInput]]:
    session, microphone = voice_session(phrases, llm_model=llm_model)
    async with session:
        await session.start(Agent(instructions=AGENT_INSTRUCTIONS, llm=agent_llm))
        yield session, microphone


@pytest.mark.parametrize("credentials", ["project", "inference", "mixed"])
@pytest.mark.parametrize("finish", [False, True])
async def test_default_models_are_auto_selected_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: Factories,
    credentials: str,
    finish: bool,
) -> None:
    """With LiveKit Cloud credentials (the project's or Inference's own), omitted models are
    ``google/gemma-4-31b-it`` and ``cartesia/ink-whisper``; AMD creates them on entry, uses them
    and closes them once it completes, whether it finished or was closed."""
    if credentials != "project":
        monkeypatch.delenv("LIVEKIT_API_KEY")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "test-key")
    if credentials == "inference":
        monkeypatch.delenv("LIVEKIT_API_SECRET")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "test-secret")
    model, amd_stt = model_factories.llm_model, model_factories.stt_model
    amd_stt.phrases.extend(["Hello?"])
    async with started(["Hello?"]) as (session, microphone):
        async with AMD(session, machine_silence_threshold=0) as detector:
            await until(lambda: detector.lifecycle is AMDLifecycle.ACTIVE, "AMD never listened")
            assert model_factories.llm_calls == [DEFAULT_LLM]
            assert model_factories.stt_calls == [DEFAULT_STT]
            assert session.amd is detector
            if finish:
                microphone.speak(0.3)
                result = await asyncio.wait_for(detector.execute(), 2)
                assert result.category == AMDCategory.HUMAN
                assert model.classified, "the default LLM classified the turn"
                assert amd_stt.streams, "the default STT heard the call"
        assert (model.closes, amd_stt.closes) == (1, 1)
        assert session.amd is None


@pytest.mark.parametrize("missing", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"])
async def test_auto_selection_requires_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: Factories,
    missing: str,
) -> None:
    """Without complete LiveKit Cloud credentials, omitted models are the session's."""
    monkeypatch.delenv(missing)
    session_model = Classifier({"Hello?": "human"})
    async with started(["Hello?"], llm_model=session_model) as (session, microphone):
        async with AMD(session, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
    assert model_factories.llm_calls == [] and model_factories.stt_calls == []
    assert session_model.classified


async def test_auto_selection_does_not_use_cloud_models_for_a_local_server(
    monkeypatch: pytest.MonkeyPatch, cloud_credentials: None, model_factories: Factories
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    session_model = Classifier({"Hello?": "human"})
    async with started(["Hello?"], llm_model=session_model) as (session, microphone):
        async with AMD(session, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
    assert model_factories.llm_calls == [] and model_factories.stt_calls == []
    assert session_model.classified


async def test_none_always_inherits_the_active_agent_models(
    cloud_credentials: None, model_factories: Factories
) -> None:
    """``llm=None`` classifies with the active agent's own LLM, not the session's, even with
    credentials; AMD does not close it."""
    agent_model = Classifier({"Hello?": "human"})
    session_model = Classifier()
    async with started(["Hello?"], llm_model=session_model, agent_llm=agent_model) as (
        session,
        microphone,
    ):
        async with AMD(session, llm=None, stt=None, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
        assert agent_model.classified and not session_model.classified
        assert agent_model.closes == 0
    assert model_factories.llm_calls == [] and model_factories.stt_calls == []


@pytest.mark.parametrize("inherited", ["llm", "stt"])
async def test_model_selection_is_independent_for_each_model(
    cloud_credentials: None,
    model_factories: Factories,
    inherited: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``None`` for one model keeps the other's default; the startup log names each model's
    source, so users know which quotas AMD uses."""
    caplog.set_level("INFO", logger="livekit.agents")
    async with started([], llm_model=Classifier()) as (session, _):
        async with AMD(session, **{inherited: None}):
            if inherited == "llm":
                assert model_factories.llm_calls == []
                assert model_factories.stt_calls == [DEFAULT_STT]
            else:
                assert model_factories.llm_calls == [DEFAULT_LLM]
                assert model_factories.stt_calls == []
    _core.flush_logs()
    [record] = [r for r in caplog.records if r.getMessage().startswith("amd initialized with")]
    # a non-default source names the message
    assert record.getMessage() == "amd initialized with session models"
    if inherited == "llm":
        assert record.llm_source == "session"
        assert record.stt_source == "default"
    else:
        assert record.llm_source == "default"
        assert (record.stt, record.stt_source) == ("session transcript only", "session")


@pytest.mark.parametrize("as_strings", [False, True])
async def test_explicit_models_override_auto_selection(
    cloud_credentials: None, model_factories: Factories, as_strings: bool
) -> None:
    """Given models win over the defaults: strings are built with the Inference factories and
    closed by AMD, instances are used as given and left open."""
    model, amd_stt = model_factories.llm_model, model_factories.stt_model
    async with started([], llm_model=Classifier()) as (session, _):
        detector = AMD(
            session,
            llm="google/gemma-4-31b-it" if as_strings else model,
            stt="cartesia/ink-2" if as_strings else amd_stt,
        )
        assert model_factories.llm_calls == [] and model_factories.stt_calls == []
        async with detector:
            await until(lambda: detector.lifecycle is AMDLifecycle.ACTIVE, "AMD never listened")
            if as_strings:
                assert model_factories.llm_calls == ["google/gemma-4-31b-it"]
                assert model_factories.stt_calls == ["cartesia/ink-2"]
            else:
                assert model_factories.llm_calls == [] and model_factories.stt_calls == []
            await until(lambda: amd_stt.streams, "AMD's STT never opened a stream")
        expected = 1 if as_strings else 0
        assert (model.closes, amd_stt.closes) == (expected, expected)


@pytest.mark.parametrize("extra_stt", [None, NOT_GIVEN])
async def test_default_falls_back_to_session_model_without_amd_credentials(
    monkeypatch: pytest.MonkeyPatch, model_factories: Factories, extra_stt: Any
) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    session_model = Classifier({"Hello?": "human"})
    async with started(["Hello?"], llm_model=session_model) as (session, microphone):
        async with AMD(session, stt=extra_stt, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
    assert session_model.classified
    assert model_factories.llm_calls == [] and model_factories.stt_calls == []


async def test_supplied_model_is_not_closed_by_amd() -> None:
    async with calling([]) as call:
        await call.amd.aclose()
        await call.result()
        assert call.classifier.closes == 0
        assert call.session.amd is None


async def test_stt_model_string_uses_the_standard_factory_and_closes_owned_model(
    model_factories: Factories,
) -> None:
    async with calling([], stt="cartesia/ink-2") as call:
        assert model_factories.stt_calls == ["cartesia/ink-2"]
        await until(lambda: model_factories.stt_model.streams, "AMD's STT never opened a stream")
        await call.amd.aclose()
        await call.result()
    assert model_factories.stt_model.closes == 1


async def test_cleanup_closes_the_run_stream_and_does_not_close_supplied_stt() -> None:
    """AMD closes the stream it opened on a supplied STT, and leaves the STT itself open."""
    amd_stt = AmdSTT(["Hello?"])
    async with calling(["Hello?"], {"Hello?": "uncertain"}, stt=amd_stt) as call:
        call.speak()
        await call.predicted(1)
        assert len(amd_stt.streams) == 1
        await call.amd.aclose()
        await call.result()
        await until(lambda: amd_stt.streams[0].closed, "AMD's stream was left open")
        assert amd_stt.closes == 0
        assert call.session.amd is None


@pytest.mark.parametrize("failure", ["factory", "nonstreaming"])
async def test_partial_model_setup_closes_owned_models(
    model_factories: Factories, failure: str
) -> None:
    """When building the second string model fails, entry fails and closes the first."""
    if failure == "factory":
        model_factories.stt_error = ValueError("STT setup failed")
    else:
        model_factories.stt_model = AmdSTT(streaming=False)
    async with started([], llm_model=Classifier()) as (session, _):
        detector = AMD(session, llm="google/gemini-3.1-flash-lite", stt="cartesia/ink-whisper")
        with refused(ValueError):
            await detector.__aenter__()
        await detector.aclose()
        assert model_factories.llm_model.closes == 1
        assert model_factories.stt_model.closes == (1 if failure == "nonstreaming" else 0)
        assert session.amd is None


async def test_owned_model_cleanup_failure_still_detaches_and_releases_turn_hooks(
    model_factories: Factories,
) -> None:
    """An owned model whose close raises does not keep AMD attached: the run completes and
    the session replies on its own again."""

    class BrokenClose(AmdSTT):
        async def aclose(self) -> None:
            self.closes += 1
            raise RuntimeError("close failed")

    model_factories.stt_model = BrokenClose(["Hello?"])
    async with calling(
        ["Hello?", "Still there?"], {"Hello?": "uncertain"}, stt="cartesia/ink-2"
    ) as call:
        call.speak()
        await call.predicted(1)
        await asyncio.wait_for(call.amd.aclose(), 2)
        assert (await call.result()).reason == AMDReason.CANCELLED
        assert model_factories.stt_model.closes == 1
        assert call.session.amd is None
        replies = len(call.agent_model.requests)
        call.speak()
        await call.replied(replies + 1)


@pytest.mark.parametrize("close_target", ["owned_stt", "owned_llm"])
async def test_slow_resource_close_does_not_keep_amd_attached(
    model_factories: Factories, close_target: str
) -> None:
    """A slow close of a model AMD created does not keep AMD attached: ``session.amd`` is
    ``None`` and the session speaks again before the close ends and ``execute()`` returns
    (``_finish`` detaches before ``_cleanup`` closes the models)."""
    close_started, release_close = asyncio.Event(), asyncio.Event()
    target = model_factories.stt_model if close_target == "owned_stt" else model_factories.llm_model
    original_close = target.aclose

    async def slow_close() -> None:
        close_started.set()
        await release_close.wait()
        await original_close()

    target.aclose = slow_close  # type: ignore[method-assign]
    async with started(["Hello?"]) as (session, microphone):
        detector = AMD(
            session, llm="google/gemma-4-31b-it", stt="cartesia/ink-2", machine_silence_threshold=0
        )
        async with detector:
            microphone.speak(0.3)
            done = asyncio.ensure_future(detector.execute())
            await until(close_started.is_set, "the owned model's close never started")
            assert session.amd is None
            session.say("Hi, this is the clinic.")
            assert not done.done()
            release_close.set()
            result = await asyncio.wait_for(done, 2)
        assert result.category == AMDCategory.HUMAN
        assert (model_factories.llm_model.closes, model_factories.stt_model.closes) == (1, 1)


async def test_failing_model_label_after_creation_leaves_nothing_attached(
    model_factories: Factories,
) -> None:
    """A model whose ``provider`` raises fails the entry before the run attaches: the models
    AMD created close, ``session.amd`` stays ``None``, and a new AMD can enter."""

    class BadLabel(Classifier):
        @property
        def provider(self) -> str:
            raise RuntimeError("provider lookup failed")

    model_factories.llm_model = BadLabel({"Hello?": "human"})
    async with started([], llm_model=Classifier()) as (session, _):
        detector = AMD(session, llm="google/gemma-4-31b-it", stt="cartesia/ink-2")
        with refused(RuntimeError, "provider lookup failed"):
            await detector.__aenter__()
        assert session.amd is None
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert (model_factories.llm_model.closes, model_factories.stt_model.closes) == (1, 1)
        async with AMD(session, llm=Classifier(), stt=None) as later:
            assert session.amd is later


async def test_llm_calls_use_session_connection_options() -> None:
    """Classification and menu requests use the session's ``llm_conn_options`` (§4.1)."""
    options = APIConnectOptions(max_retry=1, retry_interval=0, read_timeout=0.5)
    menu = record_result(menu="Sales", options=[{"label": "Sales", "dtmf": "1"}])
    classifier = Classifier({"Press 1 for sales.": "machine-ivr"}, menu=menu)
    async with calling(
        ["Press 1 for sales."],
        classifier=classifier,
        session_options={"conn_options": SessionConnectOptions(llm_conn_options=options)},
    ) as call:
        call.speak()
        await until(lambda: call.menus, "the menu was never observed")
        # v2.0's framework runs the retries and hands the provider one attempt at a time,
        # with `max_retry=0`; every other limit is the session's
        attempt = replace(options, max_retry=0)
        assert classifier.conn_options and all(o == attempt for o in classifier.conn_options)


# -- LiveKit Inference tagging (§4.1)


CLASSIFIED_HUMAN = (
    b'data: {"id":"c","choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,'
    b'"id":"call_1","type":"function","function":{"name":"record_result",'
    b'"arguments":"{\\"category\\":\\"human\\"}"}}]},"finish_reason":null,"index":0}],'
    b'"created":0,"model":"m","object":"chat.completion.chunk"}\n\n'
    b'data: {"id":"c","choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],'
    b'"created":0,"model":"m","object":"chat.completion.chunk"}\n\ndata: [DONE]\n\n'
)
SAID_OKAY = (
    b'data: {"id":"c","choices":[{"delta":{"content":"Okay.","role":"assistant"},'
    b'"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}'
    b'\n\ndata: {"id":"c","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,'
    b'"model":"m","object":"chat.completion.chunk"}\n\ndata: [DONE]\n\n'
)


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[str, list[tuple[bool, CIMultiDict[str]]]]]:
    """A LiveKit Inference gateway: classifies every AMD request as ``human`` and answers every
    other with ``Okay.``; yields its base URL and, per request, whether it was AMD's and its
    headers."""
    received: list[tuple[bool, CIMultiDict[str]]] = []

    async def complete(request: web.Request) -> web.Response:
        body = await request.json()
        tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
        amd = "record_result" in tools
        # header names are case-insensitive (RFC 9110 §5.1); v2.0's client sends them lowercase
        received.append((amd, CIMultiDict(request.headers)))
        return web.Response(
            body=CLASSIFIED_HUMAN if amd else SAID_OKAY, content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1", received
    await runner.cleanup()


async def test_classifier_tags_only_its_own_request(
    gateway: tuple[str, list[tuple[bool, CIMultiDict[str]]]],
) -> None:
    """An Inference model shared by the agent and AMD sends AMD's feature header on AMD's
    requests only; the agent's reply on the same model is ordinary traffic."""
    base_url, received = gateway
    model = inference.LLM("google/gemma-4-31b-it", base_url=base_url, api_key="k", api_secret="s")
    async with started(["Hello?"], llm_model=model) as (session, microphone):
        async with AMD(session, llm=model, stt=None, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
        await until(lambda: any(not amd for amd, _ in received), "the agent never replied")
    await model.aclose()
    classified = [headers for amd, headers in received if amd]
    replied = [headers for amd, headers in received if not amd]
    assert classified and all(h.get(FEATURE_HEADER) == "AMD" for h in classified)
    assert replied and all(FEATURE_HEADER not in h for h in replied)


@pytest.mark.parametrize(
    "model_headers",
    [{"X-Custom": "kept"}, {FEATURE_HEADER: "Agent"}],
    ids=["other_header", "feature_header"],
)
async def test_classifier_tag_beats_a_model_level_feature_header(
    gateway: tuple[str, list[tuple[bool, CIMultiDict[str]]]], model_headers: dict[str, str]
) -> None:
    """A model's own ``extra_headers`` stay on every request, and AMD's tag wins over a feature
    header of the model's on AMD's requests only. (At ffc0741b22 the model's ``extra_headers``
    replace the call's whole map, ``lk-agents-inference`` ``compose_extra``, so AMD's tag is
    lost.)"""
    base_url, received = gateway
    model = inference.LLM(
        "google/gemma-4-31b-it",
        base_url=base_url,
        api_key="k",
        api_secret="s",
        extra_kwargs={"extra_headers": dict(model_headers)},
    )
    async with started(["Hello?"], llm_model=model) as (session, microphone):
        async with AMD(session, llm=model, stt=None, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
        await until(lambda: any(not amd for amd, _ in received), "the agent never replied")
    await model.aclose()
    classified = [headers for amd, headers in received if amd]
    replied = [headers for amd, headers in received if not amd]
    assert classified and replied
    for headers in classified:
        assert headers.get(FEATURE_HEADER) == "AMD"
        assert all(headers.get(k) == v for k, v in model_headers.items() if k != FEATURE_HEADER)
    for headers in replied:
        assert all(headers.get(k) == v for k, v in model_headers.items())


@pytest.mark.parametrize("every_member", [True, False])
async def test_fallback_adapter_is_tagged_only_when_every_llm_is_inference(
    gateway: tuple[str, list[tuple[bool, CIMultiDict[str]]]], every_member: bool
) -> None:
    """A fallback adapter is an Inference model only when each of its models is one."""
    base_url, received = gateway

    def gateway_model() -> inference.LLM:
        return inference.LLM(DEFAULT_LLM, base_url=base_url, api_key="k", api_secret="s")

    adapter = llm.FallbackAdapter(
        [gateway_model(), gateway_model() if every_member else Classifier()]
    )
    async with started(["Hello?"]) as (session, microphone):
        async with AMD(session, llm=adapter, stt=None, machine_silence_threshold=0) as detector:
            microphone.speak(0.3)
            assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN
    await adapter.aclose()
    classified = [headers for amd, headers in received if amd]
    assert classified
    assert all((h.get(FEATURE_HEADER) == "AMD") is every_member for h in classified)


async def test_plugin_llm_gets_no_header() -> None:
    async with calling(["Hello?"], {"Hello?": "human"}) as call:
        call.speak()
        await call.result()
    assert call.classifier.extras
    assert FEATURE_HEADER not in json.dumps(call.classifier.extras, default=str)


# -- the session while AMD runs (§5.3, §10.5)


@pytest.mark.parametrize("call_kind", ["say", "generate_reply"])
async def test_direct_replies_are_rejected_while_amd_runs(call_kind: str) -> None:
    """``say()`` and ``generate_reply()`` raise while AMD runs; the opening options are the way
    to speak first. Once AMD completes they work again."""
    async with calling(["Hello?"], {"Hello?": "human"}) as call:

        async def speak() -> None:
            if call_kind == "say":
                await call.session.say("Hello there.")
            else:
                await call.session.generate_reply(instructions="Greet them.")

        with refused(RuntimeError):
            await speak()
        assert call.replies() == []
        call.speak()
        await call.result()
        await speak()


async def test_agent_handoff_is_rejected_until_amd_finishes() -> None:
    """``update_agent()`` raises while AMD runs and switches agents once it completed."""
    replacement = Agent(instructions="Take over.", name="replacement")
    async with calling(["Hello?"], {"Hello?": "human"}) as call:
        with refused(RuntimeError):
            call.session.update_agent(replacement)
        assert call.session.current_agent is not replacement
        call.speak()
        await call.result()
        call.session.update_agent(replacement)
        # `current_agent` names the arriving agent at once: the history records the landed handoff
        await until(lambda: handed_to(call.session, replacement), "the handoff never happened")
        assert call.session.current_agent is replacement


@pytest.mark.parametrize("session_enabled", [False, True])
async def test_amd_restores_interruption_settings(session_enabled: bool) -> None:
    """AMD forces interruptions on while it runs, whatever the session says, and the session's
    own setting decides again once it completed (§10.5; v2.0 agents carry no interruption
    setting of their own)."""
    handles: list[SpeechHandle] = []
    async with calling(
        ["Please state your name.", "Hi, this is Sam.", "Yes, I can talk."],
        {
            "Please state your name.": "machine-screening",
            "Hi, this is Sam.": "human",
            "Yes, I can talk.": "human",
        },
        session_options={
            "turn_handling": {
                **TURNS,
                "preemptive_generation": {"enabled": False},
                "interruption": {"enabled": session_enabled},
            }
        },
    ) as call:
        call.session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        call.speak()
        await until(lambda: handles, "the screening reply was never created")
        assert handles[0].allow_interruptions
        await handles[0]
        # a first human after a screener is only likely human: the next turn completes AMD
        call.speak()
        await call.predicted(2)
        await until(lambda: len(call.replies()) == 2, "the likely-human reply never played")
        call.speak()
        await call.result()
        following = call.session.generate_reply(user_input="hello")
        assert following.allow_interruptions is session_enabled
        await following


async def test_user_turn_limit_does_not_reply_while_amd_runs() -> None:
    """A voicemail greeting can run past the user turn limit; while AMD runs it owns the reply,
    so ``on_user_turn_exceeded`` is not called and the turn gets AMD's voicemail message."""
    greeting = "Hi, you've reached Sam, please leave a message after the tone."
    exceeded: list[object] = []

    def limit(session: VoiceSession) -> None:
        @session.on_user_turn_exceeded
        async def over(ctx: RunContext, event: object) -> None:
            exceeded.append(event)

    async with calling(
        [greeting],
        {greeting: "machine-vm"},
        prepare=limit,
        session_options={
            "turn_handling": {
                **TURNS,
                "preemptive_generation": {"enabled": False},
                "user_turn_limit": {"max_words": 3},
            }
        },
    ) as call:
        call.speak()
        assert (await call.predicted(1)).category == AMDCategory.MACHINE_VM
        reply = await call.replied(1)
        assert any(VOICEMAIL_INSTRUCTIONS in text for text in message_texts(reply))
        assert exceeded == []
        assert call.amd.lifecycle is AMDLifecycle.ACTIVE


async def test_amd_commits_short_turns_during_agent_speech() -> None:
    """AMD classifies every turn, so a one-word turn over the agent's speech commits and is
    classified, though ``min_words`` would drop it without AMD (§5.3)."""
    async with calling(
        ["hi"],
        {"hi": "uncertain"},
        opening_text="Hello, this is Alex calling about your appointment.",
        session_options={
            "tts_seconds": 3,
            "turn_handling": {
                **TURNS,
                "preemptive_generation": {"enabled": False},
                "interruption": {"min_words": 3},
            },
        },
    ) as call:
        output = call.session.output.audio
        assert isinstance(output, FakeAudioOutput)
        await until(lambda: output.frames > 0, "the opening never played")
        call.speak()
        prediction = await call.predicted(1)
        assert prediction.transcript == "hi"
        assert [last_user_words(ctx) for ctx in call.classifier.classified] == ["hi"]


# -- the DTMF tool (§11.1)


class DtmfRoom(AgentRoom):
    """A room whose local participant publishes keypad tones with ``publish``."""

    def __init__(self, publish: Callable[..., Any]) -> None:
        super().__init__()
        self._publish = publish

    @property
    def local_participant(self) -> Any:
        return SimpleNamespace(identity="agent-1", publish_dtmf=self._publish)


def dtmf_context(room: DtmfRoom) -> Any:
    """What the tool reads of its ``RunContext``: the call's room, as v2.0's tool reaches it."""
    return SimpleNamespace(live_session=SimpleNamespace(_room=room))


async def test_dtmf_tool_works_without_amd() -> None:
    """The tool sends each key and asks for no follow-up reply: the next participant turn
    decides what happens next."""
    sent: list[dict[str, object]] = []

    async def publish(**kwargs: object) -> None:
        sent.append(kwargs)

    result = await beta.send_dtmf_events(dtmf_context(DtmfRoom(publish)), [DtmfEvent.ONE])
    assert sent == [{"code": 1, "digit": "1"}]
    assert result == llm.ToolResult("Successfully sent DTMF events: 1", reply_required=False)


async def test_dtmf_publish_failure_does_not_expose_provider_error() -> None:
    """A failed publish is a ``ToolError`` with a fixed message and no cause, so the provider's
    payload never reaches the model or a log."""

    async def publish(**kwargs: object) -> None:
        raise RuntimeError("private provider payload")

    with pytest.raises(llm.ToolError, match="^Failed to send DTMF events\\.$") as caught:
        await beta.send_dtmf_events(dtmf_context(DtmfRoom(publish)), [DtmfEvent.ONE])
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert "private provider payload" not in "".join(traceback.format_exception(caught.value))


async def test_dtmf_tool_propagates_cancelled_publish() -> None:
    started = asyncio.Event()

    async def publish(**kwargs: object) -> None:
        started.set()
        await asyncio.Future()

    task = asyncio.ensure_future(
        beta.send_dtmf_events(dtmf_context(DtmfRoom(publish)), [DtmfEvent.ONE])
    )
    await until(started.is_set, "the tone was never published")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_dtmf_tool_propagates_a_publish_cancelled_by_the_room() -> None:
    """A publish that raises ``CancelledError`` itself, with the tool's task not cancelled, is a
    cancellation, not a failed send."""

    async def publish(**kwargs: object) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await beta.send_dtmf_events(dtmf_context(DtmfRoom(publish)), [DtmfEvent.ONE])


# From the implementation's dev tests (`tests/host/test_amd_host.py`).
async def test_an_ivr_reply_runs_the_dtmf_tool() -> None:
    """The model's ``send_dtmf_events`` call in an IVR reply finds AMD's tool with its policy, so
    the reply's run executes the body instead of refusing an unknown tool (§11.1)."""

    def answer(chat_ctx: llm.ChatContext) -> Any:
        texts = [i.text_content or "" for i in chat_ctx.items if i.type == "message"]
        if any("stage" in t for t in texts[:2]):
            return tool_call("record_result", category="machine-ivr")
        if any(i.type == "function_call_output" for i in chat_ctx.items):
            return "Done."
        return tool_call("send_dtmf_events", events=["1"])

    model = FakeLLM(answer)
    session, microphone = voice_session(["press 1 for sales"], llm_model=model)
    executed: list[Any] = []
    session.on("function_tools_executed", executed.append)
    async with session:
        await session.start(Agent(instructions="Reach sales."))
        async with AMD(session, llm=None, stt=None, machine_silence_threshold=0) as detector:
            microphone.speak(0.4)
            said = lambda: [m.text_content for m in session.history.messages()]  # noqa: E731
            await until(lambda: "Done." in said(), "the tool follow-up was said")
            assert detector.lifecycle is AMDLifecycle.ACTIVE, "an IVR stage keeps listening"
        # the IVR reply and its tool follow-up; never the classifier, menu or preemptive requests
        offered = [tool_names(tools) for tools in model.tools]
        assert sum("send_dtmf_events" in names for names in offered) == 2
        [called] = executed[0].function_calls
        assert called.name == "send_dtmf_events"
        # the room-less session has no room to publish on: the body ran and raised
        [output] = executed[0].function_call_outputs
        assert output is not None and "unknown tool" not in output.output


# -- replies the host shapes (§10.2, §11)


async def test_menu_is_observability_only_and_dtmf_tool_is_temporary() -> None:
    """In an IVR, the reply gets the IVR instructions and AMD's ``send_dtmf_events`` for that
    reply only; the agent's tools stay as they were. The menu is reported, never acted on
    (§11.1, §11.2, §14 invariant 9)."""
    agent = Agent(instructions=AGENT_INSTRUCTIONS)

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the appointment."""
        return "Tuesday"

    menu = record_result(
        menu="Choose a department",
        options=[{"label": "Billing", "dtmf": "1", "spoken_response": "billing"}],
    )
    phrases = ["For billing, press 1.", "Hi, this is Sam.", "Yes, I can talk."]
    async with calling(
        phrases,
        {phrases[0]: "machine-ivr", phrases[1]: "human", phrases[2]: "human"},
        agent=agent,
        menu=menu,
    ) as call:
        before = list(call.session.current_agent.tools)
        call.speak()
        reply = await call.replied(1)
        assert any(IVR_INSTRUCTIONS in text for text in message_texts(reply))
        assert sorted(tool_names(call.agent_model.tools[0])) == ["lookup", "send_dtmf_events"]
        await until(lambda: call.menus, "the menu was never observed")
        [observed] = call.menus
        assert observed.turn_id == 1
        assert observed.menu == "Choose a department"
        assert [(o.label, o.dtmf, o.spoken_response) for o in observed.options] == [
            ("Billing", "1", "billing")
        ]
        assert observed.session_id and observed.extraction_duration >= 0
        assert dump(observed)["type"] == "amd_menu_observed"
        assert list(call.session.current_agent.tools) == before
        assert call.amd.lifecycle is AMDLifecycle.ACTIVE

        # a first human after a menu is only likely human: the next turn completes AMD
        call.speak()
        await call.replied(2)
        await until(lambda: len(call.replies()) == 2, "the likely-human reply never played")
        call.speak()
        await call.result()
        reply = await call.replied(3)
        assert tool_names(call.agent_model.tools[2]) == ["lookup"]
        assert any(HUMAN_INSTRUCTIONS in text for text in message_texts(reply))


@pytest.mark.parametrize(
    ("category", "option"),
    [
        (AMDCategory.MACHINE_SCREENING, "screening_instructions"),
        (AMDCategory.MACHINE_VM, "voicemail_instructions"),
        (AMDCategory.MACHINE_IVR, "ivr_instructions"),
    ],
)
async def test_stage_without_instructions_completes_without_a_reply(
    category: AMDCategory, option: str
) -> None:
    """A stage whose instructions are ``None`` is the application's: entering it completes AMD
    with no reply to the turn and no menu extraction (§9.5, §14 invariant 13)."""
    async with calling(["Hello?"], {"Hello?": category.value}, **{option: None}) as call:
        call.speak()
        result = await call.result()
        assert result.category == category
        assert result.reason == AMDReason.FINISHED
        assert result.last_prediction is not None and result.last_prediction.turn_id == 1
        assert not result.voicemail_message_played
        await call.session.wait_for_idle()
        assert call.agent_model.requests == []
        assert call.classifier.menus == 0


async def test_stage_instructions_expire_after_first_human_reply() -> None:
    """Stage instructions shape only their own replies and never enter the saved history: a
    screening reply, the human reply after it, then an ordinary reply (§10.2, §14 invariant 8).
    ``main`` runs this on a realtime model; the realtime request is a Rust test here."""
    phrases = ["Please state your name.", "Hi, this is Sam.", "Sure, go ahead."]
    async with calling(
        phrases,
        {phrases[0]: "machine-screening", phrases[1]: "human", phrases[2]: "human"},
    ) as call:
        call.speak()
        reply = await call.replied(1)
        assert any(SCREENING_INSTRUCTIONS in text for text in message_texts(reply))
        await until(lambda: call.replies(), "the screening reply never played")
        # a first human after screening is likely human: it replies with no stage instructions
        call.speak()
        reply = await call.replied(2)
        assert not any("Call state:" in text for text in message_texts(reply))
        await until(lambda: len(call.replies()) == 2, "the second reply never played")
        call.speak()
        reply = await call.replied(3)
        assert any(HUMAN_INSTRUCTIONS in text for text in message_texts(reply))
        assert (await call.result()).category == AMDCategory.HUMAN
        await until(lambda: len(call.replies()) == 3, "the human reply never played")
        assert call.session.amd is None
        await call.session.generate_reply(user_input="One more thing.")
        assert not any("Call state:" in t for t in message_texts(await call.replied(4)))
        assert not any("Call state:" in t for t in message_texts(call.session.history))


async def test_voicemail_sends_one_delivered_message_and_records_playback() -> None:
    """A voicemail greeting gets one message under the voicemail instructions; once it played,
    the box going quiet completes AMD with the message recorded as played (§10.4)."""
    async with calling(
        ["Please leave a message."], {"Please leave a message.": "machine-vm"}
    ) as call:
        call.speak()
        reply = await call.replied(1)
        assert any(VOICEMAIL_INSTRUCTIONS in text for text in message_texts(reply))
        await until(lambda: call.replies() == ["Okay."], "the message never played")
        result = await call.idle_until_done()
        assert result.reason == AMDReason.IDLE_TIMEOUT
        assert result.category == AMDCategory.MACHINE_VM
        assert result.voicemail_message_played
        assert len(call.agent_model.requests) == 1


@pytest.mark.parametrize("limit", ["idle_timeout", "timeout"])
async def test_default_limits_are_ten_seconds_idle_and_one_minute_overall(limit: str) -> None:
    """By default, a silent, idle call completes with ``idle_timeout`` and is still listening
    9 s in; a call kept busy (a ``wait`` prediction keeps the idle timer reset) completes with
    ``timeout`` a minute after AMD started listening, and is still listening 57 s in."""
    if limit == "idle_timeout":
        async with calling(["Hello?"], {"Hello?": "uncertain"}) as call:
            _core.skip_host_time(9)
            call.speak()
            await call.predicted(1)  # still listening
            await until(lambda: call.replies(), "the reply never played")
            result = await call.idle_until_done()
    else:
        hold = "Please hold."
        async with calling([hold, hold], {hold: "wait"}) as call:
            call.speak()
            await call.predicted(1)
            _core.skip_host_time(56)
            call.speak()
            await call.predicted(2)  # still listening
            result = await call.idle_until_done()
    assert result.reason == limit
    assert not result.is_error


async def test_a_human_at_call_start_completes_at_once() -> None:
    """A human answering first completes AMD on that turn. ``execute()``'s result and the one
    ``amd_completed`` event carry the same completion, with the prediction's §4.3 fields. (With
    the implementation's dev test `test_a_human_answer_completes_once_and_the_agent_replies`: the
    prediction reaches listeners before the completion, a second AMD is refused while this one
    runs, and every ``execute()`` returns the same object.)"""
    async with calling(["Hello, who is this?"], {"Hello, who is this?": "human"}) as call:
        order: list[str] = []
        call.amd.on("amd_prediction", lambda _: order.append("prediction"))
        call.amd.on("amd_completed", lambda _: order.append("completed"))
        with refused(RuntimeError, "already active"):
            await AMD(call.session, llm=call.classifier, stt=None).__aenter__()
        call.speak()
        result = await call.result()
        assert await call.amd.execute() is result
        assert order == ["prediction", "completed"]
        [prediction] = call.predictions
        assert call.completions == [result]
        assert result.last_prediction == prediction
        assert prediction.turn_id == 1
        assert prediction.category == prediction.stage == AMDCategory.HUMAN
        assert prediction.reason == AMDReason.PREDICTION
        assert prediction.transcript == "Hello, who is this?"
        assert prediction.speech_duration > 0
        assert prediction.delay >= 0
        assert prediction.inference_duration is not None and prediction.inference_duration >= 0
        assert prediction.is_human
