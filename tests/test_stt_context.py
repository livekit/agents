from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest

from livekit import rtc
from livekit.agents.llm import LLM, ChatContext, ChatMessage, CollectedResponse, FunctionToolCall
from livekit.agents.stt import STT, RecognizeStream, SpeechEvent, STTCapabilities
from livekit.agents.stt.recognition_context import (
    _PENDING_TTL,
    KeytermDetector,
    _detect_keyterms,
    _format_input,
    _parse_tool_call,
    _resolve_detection,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import ConversationItemAddedEvent

pytestmark = pytest.mark.unit


def _detector(*, static_keyterms: list[str] | None = None, **options: Any) -> KeytermDetector:
    return KeytermDetector(static_keyterms=static_keyterms, options=options)


def _entries(d: KeytermDetector) -> list[tuple[str, bool]]:
    """Detected terms with their confirmed flag (confirmed first, then pending)."""
    return [(t, True) for t in d._detected_terms] + [(t, False) for t in d._pending_terms]


def _ctx(text: str = "hello") -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


class _RecordingSTT(STT):
    """STT that records every _update_session_keyterms() / _push_conversation_item() call."""

    def __init__(
        self, *, supports_keyterms: bool = True, supports_chat_context: bool = False
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=False,
                keyterms=supports_keyterms,
                chat_context=supports_chat_context,
            )
        )
        self.pushed: list[list[str]] = []
        self.chat_items: list[ChatMessage] = []

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = ...,  # type: ignore[assignment]
    ) -> RecognizeStream:
        raise NotImplementedError

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self.pushed.append(list(keyterms))

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, ChatMessage):
            self.chat_items.append(ev.item)


class _FakeStream:
    def __init__(self, pending: list[str], confirm: list[str], remove: list[str]) -> None:
        self._args = json.dumps({"pending": pending, "confirm": confirm, "remove": remove})

    async def collect(self) -> CollectedResponse:
        call = FunctionToolCall(call_id="1", name="record_keyterms", arguments=self._args)
        return CollectedResponse(text="", tool_calls=[call], usage=None, extra={})


class _RecordingLLM(LLM):
    """Fake LLM: returns a `record_keyterms` call per `chat()`, one result tuple per call.

    Subclasses LLM so the detector's ``isinstance(..., LLM)`` gate passes; the last result
    repeats once the sequence is exhausted.
    """

    def __init__(self, *results: tuple[list[str], list[str], list[str]]) -> None:
        super().__init__()
        self._results = list(results) or [([], [], [])]
        self.calls = 0
        self.last_chat_ctx: ChatContext | None = None

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> _FakeStream:  # type: ignore[override]
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        self.last_chat_ctx = chat_ctx
        return _FakeStream(*result)


class _BlockingStream(_FakeStream):
    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__([], [], [])
        self._gate = gate

    async def collect(self) -> CollectedResponse:
        await self._gate.wait()
        return await super().collect()


class _BlockingLLM(LLM):
    """Fake LLM whose response blocks until ``gate`` is set (for single-flight tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.calls = 0

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> _BlockingStream:  # type: ignore[override]
        self.calls += 1
        return _BlockingStream(self.gate)


class _FakeSession(rtc.EventEmitter[str]):
    def __init__(self) -> None:
        super().__init__()
        self.history = ChatContext.empty()

    def add_user(self, text: str) -> None:
        msg = self.history.add_message(role="user", content=text)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=msg))

    def add_assistant(self, text: str) -> None:
        msg = self.history.add_message(role="assistant", content=text)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=msg))


async def _drain(detector: KeytermDetector) -> None:
    await asyncio.sleep(0)
    if detector._detect_task is not None:
        with contextlib.suppress(Exception):  # a failed pass is logged + re-raised on the task
            await detector._detect_task


# -- keyterm state machine (driven through one detection pass each) --


async def test_only_confirmed_terms_are_applied() -> None:
    d = _detector(static_keyterms=["Acme"], llm=_RecordingLLM((["Niamh"], ["Foo"], [])))
    await d._run_once(_ctx())
    # pending terms are tracked but not applied (entries: confirmed then pending)
    assert _entries(d) == [("Foo", True), ("Niamh", False)]
    assert d.keyterms == ["Acme", "Foo"]


async def test_pending_then_confirmed() -> None:
    d = _detector(llm=_RecordingLLM((["Kubernetes"], [], []), ([], ["Kubernetes"], [])))
    await d._run_once(_ctx())
    assert d.keyterms == []
    await d._run_once(_ctx())
    assert _entries(d) == [("Kubernetes", True)]
    assert d.keyterms == ["Kubernetes"]


async def test_static_terms_shown_to_llm_as_applied() -> None:
    fake = _RecordingLLM()
    d = _detector(static_keyterms=["Acme Corp"], llm=fake)
    await d._run_once(_ctx())
    # user terms must appear in the applied list, or the LLM keeps re-proposing them
    assert fake.last_chat_ctx is not None
    user_msg = fake.last_chat_ctx.items[-1].text_content or ""
    applied_section = user_msg.split("## Applied keyterms")[1].splitlines()[1]
    assert "Acme Corp" in applied_section


async def test_user_precedence_and_dedup() -> None:
    d = _detector(
        static_keyterms=["Acme", "Acme", "LiveKit"],
        llm=_RecordingLLM(([], ["LiveKit", "Foo"], [])),
    )
    assert d.static_keyterms == ["Acme", "LiveKit"]
    await d._run_once(_ctx())  # an auto term equal to a user term is dropped
    assert [t for t, _ in _entries(d)] == ["Foo"]
    assert d.keyterms == ["Acme", "LiveKit", "Foo"]


async def test_confirmed_cannot_revert_to_pending() -> None:
    d = _detector(llm=_RecordingLLM(([], ["Niamh"], []), (["Niamh"], [], [])))
    await d._run_once(_ctx())
    assert d.keyterms == ["Niamh"]
    await d._run_once(_ctx())  # a stray `pending` must not reset a confirmed term
    assert _entries(d) == [("Niamh", True)]


async def test_correction_removes_and_replaces() -> None:
    d = _detector(llm=_RecordingLLM((["Jon"], [], []), (["John"], [], ["Jon"]), ([], ["John"], [])))
    await d._run_once(_ctx())
    assert _entries(d) == [("Jon", False)]
    await d._run_once(_ctx())  # misheard spelling removed, corrected one added as pending
    assert _entries(d) == [("John", False)]
    await d._run_once(_ctx())
    assert d.keyterms == ["John"]


async def test_remove_applies_to_confirmed_terms() -> None:
    d = _detector(llm=_RecordingLLM(([], ["Jon"], []), ([], ["John"], ["Jon"])))
    await d._run_once(_ctx())
    assert d.keyterms == ["Jon"]
    await d._run_once(_ctx())  # a user correction can remove an already-applied term
    assert d.keyterms == ["John"]


async def test_remove_unknown_is_noop() -> None:
    d = _detector(llm=_RecordingLLM(([], ["Foo"], []), ([], [], ["does-not-exist"])))
    await d._run_once(_ctx())
    await d._run_once(_ctx())
    assert d.keyterms == ["Foo"]


async def test_cap_evicts_oldest_confirmed() -> None:
    d = _detector(max_keyterms=3, llm=_RecordingLLM(([], ["a", "b", "c", "d", "e"], [])))
    await d._run_once(_ctx())
    assert [t for t, _ in _entries(d)] == ["c", "d", "e"]


async def test_pending_evicted_when_not_confirmed() -> None:
    # pass 1 adds "Tmp" pending; later passes never confirm it, so it ages out
    d = _detector(llm=_RecordingLLM((["Tmp"], [], []), ([], ["Other"], [])))
    await d._run_once(_ctx())
    for _ in range(_PENDING_TTL - 1):
        await d._run_once(_ctx())
    assert "Tmp" in dict(_entries(d))
    await d._run_once(_ctx())  # TTL exceeded
    assert "Tmp" not in dict(_entries(d))


async def test_confirmed_not_evicted_by_staleness() -> None:
    d = _detector(llm=_RecordingLLM(([], ["Keep"], []), (["x"], [], [])))
    await d._run_once(_ctx())
    for _ in range(_PENDING_TTL + 2):
        await d._run_once(_ctx())  # pending churn ages out, but the confirmed term stays
    assert d.keyterms == ["Keep"]


async def test_failed_pass_keeps_state() -> None:
    class _BoomLLM(LLM):
        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> Any:  # type: ignore[override]
            raise RuntimeError("boom")

    d = _detector(llm=_BoomLLM())
    # a failed pass is logged and re-raised on the (fire-and-forget) task; state is untouched
    with contextlib.suppress(RuntimeError):
        await d._run_once(_ctx())
    assert d.keyterms == []


# -- STT binding --


async def test_push_only_on_applied_change() -> None:
    stt = _RecordingSTT()
    session = _FakeSession()
    d = _detector(
        static_keyterms=["Acme"],
        enabled=True,
        llm=_RecordingLLM((["Foo"], [], []), ([], ["Foo"], [])),
    )
    d.start(session, stt=stt)
    assert stt.pushed == [["Acme"]]  # start pushes the current set

    session.add_user("u1")
    await _drain(d)  # pending Foo: tracked, no applied change -> no push
    assert stt.pushed == [["Acme"]]

    session.add_user("u2")
    await _drain(d)  # confirm Foo: push
    assert stt.pushed[-1] == ["Acme", "Foo"]
    await d.aclose()


async def test_start_same_stt_does_not_repush() -> None:
    stt = _RecordingSTT()
    session = _FakeSession()
    d = _detector(static_keyterms=["Acme"], enabled=True, llm=_RecordingLLM())
    d.start(session, stt=stt)
    assert stt.pushed == [["Acme"]]
    await d.aclose()
    # re-binding the same instance on the next activity must not re-push (some STTs reconnect)
    d.start(session, stt=stt)
    assert stt.pushed == [["Acme"]]
    await d.aclose()


async def test_static_terms_pushed_without_detection() -> None:
    stt = _RecordingSTT()
    session = _FakeSession()
    d = _detector(static_keyterms=["Acme"], enabled=False)
    d.start(session, stt=stt)  # detection off must still bind the STT and push
    assert stt.pushed == [["Acme"]]
    d.set_static_keyterms(["New"])
    assert stt.pushed[-1] == ["New"]
    await d.aclose()


async def test_start_without_terms_does_not_push() -> None:
    stt = _RecordingSTT()
    session = _FakeSession()
    d = _detector(enabled=False)
    d.start(session, stt=stt)  # nothing to apply -> no push, no capability warning
    assert stt.pushed == []
    await d.aclose()


async def test_set_static_keyterms_pushes() -> None:
    stt = _RecordingSTT()
    session = _FakeSession()
    d = _detector(enabled=True, llm=_RecordingLLM())
    d.start(session, stt=stt)
    d.set_static_keyterms(["New"])
    assert stt.pushed[-1] == ["New"]
    await d.aclose()


def test_unsupported_stt_warn_and_skip() -> None:
    stt = _RecordingSTT(supports_keyterms=False)
    # exercise the base method (warn-and-skip), not the recorder override
    STT._update_session_keyterms(stt, ["a", "b"])
    assert stt.pushed == []


# -- chat context sink (native carryover) --
# forwarding (subscribe + push every turn) lives in AgentActivity; here we only cover the
# STT sink contract: a supporting STT receives the pushed turns, an unsupported one warns.


def test_push_conversation_item_forwards_to_supporting_stt() -> None:
    stt = _RecordingSTT(supports_chat_context=True)
    user = ConversationItemAddedEvent(item=ChatMessage(role="user", content=["hi"]))
    agent = ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["Welcome"]))
    stt._push_conversation_item(user)  # both user and agent turns are forwarded
    stt._push_conversation_item(agent)
    assert [m.text_content for m in stt.chat_items] == ["hi", "Welcome"]


def test_unsupported_stt_chat_ctx_warn_and_skip() -> None:
    stt = _RecordingSTT(supports_chat_context=False)
    ev = ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["hi"]))
    # exercise the base method (warn-and-skip), not the recorder override
    STT._push_conversation_item(stt, ev)
    assert stt.chat_items == []


# -- triggering --


async def test_triggers_every_n_user_turns() -> None:
    session = _FakeSession()
    fake = _RecordingLLM(([], ["Acme"], []))
    d = _detector(enabled=True, turn_interval=2, llm=fake)
    d.start(session, stt=_RecordingSTT())

    session.add_user("first")  # below interval
    await _drain(d)
    assert fake.calls == 0

    session.add_assistant("ack")  # assistant turns don't advance the counter
    await _drain(d)
    assert fake.calls == 0

    session.add_user("second")  # triggers
    await _drain(d)
    assert fake.calls == 1

    await d.aclose()


async def test_ignores_assistant_messages_for_counting() -> None:
    session = _FakeSession()
    fake = _RecordingLLM()
    d = _detector(enabled=True, llm=fake)
    d.start(session, stt=_RecordingSTT())

    session.add_assistant("hello")
    await _drain(d)
    assert fake.calls == 0

    session.add_user("hi")
    await _drain(d)
    assert fake.calls == 1

    await d.aclose()


async def test_empty_user_turn_does_not_trigger() -> None:
    session = _FakeSession()
    fake = _RecordingLLM()
    d = _detector(enabled=True, llm=fake)
    d.start(session, stt=_RecordingSTT())

    session.add_user("")
    await _drain(d)
    assert fake.calls == 0

    await d.aclose()


async def test_single_flight_skips_overlapping_pass() -> None:
    session = _FakeSession()
    fake = _BlockingLLM()
    d = _detector(enabled=True, llm=fake)
    d.start(session, stt=_RecordingSTT())

    session.add_user("first")
    await asyncio.sleep(0)
    assert fake.calls == 1

    session.add_user("second")  # a pass is still in flight -> skipped, not queued
    await asyncio.sleep(0)
    assert fake.calls == 1

    fake.gate.set()
    await _drain(d)
    assert fake.calls == 1
    await d.aclose()


async def test_aclose_unsubscribes() -> None:
    session = _FakeSession()
    fake = _RecordingLLM()
    d = _detector(enabled=True, llm=fake)
    d.start(session, stt=_RecordingSTT())
    await d.aclose()

    session.add_user("hi")
    await asyncio.sleep(0)
    assert fake.calls == 0


async def test_disabled_detection_does_not_trigger() -> None:
    session = _FakeSession()
    fake = _RecordingLLM(([], ["Acme"], []))
    d = _detector(enabled=False, llm=fake)
    d.start(session, stt=_RecordingSTT())

    session.add_user("the Acme Grand")
    await _drain(d)
    assert fake.calls == 0
    assert d.keyterms == []


async def test_unsupported_stt_skips_detection() -> None:
    # no point running LLM detection passes when the STT can't consume the keyterms
    session = _FakeSession()
    fake = _RecordingLLM(([], ["Acme"], []))
    d = _detector(enabled=True, llm=fake)
    d.start(session, stt=_RecordingSTT(supports_keyterms=False))

    session.add_user("the Acme Grand")
    await _drain(d)
    assert fake.calls == 0


# -- module helpers --


async def test_detect_keyterms_parses_result() -> None:
    llm = _RecordingLLM(([], ["Niamh"], ["Jon"]))
    pending, confirm, remove = await _detect_keyterms(llm, _ctx("It's Niamh"), current_keyterms=[])
    assert pending == []
    assert confirm == ["Niamh"]
    assert remove == ["Jon"]
    # no transcript -> no LLM call, empty result
    assert await _detect_keyterms(llm, ChatContext.empty()) == ([], [], [])


def test_parse_tool_call() -> None:
    call = FunctionToolCall(
        call_id="1",
        name="record_keyterms",
        arguments=json.dumps(
            {
                "pending": ["John", "  ", 5],  # blanks and non-strings are dropped
                "confirm": ["Foo"],
                "remove": ["Jon"],
            }
        ),
    )
    pending, confirm, remove = _parse_tool_call([call])
    assert pending == ["John"]
    assert confirm == ["Foo"]
    assert remove == ["Jon"]


def test_parse_tool_call_missing() -> None:
    assert _parse_tool_call([]) == ([], [], [])
    bad = FunctionToolCall(call_id="1", name="record_keyterms", arguments="not json")
    assert _parse_tool_call([bad]) == ([], [], [])


def test_format_input_splits_applied_and_candidate() -> None:
    text = _format_input(_ctx("hi"), [("Term1", True), ("Term2", False)])
    assert text is not None
    assert "Applied keyterms" in text and "Term1" in text
    assert "Candidate keyterms" in text and "Term2" in text
    assert "record_keyterms" in text  # trailing instruction
    # no transcript yet -> nothing to send
    assert _format_input(ChatContext.empty(), []) is None


def test_resolve_detection() -> None:
    assert _resolve_detection(None)["enabled"] is False
    assert _resolve_detection({"enabled": False})["enabled"] is False

    resolved = _resolve_detection({"enabled": True})
    assert resolved["enabled"] is True
    assert resolved["turn_interval"] == 1
    assert resolved["max_keyterms"] is None
