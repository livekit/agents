from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from livekit import rtc
from livekit.agents.llm import ChatContext, CollectedResponse, FunctionToolCall
from livekit.agents.stt import STT, RecognizeStream, SpeechEvent, STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.keyterms import (
    DEFAULT_MAX_KEYTERMS,
    DEFAULT_TURN_INTERVAL,
    KeytermAnalyzer,
    KeytermManager,
    _parse_tool_call,
    _resolve_detection,
)

pytestmark = pytest.mark.unit


class _RecordingSTT(STT):
    """STT that records every update_keyterms() call."""

    def __init__(self, *, supports_keyterms: bool = True) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True, interim_results=False, keyterms=supports_keyterms
            )
        )
        self.pushed: list[list[str]] = []

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

    def update_keyterms(self, keyterms: list[str]) -> None:
        self.pushed.append(list(keyterms))


# -- KeytermManager: confirmation gating --


def test_only_confirmed_terms_are_applied() -> None:
    m = KeytermManager(user_keyterms=["Acme"])
    m.apply_detection([("Niamh", "unconfirmed"), ("Foo", "implicit")])
    # unconfirmed terms are tracked but not applied to the STT
    assert m.auto_entries == [("Niamh", "unconfirmed"), ("Foo", "implicit")]
    assert m.keyterms == ["Acme", "Foo"]

    # confirming an existing term applies it (auto insertion order preserved)
    m.apply_detection([("Niamh", "explicit")])
    assert m.keyterms == ["Acme", "Niamh", "Foo"]


def test_user_precedence_and_dedup() -> None:
    m = KeytermManager(user_keyterms=["Acme", "Acme", "LiveKit"])
    assert m.user_keyterms == ["Acme", "LiveKit"]
    # an auto term equal to a user term is dropped
    m.apply_detection([("LiveKit", "explicit"), ("Foo", "explicit")])
    assert [t for t, _ in m.auto_entries] == ["Foo"]
    assert m.keyterms == ["Acme", "LiveKit", "Foo"]


def test_user_set_immutable_by_auto() -> None:
    m = KeytermManager(user_keyterms=["Alice"])
    m.apply_detection([("Alice", "explicit"), ("Bob", "explicit")])
    assert m.user_keyterms == ["Alice"]
    assert "Alice" in m.keyterms


def test_correction_adds_confirmed_spelling() -> None:
    m = KeytermManager()
    # the misheard spelling is only ever unconfirmed, so it is never applied
    m.apply_detection([("Jon", "unconfirmed")])
    assert m.keyterms == []
    # the user's correction is recorded as confirmed and applied
    m.apply_detection([("John", "explicit")])
    assert m.keyterms == ["John"]


def test_status_upgrade_in_place() -> None:
    m = KeytermManager()
    m.apply_detection([("Kubernetes", "unconfirmed")])
    assert m.keyterms == []
    m.apply_detection([("Kubernetes", "explicit")])
    assert m.auto_entries == [("Kubernetes", "explicit")]
    assert m.keyterms == ["Kubernetes"]


def test_cap_fifo_eviction() -> None:
    m = KeytermManager(max_keyterms=3)
    m.apply_detection([(t, "explicit") for t in ["a", "b", "c", "d", "e"]])
    assert [t for t, _ in m.auto_entries] == ["c", "d", "e"]


def test_apply_detection_returns_applied_change() -> None:
    m = KeytermManager()
    # adding an unconfirmed term does not change the applied set
    assert m.apply_detection([("x", "unconfirmed")]) is False
    # confirming it does
    assert m.apply_detection([("x", "explicit")]) is True


def test_push_only_on_applied_change() -> None:
    stt = _RecordingSTT()
    m = KeytermManager(user_keyterms=["Acme"])
    m.attach_stt(stt)
    assert stt.pushed == [["Acme"]]

    # unconfirmed term: tracked, but no push (applied set unchanged)
    before = len(stt.pushed)
    m.apply_detection([("Foo", "unconfirmed")])
    assert len(stt.pushed) == before

    # confirm it: push
    m.apply_detection([("Foo", "explicit")])
    assert stt.pushed[-1] == ["Acme", "Foo"]


def test_set_user_keyterms_pushes() -> None:
    stt = _RecordingSTT()
    m = KeytermManager()
    m.attach_stt(stt)
    m.set_user_keyterms(["New"])
    assert stt.pushed[-1] == ["New"]


def test_unsupported_stt_warn_and_skip() -> None:
    stt = _RecordingSTT(supports_keyterms=False)
    # exercise the base method (warn-and-skip), not the recorder override
    STT.update_keyterms(stt, ["a", "b"])
    assert stt.pushed == []


# -- tool-call parsing & config resolution --


def test_parse_tool_call() -> None:
    call = FunctionToolCall(
        call_id="1",
        name="record_keyterms",
        arguments=json.dumps(
            {
                "keyterms": [
                    {"term": "John", "confirmation": "explicit"},
                    {"term": "x", "confirmation": "bogus"},  # invalid -> unconfirmed
                    {"confirmation": "explicit"},  # missing term -> skipped
                ],
            }
        ),
    )
    assert _parse_tool_call([call]) == [("John", "explicit"), ("x", "unconfirmed")]


def test_parse_tool_call_missing() -> None:
    assert _parse_tool_call([]) == []
    bad = FunctionToolCall(call_id="1", name="record_keyterms", arguments="not json")
    assert _parse_tool_call([bad]) == []


def test_resolve_detection() -> None:
    # always returns a fully-defaulted dict; `enabled` reflects the input
    assert _resolve_detection(None)["enabled"] is False
    assert _resolve_detection({"enabled": False})["enabled"] is False

    resolved = _resolve_detection({"enabled": True})
    assert resolved["enabled"] is True
    assert resolved["turn_interval"] == DEFAULT_TURN_INTERVAL
    assert resolved["max_keyterms"] == DEFAULT_MAX_KEYTERMS


# -- KeytermAnalyzer (Layer 2) --


class _FakeStream:
    def __init__(self, keyterms: list[dict[str, str]]) -> None:
        self._args = json.dumps({"keyterms": keyterms})

    async def collect(self) -> CollectedResponse:
        call = FunctionToolCall(call_id="1", name="record_keyterms", arguments=self._args)
        return CollectedResponse(text="", tool_calls=[call], usage=None, extra={})


class _FakeLLM:
    def __init__(self, keyterms: list[dict[str, str]] | None = None):
        self._keyterms = keyterms or []
        self.calls = 0

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> _FakeStream:
        self.calls += 1
        return _FakeStream(self._keyterms)


class _FakeSession(rtc.EventEmitter[str]):
    def __init__(self) -> None:
        super().__init__()
        self.history = ChatContext.empty()
        self._keyterm_manager = KeytermManager()

    def add_user(self, text: str) -> None:
        msg = self.history.add_message(role="user", content=text)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=msg))

    def add_assistant(self, text: str) -> None:
        msg = self.history.add_message(role="assistant", content=text)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=msg))


async def _drain(analyzer: KeytermAnalyzer) -> None:
    await asyncio.sleep(0)
    if analyzer._task is not None:
        await analyzer._task


async def test_analyzer_applies_confirmed_terms() -> None:
    session = _FakeSession()
    fake_llm = _FakeLLM(keyterms=[{"term": "Acme Grand", "confirmation": "explicit"}])
    analyzer = KeytermAnalyzer(fake_llm, turn_interval=1)
    analyzer.start(session)

    session.add_user("I'd like to book at the Acme Grand")
    await _drain(analyzer)
    assert fake_llm.calls == 1
    assert session._keyterm_manager.keyterms == ["Acme Grand"]

    await analyzer.aclose()


async def test_analyzer_holds_unconfirmed_terms() -> None:
    session = _FakeSession()
    fake_llm = _FakeLLM(keyterms=[{"term": "Flandor", "confirmation": "unconfirmed"}])
    analyzer = KeytermAnalyzer(fake_llm, turn_interval=1)
    analyzer.start(session)

    session.add_user("It crashes in the Flandor module")
    await _drain(analyzer)
    # tracked but not applied
    assert session._keyterm_manager.auto_entries == [("Flandor", "unconfirmed")]
    assert session._keyterm_manager.keyterms == []

    await analyzer.aclose()


async def test_detect_without_session() -> None:
    fake_llm = _FakeLLM(keyterms=[{"term": "Niamh", "confirmation": "explicit"}])
    analyzer = KeytermAnalyzer(fake_llm)
    result = await analyzer.detect(transcript="user: It's Niamh", current_keyterms=[])
    assert result == [("Niamh", "explicit")]


async def test_analyzer_triggers_every_n_user_turns() -> None:
    session = _FakeSession()
    fake_llm = _FakeLLM(keyterms=[{"term": "Acme", "confirmation": "explicit"}])
    analyzer = KeytermAnalyzer(fake_llm, turn_interval=2)
    analyzer.start(session)

    session.add_user("first")  # below interval
    await _drain(analyzer)
    assert fake_llm.calls == 0

    session.add_assistant("ack")  # agent turns don't advance the counter
    await _drain(analyzer)
    assert fake_llm.calls == 0

    session.add_user("second")  # triggers
    await _drain(analyzer)
    assert fake_llm.calls == 1

    await analyzer.aclose()


async def test_analyzer_ignores_assistant_messages_for_counting() -> None:
    session = _FakeSession()
    fake_llm = _FakeLLM()
    analyzer = KeytermAnalyzer(fake_llm, turn_interval=1)
    analyzer.start(session)

    session.add_assistant("hi")
    await _drain(analyzer)
    assert fake_llm.calls == 0

    session.add_user("hello")
    await _drain(analyzer)
    assert fake_llm.calls == 1

    await analyzer.aclose()


async def test_analyzer_stops_on_aclose() -> None:
    session = _FakeSession()
    fake_llm = _FakeLLM()
    analyzer = KeytermAnalyzer(fake_llm, turn_interval=1)
    analyzer.start(session)
    await analyzer.aclose()

    session.add_user("hi")
    await asyncio.sleep(0)
    assert fake_llm.calls == 0


async def test_analyzer_swallows_llm_errors() -> None:
    session = _FakeSession()

    class _BoomLLM:
        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    analyzer = KeytermAnalyzer(_BoomLLM(), turn_interval=1)
    analyzer.start(session)

    session.add_user("hi")
    await _drain(analyzer)  # must not raise
    assert session._keyterm_manager.keyterms == []

    await analyzer.aclose()
