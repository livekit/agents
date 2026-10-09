"""Accepted transcript provenance through the real AgentSession user-turn hook.

Only synthetic provider events and in-process session components are used.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest

from livekit.agents import Agent, AgentSession, LanguageCode
from livekit.agents.llm import LLM, ChatContext, ChatMessage
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
    _TranscriptSource,
)

from .fake_io import FakeAudioInput
from .fake_stt import FakeSTT
from .test_preemptive_pause_deadlock import _GatedLLM

pytestmark = pytest.mark.unit


class CapturingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test only")
        self.messages: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self.gate: asyncio.Event | None = None

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        self.messages.put_nowait(new_message.model_copy(deep=True))
        if self.gate is not None:
            await self.gate.wait()
        new_message.extra["application_marker"] = "hook ran"


@asynccontextmanager
async def running_session(
    llm: LLM | None = None,
) -> AsyncIterator[tuple[AgentSession, CapturingAgent, FakeSTT]]:
    provider = FakeSTT()
    session = AgentSession(
        stt=provider,
        llm=llm,
        aec_warmup_duration=None,
        turn_handling={
            "turn_detection": "manual",
            "endpointing": {"min_delay": 0.0, "max_delay": 0.0},
        },
    )
    audio = FakeAudioInput()
    session.input.audio = audio
    agent = CapturingAgent()
    await session.start(agent)
    audio.push(0.01)
    try:
        yield session, agent, provider
    finally:
        if agent.gate is not None:
            agent.gate.set()
        if isinstance(llm, _GatedLLM):
            llm.release()
        await session.aclose()


async def wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("session condition did not complete")


def event(
    text: str,
    request_id: str,
    kind: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT,
) -> SpeechEvent:
    return SpeechEvent(
        type=kind,
        request_id=request_id,
        alternatives=[SpeechData(text=text, language=LanguageCode(""))],
    )


async def commit(session: AgentSession, agent: CapturingAgent) -> ChatMessage:
    await asyncio.wait_for(
        session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0), 1.0
    )
    return await asyncio.wait_for(agent.messages.get(), 1.0)


async def test_same_text_in_repeated_turns_has_distinct_exact_ids() -> None:
    async with running_session() as (session, agent, provider):
        stream = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        for request_id in ("first", "second"):
            stream._event_ch.send_nowait(event("same words", request_id))
            await wait_until(lambda: recognition._audio_transcript == "same words")
            message = await commit(session, agent)
            assert message.raw_text_content == "same words"
            assert message.extra == {
                "stt_request_ids": [request_id],
                "stt_request_ids_complete": True,
            }
            assert recognition._transcript_request_ids == []


async def test_multiple_finals_merge_ids_in_order_and_deduplicate_within_turn() -> None:
    async with running_session() as (session, agent, provider):
        stream = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        stream._event_ch.send_nowait(
            event("speculation", "unused", SpeechEventType.INTERIM_TRANSCRIPT)
        )
        for text, request_id in (("one", "a"), ("two", "b"), ("three", "a")):
            stream._event_ch.send_nowait(event(text, request_id))
        await wait_until(lambda: recognition._audio_transcript == "one two three")
        message = await commit(session, agent)
        assert message.extra["stt_request_ids"] == ["a", "b"]
        assert message.extra["stt_request_ids_complete"] is True


async def test_clear_discards_ids_and_incomplete_state() -> None:
    async with running_session() as (session, agent, provider):
        stream = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        assert session._activity is not None
        recognition = session._activity._audio_recognition
        assert recognition is not None
        stream._event_ch.send_nowait(event("discard", ""))
        await wait_until(lambda: recognition._audio_transcript == "discard")
        session.clear_user_turn()
        replacement = await asyncio.wait_for(provider.stream_ch.recv(), 1.0)
        replacement._event_ch.send_nowait(event("keep", "new"))
        await wait_until(lambda: recognition._audio_transcript == "keep")
        message = await commit(session, agent)
        assert message.extra["stt_request_ids"] == ["new"]
        assert message.extra["stt_request_ids_complete"] is True


async def test_rejected_endpoint_keeps_provenance_for_later_merged_turn(monkeypatch) -> None:
    async with running_session() as (session, agent, _):
        activity = session._activity
        recognition = activity._audio_recognition
        original = activity.on_end_of_turn
        attempts = []

        def reject_once(info):
            attempts.append(info)
            return False if len(attempts) == 1 else original(info)

        monkeypatch.setattr(activity, "on_end_of_turn", reject_once)
        await recognition._on_stt_event(event("first", "a"))
        await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0)
        await recognition._end_of_turn_task
        assert agent.messages.empty()
        assert recognition._audio_transcript == "first"
        await recognition._on_stt_event(event("continued", "b"))
        message = await commit(session, agent)
        assert message.raw_text_content == "first continued"
        assert message.extra["stt_request_ids"] == ["a", "b"]
        assert attempts[0].transcript_source.request_ids == ("a",)


@pytest.mark.parametrize("bad_id", ["", "x" * 257])
async def test_missing_or_overlong_id_marks_coverage_incomplete(bad_id: str) -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(event("known", "good"))
        await recognition._on_stt_event(event("unknown", bad_id))
        message = await commit(session, agent)
        assert message.extra["stt_request_ids"] == ["good"]
        assert message.extra["stt_request_ids_complete"] is False


async def test_promoted_interim_is_not_mislabelled_finalized_identity() -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(event("known", "final"))
        await recognition._on_stt_event(
            event("unfinished", "interim", SpeechEventType.INTERIM_TRANSCRIPT)
        )
        message = await commit(session, agent)
        assert message.raw_text_content == "known unfinished"
        assert message.extra["stt_request_ids"] == ["final"]
        assert message.extra["stt_request_ids_complete"] is False


async def test_preflight_correction_uses_final_id_only() -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(
            event("wrong", "draft", SpeechEventType.PREFLIGHT_TRANSCRIPT)
        )
        await recognition._on_stt_event(event("corrected", "final"))
        message = await commit(session, agent)
        assert message.raw_text_content == "corrected"
        assert message.extra["stt_request_ids"] == ["final"]
        assert message.extra["stt_request_ids_complete"] is True


async def test_held_then_trimmed_transcript_cannot_leak_identity() -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        recognition._transcript_gate_active = True
        discarded = event("discard", "stale")
        discarded.created_at = 1.0
        kept = event("keep", "retained")
        kept.created_at = 10.0
        await recognition._on_stt_event(discarded)
        await recognition._on_stt_event(kept)
        assert recognition._transcript_request_ids == []
        recognition._flush_held_transcripts(resolved_at=5.0)
        message = await commit(session, agent)
        assert message.raw_text_content == "keep"
        assert message.extra["stt_request_ids"] == ["retained"]


async def test_identity_overflow_is_bounded_and_explicit() -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        for index in range(129):
            await recognition._on_stt_event(event("word", str(index)))
        message = await commit(session, agent)
        assert message.extra["stt_request_ids"] == [str(index) for index in range(128)]
        assert message.extra["stt_request_ids_complete"] is False


async def test_next_turn_during_slow_hook_cannot_mutate_prior_snapshot() -> None:
    async with running_session() as (session, agent, _):
        recognition = session._activity._audio_recognition
        agent.gate = asyncio.Event()
        await recognition._on_stt_event(event("repeat", "first"))
        first = await commit(session, agent)
        await recognition._on_stt_event(event("repeat", "second"))
        await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0)
        assert agent.messages.empty()  # the real activity serializes the user hooks
        assert first.extra["stt_request_ids"] == ["first"]
        agent.gate.set()
        second = await asyncio.wait_for(agent.messages.get(), 1.0)
        assert second.extra["stt_request_ids"] == ["second"]


async def test_interruption_of_reply_does_not_carry_prior_turn_ids() -> None:
    llm = _GatedLLM()
    try:
        async with running_session(llm) as (session, agent, _):
            recognition = session._activity._audio_recognition
            await recognition._on_stt_event(event("first turn", "first"))
            first = await commit(session, agent)
            await wait_until(lambda: session.current_speech is not None)
            first_reply = session.current_speech
            await recognition._on_stt_event(event("second turn", "second"))
            second = await commit(session, agent)
            assert first_reply is not None and first_reply.interrupted
            assert first.extra["stt_request_ids"] == ["first"]
            assert second.extra["stt_request_ids"] == ["second"]
    finally:
        llm.release()


async def test_preemptive_reuse_preserves_final_ids_and_hook_extra_edits() -> None:
    llm = _GatedLLM()
    session = AgentSession(llm=llm)
    agent = CapturingAgent()
    await session.start(agent)
    activity = session._activity
    assert activity is not None
    history: list[ChatMessage] = []
    session.on("conversation_item_added", lambda ev: history.append(ev.item))
    try:
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="same",
                transcript_confidence=0.0,
                started_speaking_at=None,
                transcript_source=_TranscriptSource(("old-speculative",), False),
            )
        )
        preemptive = activity._preemptive_generation
        assert preemptive is not None
        activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="same",
                transcript_confidence=0.0,
                metrics=_EndOfTurnMetrics(None, None, None, None),
                transcript_source=_TranscriptSource(("final",), True),
            )
        )
        await asyncio.wait_for(agent.messages.get(), 1.0)
        await wait_until(lambda: "application_marker" in preemptive.user_message.extra)
        assert preemptive.user_message.extra == {
            "stt_request_ids": ["final"],
            "stt_request_ids_complete": True,
            "application_marker": "hook ran",
        }
        llm.release()
        await wait_until(lambda: any(item.role == "user" for item in history))
        persisted = next(item for item in history if item.role == "user")
        assert persisted.extra == preemptive.user_message.extra
    finally:
        llm.release()
        await session.aclose()
