"""An interim transcript after a preflight one stops a preemptive reply that went stale.

Turn-detecting STTs send PREFLIGHT_TRANSCRIPT when the turn may be over (Deepgram Flux
``EagerEndOfTurn``, Cartesia ``turn.eager_end``) and an interim when the user keeps talking
(Flux ``TurnResumed``, Cartesia ``turn.resume``). Once the transcript has moved on, the reply
started for the preflight can no longer be adopted, so it should stop right away instead of
generating until the next preflight or the final transcript.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, LanguageCode, TurnHandlingOptions
from livekit.agents.llm import LLM, ChatChunk, ChatContext, ChoiceDelta, LLMStream
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .fake_turn_stt import TurnScriptedSTT

pytestmark = pytest.mark.unit

TIMEOUT = 2.0


class _OpenStream(LLMStream):
    """Streams one token, then stays open like a provider that is still generating."""

    def __init__(self, llm: _OpenLLM, **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._owner = llm

    async def _run(self) -> None:
        self._owner.started.set()
        self._event_ch.send_nowait(
            ChatChunk(id="chunk", delta=ChoiceDelta(role="assistant", content="sure, one"))
        )
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._owner.cancelled.set()
            raise


class _OpenLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    @property
    def model(self) -> str:
        return "open"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _OpenStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def _event(type: SpeechEventType, text: str) -> SpeechEvent:
    return SpeechEvent(
        type=type,
        alternatives=[SpeechData(text=text, language=LanguageCode("en"), confidence=0.9)],
    )


async def _start_session() -> tuple[_OpenLLM, AgentSession[None]]:
    llm = _OpenLLM()
    session: AgentSession[None] = AgentSession(
        stt=TurnScriptedSTT(turns=[]),
        vad=None,
        llm=llm,
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
    )
    await session.start(agent=Agent(instructions="help the caller"))
    return llm, session


async def test_resumed_turn_cancels_the_preemptive_reply() -> None:
    """eager("...subscription.") -> resumed("...subscription please"), before any final."""
    llm, session = await _start_session()
    activity = session._activity
    assert activity is not None and activity._audio_recognition is not None
    recognition = activity._audio_recognition

    try:
        await recognition._on_stt_event(
            _event(SpeechEventType.PREFLIGHT_TRANSCRIPT, "I need to cancel my subscription.")
        )
        assert activity._preemptive_generation is not None
        handle = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(llm.started.wait(), TIMEOUT)

        await recognition._on_stt_event(
            _event(SpeechEventType.INTERIM_TRANSCRIPT, "I need to cancel my subscription please")
        )

        # dropped on the event itself, not at the next preflight or final transcript
        assert activity._preemptive_generation is None
        assert handle.interrupted
        await asyncio.wait_for(llm.cancelled.wait(), TIMEOUT)
        await asyncio.wait_for(handle.wait_for_playout(), TIMEOUT)

        # later interims find nothing to cancel
        await recognition._on_stt_event(
            _event(
                SpeechEventType.INTERIM_TRANSCRIPT, "I need to cancel my subscription please and"
            )
        )
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)


@pytest.mark.parametrize(
    "interim",
    [
        # Flux keeps sending `Update` after `EagerEndOfTurn`, with the same transcript
        pytest.param("Hello, how are you?", id="same-transcript"),
        # AssemblyAI interims join the raw words, its preflight is the formatted utterance
        pytest.param("hello how are you", id="same-words"),
        # carries nothing the user said
        pytest.param("", id="empty"),
    ],
)
async def test_interim_without_new_words_keeps_the_preemptive_reply(interim: str) -> None:
    llm, session = await _start_session()
    activity = session._activity
    assert activity is not None and activity._audio_recognition is not None
    recognition = activity._audio_recognition

    try:
        await recognition._on_stt_event(
            _event(SpeechEventType.PREFLIGHT_TRANSCRIPT, "Hello, how are you?")
        )
        assert activity._preemptive_generation is not None
        handle = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(llm.started.wait(), TIMEOUT)

        await recognition._on_stt_event(_event(SpeechEventType.INTERIM_TRANSCRIPT, interim))

        # the reply still matches what the user said, so the final transcript can adopt it
        assert activity._preemptive_generation is not None
        assert activity._preemptive_generation.speech_handle is handle
        assert not handle.interrupted
    finally:
        await asyncio.gather(session.aclose(), return_exceptions=True)
