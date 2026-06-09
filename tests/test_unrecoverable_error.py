"""Tests for how AgentSession surfaces unrecoverable errors to the end user.

A terminal quota error (out of LiveKit Inference credits) must:
- close the session on the FIRST occurrence rather than after
  ``max_unrecoverable_errors`` silent dead turns, and
- speak a perceptible fallback message (the gateway ``hint`` by default) before
  closing, so the agent never goes silently unresponsive.

See https://github.com/livekit/agents/issues/6009
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from livekit.agents import Agent, APIConnectOptions, APIError, APIQuotaExceededError
from livekit.agents.llm import LLMError, RealtimeModelError
from livekit.agents.tts import TTS
from livekit.agents.voice.agent_session import (
    DEFAULT_ERROR_MESSAGE,
    AgentSession,
    SessionConnectOptions,
)
from livekit.agents.voice.events import CloseReason

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit

INFERENCE_QUOTA_BODY = {
    "type": "inference_quota_exceeded",
    "hint": "You're out of credits. Upgrade your plan or wait for the next cycle.",
    "quota_type": "llm",
    "category": "MaxGatewayCredits",
    "remaining_limit": "0",
}


class _Agent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _make_session(*, tts: TTS | None = None, **kwargs: Any) -> AgentSession:
    session = AgentSession(
        stt=FakeSTT(),
        vad=FakeVAD(),
        llm=FakeLLM(),
        tts=tts or FakeTTS(fake_audio_duration=0.05),
        aec_warmup_duration=None,
        **kwargs,
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session


def _llm_error(exc: Exception) -> LLMError:
    return LLMError(timestamp=time.time(), label="test-llm", error=exc, recoverable=False)


def _realtime_error(exc: Exception) -> RealtimeModelError:
    return RealtimeModelError(timestamp=time.time(), label="test-rt", error=exc, recoverable=False)


def _quota_error(*, hint: str | None = INFERENCE_QUOTA_BODY["hint"]) -> APIQuotaExceededError:
    body = {**INFERENCE_QUOTA_BODY, "hint": hint}
    return APIQuotaExceededError("LLM token credit quota exceeded", status_code=429, body=body)


def _rate_limit_error() -> APIQuotaExceededError:
    # same `type`, but a transient rate-limit category -> non-terminal, recoverable
    body = {
        "type": "inference_quota_exceeded",
        "hint": "LLM request rate limit reached. Reduce request rate or upgrade your plan.",
        "quota_type": "llm",
        "category": "MaxConcurrentGatewayLLMRpm",
        "remaining_limit": "0",
    }
    return APIQuotaExceededError("rate limited", status_code=429, body=body)


def _spy_say(session: AgentSession) -> list[str]:
    """Record the text passed to ``session.say`` while still calling through."""
    spoken: list[str] = []
    original_say = session.say

    def spy(text: Any, **kwargs: Any) -> Any:
        spoken.append(text)
        return original_say(text, **kwargs)

    session.say = spy  # type: ignore[method-assign]
    return spoken


async def test_quota_error_closes_on_first_occurrence() -> None:
    # error_message=None isolates the close behavior from the spoken-message path
    session = _make_session(error_message=None)
    await session.start(_Agent())

    close_events: list = []
    session.on("close", close_events.append)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None, "a terminal quota error must close on the first occurrence"
    await closing_task

    assert len(close_events) == 1
    assert close_events[0].reason == CloseReason.ERROR
    assert isinstance(close_events[0].error.error, APIQuotaExceededError)

    await session.aclose()


async def test_generic_unrecoverable_error_is_tolerated() -> None:
    # a non-terminal unrecoverable error is still absorbed up to max_unrecoverable_errors,
    # so a single blip does not close the session
    session = _make_session()
    await session.start(_Agent())

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(APIError("transient blip")))

    assert session._closing_task is None
    assert session._llm_error_counts == 1

    await session.aclose()


async def test_quota_error_speaks_hint_by_default() -> None:
    session = _make_session()  # error_message defaults to NOT_GIVEN
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error(hint="You are out of credits.")))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == ["You are out of credits."]

    await session.aclose()


async def test_quota_error_speaks_generic_message_without_hint() -> None:
    session = _make_session()
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error(hint=None)))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == [DEFAULT_ERROR_MESSAGE]

    await session.aclose()


async def test_error_message_none_disables_spoken_fallback() -> None:
    session = _make_session(error_message=None)
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == []

    await session.aclose()


async def test_custom_error_message_spoken_on_unrecoverable_error() -> None:
    # a configured error_message is spoken for any unrecoverable error, not just quota;
    # max_unrecoverable_errors=0 forces the close on the first occurrence
    session = _make_session(
        error_message="Goodbye for now.",
        conn_options=SessionConnectOptions(max_unrecoverable_errors=0),
    )
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(APIError("boom")))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == ["Goodbye for now."]

    await session.aclose()


async def test_custom_error_message_overrides_quota_hint() -> None:
    # an explicit error_message wins over the gateway hint even for quota errors
    session = _make_session(error_message="Custom branded message.")
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error(hint="Gateway hint.")))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == ["Custom branded message."]

    await session.aclose()


async def test_rate_limit_quota_error_is_tolerated_not_terminal() -> None:
    # a transient rate-limit quota error must NOT close on the first occurrence and
    # must NOT speak "out of credits" — it falls through the normal tolerance
    session = _make_session()
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_rate_limit_error()))

    assert session._closing_task is None
    assert session._llm_error_counts == 1
    assert spoken == []

    await session.aclose()


async def test_realtime_quota_error_speaks_and_closes_on_first_occurrence() -> None:
    # the realtime path routes through the same _on_error mechanism
    session = _make_session()
    await session.start(_Agent())

    spoken = _spy_say(session)
    close_events: list = []
    session.on("close", close_events.append)

    activity = session._activity
    assert activity is not None
    activity._on_error(_realtime_error(_quota_error(hint="Out of credits.")))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == ["Out of credits."]
    assert close_events[0].reason == CloseReason.ERROR

    await session.aclose()


async def test_speaking_error_message_survives_tts_failure() -> None:
    # best-effort: if the TTS fails while speaking the fallback, the session must still
    # close cleanly and emit the close event (never strand the session open)
    session = _make_session(
        tts=FakeTTS(fake_audio_duration=0.05, fake_exception=APIError("tts down")),
        # fail fast so the deliberate TTS error isn't retried with backoff
        conn_options=SessionConnectOptions(
            tts_conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0)
        ),
    )
    await session.start(_Agent())

    close_events: list = []
    session.on("close", close_events.append)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert len(close_events) == 1
    assert close_events[0].reason == CloseReason.ERROR

    await session.aclose()


async def test_no_spoken_fallback_without_audio_output() -> None:
    # with no audio output sink, speaking is skipped but the session still closes
    session = _make_session()
    session.output.audio = None
    await session.start(_Agent())

    spoken = _spy_say(session)
    close_events: list = []
    session.on("close", close_events.append)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == []
    assert close_events[0].reason == CloseReason.ERROR

    await session.aclose()
