"""Tests for how AgentSession surfaces unrecoverable errors to the end user.

A terminal quota error (out of LiveKit Inference credits) must:
- close the session on the FIRST occurrence rather than after
  ``max_unrecoverable_errors`` silent dead turns, and
- speak a generic, provider-agnostic fallback message before closing, so the
  agent never goes silently unresponsive (the gateway ``hint`` is never spoken —
  quota details aren't surfaced to end users).

See https://github.com/livekit/agents/issues/6009
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    APIConnectOptions,
    APIError,
    BuiltinAudioClip,
)
from livekit.agents.inference import InferenceQuotaExceededError
from livekit.agents.llm import LLMError, RealtimeModelError
from livekit.agents.tts import TTS
from livekit.agents.voice.agent_session import (
    DEFAULT_UNRECOVERABLE_ERROR_MESSAGE,
    AgentSession,
    SessionConnectOptions,
)
from livekit.agents.voice.events import CloseReason, ErrorEvent

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


def _quota_error(*, hint: str | None = INFERENCE_QUOTA_BODY["hint"]) -> InferenceQuotaExceededError:
    body = {**INFERENCE_QUOTA_BODY, "hint": hint}
    return InferenceQuotaExceededError(
        "LLM token credit quota exceeded", status_code=429, body=body
    )


def _rate_limit_error() -> InferenceQuotaExceededError:
    # same `type`, but a transient rate-limit category -> non-terminal, recoverable
    body = {
        "type": "inference_quota_exceeded",
        "hint": "LLM request rate limit reached. Reduce request rate or upgrade your plan.",
        "quota_type": "llm",
        "category": "MaxConcurrentGatewayLLMRpm",
        "remaining_limit": "0",
    }
    return InferenceQuotaExceededError("rate limited", status_code=429, body=body)


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
    # unrecoverable_error_message=None isolates the close behavior from the spoken-message path
    session = _make_session(unrecoverable_error_message=None)
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
    assert isinstance(close_events[0].error.error, InferenceQuotaExceededError)

    await session.aclose()


async def test_user_error_handler_runs_before_close_decision() -> None:
    session = _make_session()
    await session.start(_Agent())

    @session.on("error")
    def _keep_alive(ev: ErrorEvent) -> None:
        ev.error.recoverable = True

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))  # terminal: would close without the flip

    assert session._closing_task is None

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


@pytest.mark.parametrize("hint", ["You are out of credits.", None])
async def test_quota_error_speaks_generic_message_by_default(hint: str | None) -> None:
    # the gateway hint is never spoken — a terminal quota error always falls back to the
    # generic, provider-agnostic message regardless of what the gateway sent.
    session = _make_session()  # unrecoverable_error_message defaults to NOT_GIVEN
    await session.start(_Agent())

    spoken = _spy_say(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error(hint=hint)))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert spoken == [DEFAULT_UNRECOVERABLE_ERROR_MESSAGE]

    await session.aclose()


async def test_error_message_none_disables_spoken_fallback() -> None:
    session = _make_session(unrecoverable_error_message=None)
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
    # a configured unrecoverable_error_message is spoken for any unrecoverable error, not
    # just quota;
    # max_unrecoverable_errors=0 forces the close on the first occurrence
    session = _make_session(
        unrecoverable_error_message="Goodbye for now.",
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


async def test_custom_error_message_overrides_default_for_quota() -> None:
    # an explicit unrecoverable_error_message wins over the default generic message for
    # quota errors
    session = _make_session(unrecoverable_error_message="Custom branded message.")
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

    assert spoken == [DEFAULT_UNRECOVERABLE_ERROR_MESSAGE]
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


# --- str | AudioSource error messages -------------------------------------------------


async def _silent_frames() -> Any:
    yield rtc.AudioFrame(
        data=b"\x00\x00" * 240, sample_rate=24000, num_channels=1, samples_per_channel=240
    )


def _spy_say_calls(session: AgentSession) -> list[dict[str, Any]]:
    """Record both the text and the ``audio`` kwarg passed to ``session.say``."""
    calls: list[dict[str, Any]] = []
    original_say = session.say

    def spy(text: Any, **kwargs: Any) -> Any:
        calls.append({"text": text, "audio": kwargs.get("audio", NOT_GIVEN)})
        return original_say(text, **kwargs)

    session.say = spy  # type: ignore[method-assign]
    return calls


def _patch_audio_frames(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture paths passed to audio_frames_from_file; avoid real audio decoding."""
    paths: list[str] = []

    def fake(path: str, *args: Any, **kwargs: Any) -> Any:
        paths.append(path)
        return _silent_frames()

    monkeypatch.setattr("livekit.agents.voice.agent_session.audio_frames_from_file", fake)
    return paths


async def test_error_message_file_path_plays_audio(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a str that points at an existing file is played as audio, not synthesized via TTS
    audio_file = tmp_path / "bye.ogg"
    audio_file.write_bytes(b"fake-ogg")
    decoded_paths = _patch_audio_frames(monkeypatch)

    session = _make_session(unrecoverable_error_message=str(audio_file))
    await session.start(_Agent())
    calls = _spy_say_calls(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert decoded_paths == [str(audio_file)]
    assert len(calls) == 1
    assert calls[0]["text"] == ""  # no TTS text — audio is played as-is
    assert calls[0]["audio"] is not NOT_GIVEN

    await session.aclose()


async def test_error_message_builtin_clip_plays_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    decoded_paths = _patch_audio_frames(monkeypatch)

    session = _make_session(unrecoverable_error_message=BuiltinAudioClip.HOLD_MUSIC)
    await session.start(_Agent())
    calls = _spy_say_calls(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert decoded_paths == [BuiltinAudioClip.HOLD_MUSIC.path()]
    assert calls[0]["text"] == ""
    assert calls[0]["audio"] is not NOT_GIVEN

    await session.aclose()


async def test_error_message_audio_iterator_played_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a pre-built AudioSource iterator is forwarded as-is (no file decoding)
    decoded_paths = _patch_audio_frames(monkeypatch)
    frames = _silent_frames()

    session = _make_session(unrecoverable_error_message=frames)
    await session.start(_Agent())
    calls = _spy_say_calls(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert decoded_paths == []  # not a path/clip — never decoded from file
    assert calls[0]["text"] == ""
    assert calls[0]["audio"] is frames

    await session.aclose()


async def test_error_message_non_path_string_falls_back_to_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a str that is NOT an existing file is synthesized through TTS (legacy behavior)
    decoded_paths = _patch_audio_frames(monkeypatch)

    session = _make_session(unrecoverable_error_message="The assistant is unavailable.")
    await session.start(_Agent())
    calls = _spy_say_calls(session)

    activity = session._activity
    assert activity is not None
    activity._on_error(_llm_error(_quota_error()))

    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert decoded_paths == []
    assert calls[0]["text"] == "The assistant is unavailable."
    assert calls[0]["audio"] is NOT_GIVEN

    await session.aclose()


async def test_bare_emit_error_does_not_close() -> None:
    # teardown is owned by the activity's _on_error, not by the "error" event itself, so
    # a bare emit() (e.g. from user code) must not close the session.
    session = _make_session(unrecoverable_error_message=None)
    await session.start(_Agent())

    activity = session._activity
    assert activity is not None
    session.emit("error", ErrorEvent(error=_llm_error(_quota_error()), source=activity.llm))

    assert session._closing_task is None

    await session.aclose()
