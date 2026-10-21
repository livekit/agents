"""
Tests for the beta translation primitives: a translator listens and speaks but holds no
conversation, so ``TranslationModel`` is its own model class and ``TranslationSession`` its own
session — no agent, no chat context, no tools, nothing to hand it but audio.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable

import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.beta.translation import (
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    OutputSpeechStopped,
    OutputTranscript,
    TranslatedAudioFrame,
    TranslationCapabilities,
    TranslationModel,
    TranslationModelError,
    TranslationModelSession,
    TranslationSession,
)
from livekit.agents.voice.events import CloseEvent, CloseReason

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput

pytestmark = pytest.mark.unit


def _frame(duration: float = 0.02, sample_rate: int = 16000) -> rtc.AudioFrame:
    num_samples = int(sample_rate * duration)
    return rtc.AudioFrame(
        data=b"\x00\x00" * num_samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )


async def _wait_for(pred: Callable[[], bool], timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not pred():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


class _FakeTranslationSession(TranslationModelSession):
    def __init__(self, model: _FakeTranslationModel) -> None:
        super().__init__(model)
        self.pushed: list[rtc.AudioFrame] = []
        self.retargeted: list[str] = []
        self.closed = False
        self._audio_ch = utils.aio.Chan[TranslatedAudioFrame]()

    @property
    def audio_stream(self) -> AsyncIterable[TranslatedAudioFrame]:
        return self._audio_ch

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.pushed.append(frame)

    def update_options(self, *, target_language: str) -> None:
        if not self.capabilities.retargetable:
            super().update_options(target_language=target_language)

        self.retargeted.append(target_language)

    async def aclose(self) -> None:
        self.closed = True
        self._audio_ch.close()

    # -- test helpers: what the provider would report --

    def speak(self, frame: rtc.AudioFrame | None = None) -> None:
        self._audio_ch.send_nowait(TranslatedAudioFrame(frame=frame or _frame()))


class _FakeTranslationModel(TranslationModel):
    """A translator whose every report is driven by the test."""

    def __init__(self, *, target_language: str = "es", retargetable: bool = False) -> None:
        super().__init__(
            capabilities=TranslationCapabilities(
                input_transcription=True,
                output_transcription=True,
                turn_reporting=True,
                retargetable=retargetable,
            ),
            target_language=target_language,
        )
        self.sessions: list[_FakeTranslationSession] = []
        self.closed = False

    def session(self) -> _FakeTranslationSession:
        model_session = _FakeTranslationSession(self)
        self.sessions.append(model_session)
        return model_session

    async def aclose(self) -> None:
        self.closed = True


def _session(
    model: _FakeTranslationModel | None = None, **kwargs: object
) -> tuple[TranslationSession, FakeAudioInput, FakeAudioOutput, FakeTextOutput]:
    session = TranslationSession(model=model or _FakeTranslationModel(), **kwargs)  # type: ignore[arg-type]
    audio_in, audio_out, text_out = FakeAudioInput(), FakeAudioOutput(), FakeTextOutput()
    session.input.audio = audio_in
    session.output.audio = audio_out
    session.output.transcription = text_out
    return session, audio_in, audio_out, text_out


def test_the_model_is_its_own_primitive() -> None:
    model = _FakeTranslationModel(target_language="de")
    # it answers for a language, not for instructions, a chat context or tools
    assert model.target_language == "de"
    assert model.source_language is None
    assert model.capabilities.retargetable is False
    assert not isinstance(model, TranslationModelSession)


def test_the_target_language_comes_from_the_model() -> None:
    model = _FakeTranslationModel(target_language="de")
    session = TranslationSession(model=model)
    assert session.model is model
    assert session.model.target_language == "de"


def test_retargeting_before_start_only_changes_the_next_session() -> None:
    session = TranslationSession(model=_FakeTranslationModel(target_language="es"))
    session.set_target_language("fr")
    assert session.model.target_language == "fr"


async def test_retargeting_a_running_session_reaches_the_model() -> None:
    model = _FakeTranslationModel(target_language="es", retargetable=True)
    session, *_ = _session(model)
    await session.start()
    try:
        session.set_target_language("fr")
        assert session.model.target_language == "fr"
        assert model.sessions[0].retargeted == ["fr"]
    finally:
        await session.aclose()


async def test_retargeting_a_running_fixed_model_is_refused() -> None:
    model = _FakeTranslationModel(target_language="es")
    session, *_ = _session(model)
    await session.start()
    try:
        with pytest.raises(RuntimeError, match="fixed for the session's life"):
            session.set_target_language("fr")
        assert session.model.target_language == "es"
        assert model.sessions[0].retargeted == []
    finally:
        await session.aclose()


def test_a_fixed_model_session_refuses_to_be_retargeted() -> None:
    model_session = _FakeTranslationModel().session()
    with pytest.raises(RuntimeError, match="fixed for the session's life"):
        model_session.update_options(target_language="fr")


async def test_the_speech_it_hears_reaches_the_model() -> None:
    model = _FakeTranslationModel()
    session, audio_in, _, _ = _session(model)
    await session.start()
    try:
        audio_in.push(_frame())
        audio_in.push(_frame())
        await _wait_for(lambda: len(model.sessions[0].pushed) == 2)
    finally:
        await session.aclose()


async def test_the_translation_it_speaks_reaches_the_audio_output() -> None:
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model)
    await session.start()
    try:
        model.sessions[0].speak()
        await _wait_for(lambda: audio_out._pushed_duration > 0)
    finally:
        await session.aclose()


async def test_both_transcripts_are_reported() -> None:
    model = _FakeTranslationModel(target_language="es")
    session, _, _, text_out = _session(model)
    heard: list[tuple[str, bool, str | None]] = []
    spoken: list[tuple[str, bool, str | None]] = []
    session.on(
        "user_input_transcribed", lambda ev: heard.append((ev.transcript, ev.is_final, ev.language))
    )
    session.on(
        "translation_transcribed",
        lambda ev: spoken.append((ev.transcript, ev.is_final, ev.language)),
    )

    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.emit(
            "input_transcript", InputTranscript(text="hello", is_final=True, language="en")
        )
        model_session.emit("output_transcript", OutputTranscript(text="hola", is_final=True))

        assert heard == [("hello", True, "en")]
        # the language the model translates into stands in when it stamps none
        assert spoken == [("hola", True, "es")]
        await _wait_for(lambda: text_out._messages == ["hola"])
    finally:
        await session.aclose()


async def test_the_speakers_turns_drive_the_user_state() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    states: list[str] = []
    session.on("user_state_changed", lambda ev: states.append(ev.new_state))

    await session.start()
    try:
        model_session = model.sessions[0]
        assert session.user_state == "listening"
        model_session.emit("input_speech_started", InputSpeechStarted())
        assert session.user_state == "speaking"
        model_session.emit("input_speech_stopped", InputSpeechStopped())
        assert session.user_state == "listening"
        assert states == ["speaking", "listening"]
    finally:
        await session.aclose()


async def test_the_model_reporting_its_speech_ended_ends_the_segment() -> None:
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model.sessions[0].speak()
        await _wait_for(lambda: audio_out._pushed_duration > 0)

        model.sessions[0].emit("output_speech_stopped", OutputSpeechStopped())
        await _wait_for(lambda: bool(played))
        assert played[0] > 0  # the segment played out rather than being cut
    finally:
        await session.aclose()


async def test_an_idle_model_still_ends_its_segment() -> None:
    # a provider that never reports where its speech ends
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model.sessions[0].speak()
        await _wait_for(lambda: bool(played))
    finally:
        await session.aclose()


async def test_closing_reports_why_and_releases_the_model_session() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    closed: list[CloseEvent] = []
    session.on("close", closed.append)

    await session.start()
    await session.aclose()

    assert [ev.reason for ev in closed] == [CloseReason.USER_INITIATED]
    assert model.sessions[0].closed
    # the session closes its run of the model, never the model itself
    assert not model.closed


async def test_an_unrecoverable_model_error_closes_the_session() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    errors: list[object] = []
    closed: list[CloseEvent] = []
    session.on("error", lambda ev: errors.append(ev.error))
    session.on("close", closed.append)

    await session.start()
    model.sessions[0].emit(
        "error",
        TranslationModelError(
            timestamp=0.0, label=model.label, error=RuntimeError("gone"), recoverable=False
        ),
    )

    await _wait_for(lambda: bool(closed))
    assert len(errors) == 1
    assert closed[0].reason == CloseReason.ERROR


async def test_a_recoverable_model_error_is_only_reported() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    closed: list[CloseEvent] = []
    session.on("close", closed.append)

    await session.start()
    try:
        model.sessions[0].emit(
            "error",
            TranslationModelError(
                timestamp=0.0, label=model.label, error=RuntimeError("blip"), recoverable=True
            ),
        )
        await asyncio.sleep(0.05)
        assert closed == []
    finally:
        await session.aclose()


async def test_starting_twice_is_a_no_op() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    await session.start()
    try:
        await session.start()
        assert len(model.sessions) == 1
    finally:
        await session.aclose()
