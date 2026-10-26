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
    TranslationRoomIO,
    TranslationRoomOptions,
    TranslationSession,
)
from livekit.agents.beta.translation.room_io import _LinkedTrack
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
        self._audio_ch = utils.aio.Chan[TranslatedAudioFrame | OutputSpeechStopped]()

    @property
    def audio_stream(self) -> AsyncIterable[TranslatedAudioFrame | OutputSpeechStopped]:
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

    def end_speech(self) -> None:
        self._audio_ch.send_nowait(OutputSpeechStopped())


class _FakeTranslationModel(TranslationModel):
    """A translator whose every report is driven by the test."""

    def __init__(
        self,
        *,
        target_language: str = "es",
        retargetable: bool = False,
        output_segmentation: bool = False,
    ) -> None:
        super().__init__(
            capabilities=TranslationCapabilities(
                input_transcription=True,
                output_transcription=True,
                turn_reporting=True,
                retargetable=retargetable,
                output_segmentation=output_segmentation,
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
        model_session.speak()

        assert heard == [("hello", True, "en")]
        # the language the model translates into stands in when it stamps none
        assert spoken == [("hola", True, "es")]
        # published once its audio segment ends, which the text segment ends with
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


async def test_the_model_marking_its_speech_ended_ends_the_segment() -> None:
    model = _FakeTranslationModel(output_segmentation=True)
    session, _, audio_out, _ = _session(model)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model.sessions[0].speak()
        model.sessions[0].end_speech()
        await _wait_for(lambda: bool(played))
        assert played[0] > 0  # the segment played out rather than being cut
    finally:
        await session.aclose()


async def test_the_mark_cannot_overtake_the_audio_it_closes() -> None:
    # queued together, the mark is read behind the frames, so none land in the next segment
    model = _FakeTranslationModel(output_segmentation=True)
    session, _, audio_out, _ = _session(model)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model_session = model.sessions[0]
        for _ in range(3):
            model_session.speak()
        model_session.end_speech()

        await _wait_for(lambda: bool(played))
        assert len(played) == 1
        assert played[0] == pytest.approx(3 * 0.02, abs=1e-6)
    finally:
        await session.aclose()


async def test_a_model_that_marks_its_own_segments_runs_no_silence_timer() -> None:
    # a pause inside one translation must not split it
    model = _FakeTranslationModel(output_segmentation=True)
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model.sessions[0].speak()
        await asyncio.sleep(0.15)  # longer than the timer, which must not be armed
        assert played == []

        model.sessions[0].end_speech()
        await _wait_for(lambda: bool(played))
    finally:
        await session.aclose()


async def test_an_idle_model_still_ends_its_segment() -> None:
    # a provider that never reports where its speech ends
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    # and never sends the transcript its segment would wait for
    session._text_wait_timeout = 0.05
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


# -- room wiring: what the session links is a track, not a participant --


class _FakePublication:
    def __init__(
        self,
        sid: str,
        *,
        kind: rtc.TrackKind.ValueType = rtc.TrackKind.KIND_AUDIO,
        source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE,
    ) -> None:
        self.sid = sid
        self.kind = kind
        self.source = source


class _FakeParticipant:
    def __init__(self, identity: str, *, attributes: dict[str, str] | None = None) -> None:
        self.identity = identity
        self.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
        self.attributes = attributes or {}


def _room_io(**options: object) -> TranslationRoomIO:
    session = TranslationSession(model=_FakeTranslationModel())
    return TranslationRoomIO(
        session,
        rtc.Room(),
        options=TranslationRoomOptions(**options),  # type: ignore[arg-type]
    )


async def test_a_named_track_is_taken_whoever_published_it() -> None:
    room_io = _room_io(track="TR_wanted")
    other, wanted = _FakePublication("TR_other"), _FakePublication("TR_wanted")
    participant = _FakeParticipant("speaker-1")

    room_io._on_track_published(other, participant)  # type: ignore[arg-type]
    assert room_io.linked_track is None

    room_io._on_track_published(wanted, participant)  # type: ignore[arg-type]
    assert room_io.linked_track is not None
    assert room_io.linked_track.publication is wanted
    # the publisher is read off the track rather than supplied
    assert room_io.linked_participant is participant  # type: ignore[comparison-overlap]


async def test_a_video_track_is_never_linked() -> None:
    room_io = _room_io(participant_identity="speaker-1")
    video = _FakePublication("TR_video", kind=rtc.TrackKind.KIND_VIDEO)

    room_io._on_track_published(video, _FakeParticipant("speaker-1"))  # type: ignore[arg-type]
    assert room_io.linked_track is None


async def test_auto_selection_takes_a_microphone_track_only() -> None:
    room_io = _room_io(participant_identity="speaker-1")
    participant = _FakeParticipant("speaker-1")
    screen = _FakePublication("TR_screen", source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)
    mic = _FakePublication("TR_mic")

    room_io._on_track_published(screen, participant)  # type: ignore[arg-type]
    assert room_io.linked_track is None

    room_io._on_track_published(mic, participant)  # type: ignore[arg-type]
    assert room_io.linked_track is not None
    assert room_io.linked_track.publication is mic


async def test_auto_selection_is_narrowed_to_the_named_participant() -> None:
    room_io = _room_io(participant_identity="speaker-1")

    room_io._on_track_published(_FakePublication("TR_a"), _FakeParticipant("someone-else"))  # type: ignore[arg-type]
    assert room_io.linked_track is None

    room_io._on_track_published(_FakePublication("TR_b"), _FakeParticipant("speaker-1"))  # type: ignore[arg-type]
    assert room_io.linked_track is not None


async def test_a_named_track_ignores_the_participant_filter() -> None:
    # the sid identifies a publication outright, so the search needs no narrowing
    room_io = _room_io(track="TR_wanted", participant_identity="speaker-1")
    screen = _FakePublication("TR_wanted", source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)

    room_io._on_track_published(screen, _FakeParticipant("someone-else"))  # type: ignore[arg-type]
    assert room_io.linked_track is not None
    assert room_io.linked_track.publication is screen


# -- what the fixes above are about --


class _RecordingAudioOutput(FakeAudioOutput):
    """Keeps every frame it was handed, to check what rate reached the room."""

    def __init__(self, *, sample_rate: int) -> None:
        super().__init__(sample_rate=sample_rate)
        self.frames: list[rtc.AudioFrame] = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)
        await super().capture_frame(frame)


async def test_model_audio_is_resampled_to_the_rate_the_room_plays_at() -> None:
    model = _FakeTranslationModel(output_segmentation=True)
    session = TranslationSession(model=model)
    audio_out = _RecordingAudioOutput(sample_rate=24000)
    session.input.audio = FakeAudioInput()
    session.output.audio = audio_out

    await session.start()
    try:
        # the provider speaks at 16 kHz into a sink that plays at 24 kHz
        model.sessions[0].speak(_frame(duration=0.1, sample_rate=16000))
        model.sessions[0].end_speech()
        await _wait_for(lambda: bool(audio_out.frames))

        assert {frame.sample_rate for frame in audio_out.frames} == {24000}
    finally:
        await session.aclose()


async def test_audio_at_the_rooms_own_rate_is_passed_through() -> None:
    model = _FakeTranslationModel(output_segmentation=True)
    session = TranslationSession(model=model)
    audio_out = _RecordingAudioOutput(sample_rate=24000)
    session.input.audio = FakeAudioInput()
    session.output.audio = audio_out

    await session.start()
    try:
        sent = _frame(duration=0.1, sample_rate=24000)
        model.sessions[0].speak(sent)
        await _wait_for(lambda: bool(audio_out.frames))
        assert audio_out.frames[0] is sent  # no resampler in the way
    finally:
        await session.aclose()


async def test_closing_keeps_the_devices_the_caller_supplied() -> None:
    # room io closes only the devices it made; a caller's own were never closed here, and
    # dropping the reference would strand them
    session, audio_in, audio_out, text_out = _session()
    await session.start()
    await session.aclose()

    assert session.input.audio is audio_in
    assert session.output.audio is audio_out
    assert session.output.transcription is text_out


async def test_a_session_restarts_on_the_devices_it_was_given() -> None:
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model)
    await session.start()
    await session.aclose()

    await session.start()
    try:
        assert len(model.sessions) == 2
        model.sessions[1].speak()
        await _wait_for(lambda: audio_out._pushed_duration > 0)
    finally:
        await session.aclose()


async def test_a_refused_retarget_leaves_the_model_language_alone() -> None:
    class _RefusingSession(_FakeTranslationSession):
        def update_options(self, *, target_language: str) -> None:
            raise RuntimeError(f"{target_language} is not available on this connection")

    class _RefusingModel(_FakeTranslationModel):
        def session(self) -> _FakeTranslationSession:
            model_session = _RefusingSession(self)
            self.sessions.append(model_session)
            return model_session

    model = _RefusingModel(target_language="es", retargetable=True)
    session, *_ = _session(model)
    await session.start()
    try:
        with pytest.raises(RuntimeError, match="not available on this connection"):
            session.set_target_language("fr")

        # the session is still speaking Spanish, so the model must still say so
        assert model.target_language == "es"
    finally:
        await session.aclose()


# -- switching the linked track --


class _RecordingAudioInput:
    def __init__(self) -> None:
        self.participants: list[str | None] = []
        self.tracks: list[str | None] = []

    def set_participant(self, participant: str | None) -> None:
        self.participants.append(participant)

    def set_track_sid(self, track_sid: str | None) -> None:
        self.tracks.append(track_sid)


class _RecordingTranscriptOutput:
    def __init__(self) -> None:
        self.participants: list[str | None] = []
        self.tracks: list[str | None] = []

    def set_participant(self, participant: str | None) -> None:
        self.participants.append(participant)

    def set_track(self, track_sid: str | None) -> None:
        self.tracks.append(track_sid)


def _linked(room_io: TranslationRoomIO) -> tuple[_RecordingAudioInput, _RecordingTranscriptOutput]:
    audio_in, transcript = _RecordingAudioInput(), _RecordingTranscriptOutput()
    room_io._audio_input = audio_in  # type: ignore[assignment]
    room_io._input_tr_output = transcript  # type: ignore[assignment]
    return audio_in, transcript


async def test_switching_tracks_moves_the_transcript_stamp_too() -> None:
    room_io = _room_io()
    audio_in, transcript = _linked(room_io)
    bob = _FakeParticipant("bob")

    room_io._apply_link(_LinkedTrack(_FakePublication("TR_b"), bob))  # type: ignore[arg-type]

    assert transcript.participants == ["bob"]
    # the stamp follows the track, or Bob's words keep Alice's track sid
    assert transcript.tracks == ["TR_b"]
    # pinned before the participant, so the scan cannot latch another of their tracks
    assert audio_in.tracks == ["TR_b"]
    assert audio_in.participants == ["bob"]


async def test_a_track_named_before_it_exists_is_linked_once_published() -> None:
    room_io = _room_io(track="TR_late")
    audio_in, transcript = _linked(room_io)

    # startup has already finished waiting, so only the publish event can apply this
    done: asyncio.Future[None] = asyncio.Future()
    done.set_result(None)
    room_io._init_atask = done  # type: ignore[assignment]

    bob = _FakeParticipant("bob")
    room_io._on_track_published(_FakePublication("TR_late"), bob)  # type: ignore[arg-type]

    assert room_io.linked_track is not None
    assert audio_in.participants == ["bob"]
    assert transcript.tracks == ["TR_late"]


async def test_unsetting_the_track_before_one_is_linked_does_not_stall_startup() -> None:
    room_io = _room_io(text_output=False)
    audio_in, _ = _linked(room_io)
    room_io._init_atask = asyncio.create_task(room_io._init_task())
    room_io._room_connected_fut.set_result(None)
    try:
        # startup is now waiting for a track to link
        await asyncio.sleep(0)

        room_io.unset_track()
        room_io.set_track("TR_late")
        bob = _FakeParticipant("bob")
        room_io._on_track_published(_FakePublication("TR_late"), bob)  # type: ignore[arg-type]

        await asyncio.wait_for(room_io.wait_for_ready(), timeout=1.0)
        assert audio_in.participants[-1] == "bob"
        assert audio_in.tracks[-1] == "TR_late"
    finally:
        await utils.aio.cancel_and_wait(room_io._init_atask)


async def test_switching_to_a_video_track_is_refused() -> None:
    room_io = _room_io()
    _linked(room_io)
    camera = _FakePublication("TR_camera", kind=rtc.TrackKind.KIND_VIDEO)
    room_io.room._remote_participants["bob"] = _FakeParticipant("bob")  # type: ignore[assignment]
    room_io.room._remote_participants["bob"].track_publications = {"TR_camera": camera}  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="not audio"):
        room_io.set_track("TR_camera")


async def test_a_named_screenshare_track_can_reach_the_model() -> None:
    # selecting screen-share audio is pointless if the input only ever accepts a microphone
    session = TranslationSession(model=_FakeTranslationModel())
    room_io = TranslationRoomIO(session, rtc.Room(), options=TranslationRoomOptions())
    await room_io.start()
    try:
        assert room_io._audio_input is not None
        assert rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO in room_io._audio_input._accepted_sources
        assert rtc.TrackSource.SOURCE_MICROPHONE in room_io._audio_input._accepted_sources
    finally:
        await room_io.aclose()


async def test_a_speech_mark_ends_the_segment_even_while_output_is_muted() -> None:
    # or the next translation extends the segment the mute interrupted
    model = _FakeTranslationModel(output_segmentation=True)
    session, _, audio_out, _ = _session(model)
    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.speak()
        await _wait_for(lambda: audio_out._pushed_duration > 0)

        session.output.set_audio_enabled(False)
        model_session.end_speech()

        await _wait_for(lambda: bool(played))
        assert played[0] > 0
    finally:
        await session.aclose()


async def test_replacing_the_output_rebuilds_the_resampler_for_its_rate() -> None:
    model = _FakeTranslationModel(output_segmentation=True)
    session = TranslationSession(model=model)
    first = _RecordingAudioOutput(sample_rate=24000)
    session.input.audio = FakeAudioInput()
    session.output.audio = first

    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.speak(_frame(duration=0.1, sample_rate=16000))
        await _wait_for(lambda: bool(first.frames))
        assert {f.sample_rate for f in first.frames} == {24000}

        # the caller swaps in a sink that plays at another rate
        second = _RecordingAudioOutput(sample_rate=48000)
        session.output.audio = second

        model_session.speak(_frame(duration=0.1, sample_rate=16000))
        await _wait_for(lambda: bool(second.frames))
        assert {f.sample_rate for f in second.frames} == {48000}
    finally:
        await session.aclose()


async def test_the_silence_timer_drains_the_resampler_into_its_own_segment() -> None:
    # a tail the resampler holds belongs to the segment being closed, not the next one
    model = _FakeTranslationModel()  # marks no boundaries, so the timer ends the segment
    session = TranslationSession(model=model, output_segment_timeout=0.05)
    audio_out = _RecordingAudioOutput(sample_rate=24000)
    session.input.audio = FakeAudioInput()
    session.output.audio = audio_out

    played: list[float] = []
    audio_out.on("playback_finished", lambda ev: played.append(ev.playback_position))

    await session.start()
    try:
        # short enough that resampling holds a partial frame back
        model.sessions[0].speak(_frame(duration=0.03, sample_rate=16000))

        # the timer drains before it flushes, so the tail reaches the sink inside this
        # segment; undrained, the flush would find nothing and the tail would surface later
        await _wait_for(lambda: bool(played))
        assert session._resampler is None
        assert audio_out.frames
        assert {f.sample_rate for f in audio_out.frames} == {24000}
    finally:
        await session.aclose()


async def test_a_pending_tail_goes_to_the_sink_it_was_converted_for() -> None:
    # replacing the sink mid-translation must not replay the old sink's tail at a new rate
    model = _FakeTranslationModel(output_segmentation=True)
    session = TranslationSession(model=model, output_segment_timeout=10)
    first = _RecordingAudioOutput(sample_rate=24000)
    session.input.audio = FakeAudioInput()
    session.output.audio = first

    await session.start()
    try:
        model_session = model.sessions[0]
        # short enough that the resampler holds part of it back
        model_session.speak(_frame(duration=0.03, sample_rate=16000))
        await _wait_for(lambda: session._resampler is not None)

        second = _RecordingAudioOutput(sample_rate=48000)
        session.output.audio = second
        model_session.speak(_frame(duration=0.1, sample_rate=16000))
        await _wait_for(lambda: bool(second.frames))

        # the tail went to the sink it was built for, never to the replacement
        assert {f.sample_rate for f in first.frames} <= {24000}
        assert {f.sample_rate for f in second.frames} == {48000}
    finally:
        await session.aclose()


async def test_pinning_the_track_already_linked_still_pins_it() -> None:
    room_io = _room_io()
    audio_in, _ = _linked(room_io)
    alice = _FakeParticipant("alice")
    room_io._apply_link(_LinkedTrack(_FakePublication("TR_a"), alice))  # type: ignore[arg-type]

    room_io.set_track("TR_a")

    # unpinned, a later automatic search is free to settle on another microphone
    assert room_io._track_sid == "TR_a"
    assert audio_in.tracks[-1] == "TR_a"


async def test_a_segment_is_finalized_on_the_track_that_carried_it() -> None:
    # the flush runs after set_track has moved on, so it must carry the segment's own sid
    from livekit.agents.voice.room_io._output import _ParticipantStreamTranscriptionOutput

    sink = _ParticipantStreamTranscriptionOutput(
        rtc.Room(), is_delta_stream=True, participant="alice", track="TR_a"
    )
    flushed: list[tuple[str | None, str | None]] = []

    async def _record_flush(
        writer: object = None,
        extra_attributes: dict[str, str] | None = None,
        pending_text: str = "",
        track_id: str | None = None,
        sender_identity: str | None = None,
        additional_attributes: dict[str, str] | None = None,
    ) -> None:
        flushed.append((track_id, sender_identity))

    sink._flush_task = _record_flush  # type: ignore[assignment,method-assign]
    sink._capturing = True

    sink.set_track("TR_b")
    await asyncio.sleep(0)

    # the closing packet belongs to TR_a, although the sink now stamps TR_b
    assert flushed == [("TR_a", "alice")]
    assert sink._pinned_track_id == "TR_b"
    await sink.aclose()


# -- attributing the translation to the speaker, and naming its track --


class _RecordingTaggedOutput(_RecordingTranscriptOutput):
    def __init__(self) -> None:
        super().__init__()
        self.attributes: list[dict[str, str]] = []
        self.captured: list[tuple[str, dict[str, str]]] = []

    def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attributes.append(dict(attributes))

    async def capture_text(self, text: str) -> None:
        self.captured.append((text, self.attributes[-1] if self.attributes else {}))

    def flush(self) -> None:
        pass


class _NamingAudioOutput:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.started = False

    async def set_track_name(self, name: str) -> None:
        self.names.append(name)

    async def start(self) -> None:
        self.started = True


async def test_the_translation_stays_the_agents_and_carries_its_language() -> None:
    room_io = _room_io()
    _linked(room_io)
    translation = _RecordingTaggedOutput()
    room_io._output_tr_output = translation  # type: ignore[assignment]
    alice = _FakeParticipant("alice")

    room_io._apply_link(_LinkedTrack(_FakePublication("TR_a"), alice))  # type: ignore[arg-type]

    # clients file a transcript under its sender: sent as Alice, the translation would read
    # as her own words, so it stays the agent's, on the agent's track
    assert translation.participants == []
    assert translation.tracks == []
    assert translation.attributes == [{"lk.transcription_language": "es"}]


async def test_the_source_transcript_carries_the_language_heard() -> None:
    room_io = _room_io()
    heard = _RecordingTaggedOutput()
    room_io._input_tr_output = heard  # type: ignore[assignment]
    ch = utils.aio.Chan[tuple[str, bool, str | None]]()
    task = asyncio.create_task(room_io._forward_input_transcript(ch))
    try:
        ch.send_nowait(("hello", False, "en"))
        ch.send_nowait(("hello there", True, None))
        ch.close()
        await asyncio.wait_for(task, 1.0)
    finally:
        await utils.aio.cancel_and_wait(task)

    assert heard.captured == [
        ("hello", {"lk.transcription_language": "en"}),
        # a model that does not say leaves it off rather than guess
        ("hello there", {}),
    ]


async def test_the_translated_track_is_named_for_its_source_and_language() -> None:
    room_io = _room_io(text_output=False, track="TR_a")
    _linked(room_io)
    audio_out = _NamingAudioOutput()
    room_io._audio_output = audio_out  # type: ignore[assignment]
    room_io._init_atask = asyncio.create_task(room_io._init_task())
    room_io._room_connected_fut.set_result(None)
    try:
        room_io._on_track_published(
            _FakePublication("TR_a"),  # type: ignore[arg-type]
            _FakeParticipant("alice"),  # type: ignore[arg-type]
        )
        await asyncio.wait_for(room_io.wait_for_ready(), timeout=1.0)
    finally:
        await utils.aio.cancel_and_wait(room_io._init_atask)

    # named before the first publish, so it never goes out under the default
    assert audio_out.names == ["TR_a-es"]
    assert audio_out.started


async def test_retargeting_or_relinking_renames_the_translated_track() -> None:
    model = _FakeTranslationModel()
    session = TranslationSession(model=model)
    room_io = TranslationRoomIO(session, rtc.Room())
    session._room_io = room_io
    _linked(room_io)
    translation = _RecordingTaggedOutput()
    room_io._output_tr_output = translation  # type: ignore[assignment]
    audio_out = _NamingAudioOutput()
    room_io._audio_output = audio_out  # type: ignore[assignment]
    done: asyncio.Future[None] = asyncio.Future()
    done.set_result(None)
    room_io._init_atask = done  # type: ignore[assignment]
    alice = _FakeParticipant("alice")
    room_io._apply_link(_LinkedTrack(_FakePublication("TR_a"), alice))  # type: ignore[arg-type]

    session.set_target_language("fr")
    room_io._apply_link(_LinkedTrack(_FakePublication("TR_b"), alice))  # type: ignore[arg-type]
    await asyncio.gather(*room_io._tasks)

    assert audio_out.names == ["TR_a-es", "TR_a-fr", "TR_b-fr"]
    assert translation.attributes[-1] == {"lk.transcription_language": "fr"}


async def test_a_track_name_given_outright_is_kept() -> None:
    from livekit.agents.voice.room_io import AudioOutputOptions

    room_io = _room_io(audio_output=AudioOutputOptions(track_name="translation"))
    _linked(room_io)
    audio_out = _NamingAudioOutput()
    room_io._audio_output = audio_out  # type: ignore[assignment]
    done: asyncio.Future[None] = asyncio.Future()
    done.set_result(None)
    room_io._init_atask = done  # type: ignore[assignment]

    room_io._apply_link(
        _LinkedTrack(_FakePublication("TR_a"), _FakeParticipant("alice"))  # type: ignore[arg-type]
    )
    await asyncio.gather(*room_io._tasks)

    assert audio_out.names == []


async def test_renaming_a_published_track_replaces_it() -> None:
    from types import SimpleNamespace

    from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

    published: list[str] = []
    unpublished: list[str] = []

    async def _publish(track: rtc.LocalAudioTrack, _: object) -> object:
        published.append(track.name)
        return SimpleNamespace(sid=f"TR_{len(published)}")

    async def _unpublish(sid: str) -> None:
        unpublished.append(sid)

    room = SimpleNamespace(
        local_participant=SimpleNamespace(publish_track=_publish, unpublish_track=_unpublish)
    )
    output = _ParticipantAudioOutput(
        room,  # type: ignore[arg-type]
        sample_rate=24000,
        num_channels=1,
        track_publish_options=rtc.TrackPublishOptions(),
    )
    try:
        # before publishing, only the name changes
        await output.set_track_name("TR_a-es")
        assert published == []

        output._publication = await _publish(  # type: ignore[assignment]
            rtc.LocalAudioTrack.create_audio_track("TR_a-es", output._audio_source), None
        )
        await output.set_track_name("TR_a-fr")
        await output.set_track_name("TR_a-fr")

        # a track's name is fixed once published, so the track itself is replaced, once
        assert unpublished == ["TR_1"]
        assert published == ["TR_a-es", "TR_a-fr"]
        assert output.track_name == "TR_a-fr"
    finally:
        await output.aclose()


# -- the label follows the language actually spoken, and segments close under their own --


class _LabelRecorder:
    def __init__(self) -> None:
        self.languages: list[str] = []

    def _on_output_language_changed(self, language: str) -> None:
        self.languages.append(language)


class _Counts:
    def __init__(self, audio_out: FakeAudioOutput) -> None:
        self.frames = 0
        self.flushes = 0
        capture, flush = audio_out.capture_frame, audio_out.flush

        async def _capture(frame: rtc.AudioFrame) -> None:
            self.frames += 1
            await capture(frame)

        def _flush() -> None:
            self.flushes += 1
            flush()

        audio_out.capture_frame = _capture  # type: ignore[method-assign]
        audio_out.flush = _flush  # type: ignore[method-assign]


async def test_the_label_follows_the_language_the_speech_is_stamped_with() -> None:
    model = _FakeTranslationModel(retargetable=True)
    session, _, audio_out, _ = _session(model)
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        model_session = model.sessions[0]
        model_session._audio_ch.send_nowait(TranslatedAudioFrame(frame=_frame(), language="es"))
        await _wait_for(lambda: counts.frames == 1)

        session.set_target_language("fr")
        # the provider has not applied it yet, so Spanish is still what is being spoken
        model_session._audio_ch.send_nowait(TranslatedAudioFrame(frame=_frame(), language="es"))
        await _wait_for(lambda: counts.frames == 2)
        assert labels.languages == []

        model_session._audio_ch.send_nowait(TranslatedAudioFrame(frame=_frame(), language="fr"))
        await _wait_for(lambda: counts.frames == 3)
        assert labels.languages == ["fr"]
        # the Spanish segment ended before the French speech went into the next
        assert counts.flushes >= 1
    finally:
        session._room_io = None
        await session.aclose()


async def test_an_unstamped_provider_is_labelled_with_the_request() -> None:
    model = _FakeTranslationModel(retargetable=True)
    session, _, audio_out, _ = _session(model)
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        model.sessions[0].speak()
        await _wait_for(lambda: counts.frames == 1)

        session.set_target_language("fr")
        await _wait_for(lambda: labels.languages == ["fr"])
    finally:
        session._room_io = None
        await session.aclose()


async def test_an_unstamped_retarget_ends_a_segment_that_never_pauses() -> None:
    model = _FakeTranslationModel(retargetable=True)
    # no pause in the speech is ever long enough to end the segment on its own
    session, _, audio_out, _ = _session(model, output_segment_timeout=60)
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        model.sessions[0].speak()
        await _wait_for(lambda: counts.frames == 1)

        session.set_target_language("fr")
        await _wait_for(lambda: labels.languages == ["fr"])
        # the Spanish segment ended, so its playout can finish and the label can land
        assert counts.flushes == 1
    finally:
        session._room_io = None
        await session.aclose()


def _record_stream_flushes(sink: object) -> list[tuple[str | None, dict[str, str] | None]]:
    flushed: list[tuple[str | None, dict[str, str] | None]] = []

    async def _record_flush(
        writer: object = None,
        extra_attributes: dict[str, str] | None = None,
        pending_text: str = "",
        track_id: str | None = None,
        sender_identity: str | None = None,
        additional_attributes: dict[str, str] | None = None,
    ) -> None:
        flushed.append((sender_identity, additional_attributes))

    sink._flush_task = _record_flush  # type: ignore[attr-defined]
    return flushed


async def test_a_language_change_closes_the_open_translation_segment_under_the_old_one() -> None:
    from livekit.agents.voice.room_io._output import _ParticipantStreamTranscriptionOutput

    sink = _ParticipantStreamTranscriptionOutput(
        rtc.Room(),
        is_delta_stream=True,
        participant="alice",
        track="TR_a",
        attributes={"lk.transcription_language": "es"},
    )
    flushed = _record_stream_flushes(sink)
    try:
        await sink.capture_text("hola")
        sink.set_attributes({"lk.transcription_language": "fr"})
        await asyncio.sleep(0)

        # its header went out in Spanish, so it closes in Spanish and French opens anew
        assert flushed == [("alice", {"lk.transcription_language": "es"})]
        assert not sink._capturing
    finally:
        await sink.aclose()


async def test_a_whole_text_segment_takes_a_new_language_without_closing() -> None:
    from livekit.agents.voice.room_io._output import _ParticipantStreamTranscriptionOutput

    sink = _ParticipantStreamTranscriptionOutput(
        rtc.Room(), is_delta_stream=False, participant="alice", track="TR_a"
    )
    flushed = _record_stream_flushes(sink)
    try:
        await sink.capture_text("hello")
        # a source transcript learns its language part way, and is republished whole anyway
        sink.set_attributes({"lk.transcription_language": "en"})
        await asyncio.sleep(0)
        assert flushed == []

        sink.flush()
        sink.set_attributes({})
        await asyncio.sleep(0)
        # the final keeps the language it was flushed with, not the one set after
        assert flushed == [("alice", {"lk.transcription_language": "en"})]
    finally:
        await sink.aclose()


async def test_clearing_the_speaker_closes_their_open_segment_as_theirs() -> None:
    from livekit.agents.voice.room_io._output import _ParticipantStreamTranscriptionOutput

    sink = _ParticipantStreamTranscriptionOutput(
        rtc.Room(), is_delta_stream=True, participant="alice", track="TR_a"
    )
    flushed = _record_stream_flushes(sink)
    try:
        await sink.capture_text("hola")
        sink.set_participant(None)
        await asyncio.sleep(0)

        assert [identity for identity, _ in flushed] == ["alice"]
    finally:
        await sink.aclose()


async def test_legacy_transcriptions_carry_the_language() -> None:
    from livekit.agents.voice.room_io._output import _ParticipantLegacyTranscriptionOutput

    sink = _ParticipantLegacyTranscriptionOutput(
        rtc.Room(), is_delta_stream=True, participant="alice", track="TR_a"
    )
    published: list[tuple[str, bool, str]] = []

    async def _record(
        id: str, text: str, final: bool, *, language: str | None = None, **_: object
    ) -> None:
        published.append((text, final, language if language is not None else sink._language))

    sink._publish_transcription = _record  # type: ignore[method-assign]
    try:
        sink.set_attributes({"lk.transcription_language": "es"})
        await sink.capture_text("hola")
        # a change closes the Spanish segment as Spanish
        sink.set_attributes({"lk.transcription_language": "fr"})
        await asyncio.sleep(0)
        await sink.capture_text("bonjour")
        sink.flush()
        await asyncio.sleep(0)

        assert published == [
            ("hola", False, "es"),
            ("hola", True, "es"),
            ("bonjour", False, "fr"),
            ("bonjour", True, "fr"),
        ]
    finally:
        await sink.aclose()


class _HeldSynchronizer:
    """Stands in for a TranscriptSynchronizer whose old segment is still playing out."""

    def __init__(self) -> None:
        self.played: asyncio.Future[None] = asyncio.Future()
        self.rotated: asyncio.Future[None] = asyncio.Future()

        class _Audio:
            async def wait_for_playout(_self) -> None:
                await self.played

        self.audio_output = _Audio()

    async def barrier(self) -> None:
        await self.rotated


async def test_a_language_change_waits_for_the_old_language_to_play_out() -> None:
    room_io = _room_io()
    _linked(room_io)
    translation = _RecordingTaggedOutput()
    room_io._output_tr_output = translation  # type: ignore[assignment]
    synchronizer = _HeldSynchronizer()
    room_io._tr_synchronizer = synchronizer  # type: ignore[assignment]
    room_io._apply_link(
        _LinkedTrack(_FakePublication("TR_a"), _FakeParticipant("alice"))  # type: ignore[arg-type]
    )

    room_io._on_output_language_changed("fr")
    await asyncio.sleep(0)
    # Spanish words the synchronizer has yet to release must still go out as Spanish
    assert translation.attributes[-1] == {"lk.transcription_language": "es"}

    synchronizer.played.set_result(None)
    await asyncio.sleep(0)
    # played, but the synchronizer has not closed the Spanish segment yet
    assert translation.attributes[-1] == {"lk.transcription_language": "es"}

    synchronizer.rotated.set_result(None)
    await asyncio.gather(*room_io._tasks)
    assert translation.attributes[-1] == {"lk.transcription_language": "fr"}


class _GatedTextRoom:
    """A connected room whose text streams open only when the test lets them."""

    def __init__(self) -> None:
        self.open_gate: asyncio.Future[None] = asyncio.Future()
        self.streams: list[dict[str, object]] = []
        room = self

        class _Writer:
            def __init__(self, attributes: dict[str, str]) -> None:
                self.record: dict[str, object] = {
                    "header": dict(attributes),
                    "text": [],
                    "closed_with": None,
                }
                room.streams.append(self.record)

            async def write(self, text: str) -> None:
                self.record["text"].append(text)  # type: ignore[union-attr]

            async def aclose(self, attributes: dict[str, str] | None = None) -> None:
                self.record["closed_with"] = dict(attributes or {})

        class _Local:
            identity = "agent"

            async def stream_text(self, *, attributes: dict[str, str], **_: object) -> _Writer:
                header = dict(attributes)
                await room.open_gate
                return _Writer(header)

        self.local_participant = _Local()

    def isconnected(self) -> bool:
        return True

    def on(self, *_: object) -> None: ...

    def off(self, *_: object) -> None: ...


async def test_a_language_change_while_the_stream_opens_closes_it_as_the_old_language() -> None:
    from livekit.agents.voice.room_io._output import _ParticipantStreamTranscriptionOutput

    room = _GatedTextRoom()
    sink = _ParticipantStreamTranscriptionOutput(
        room,  # type: ignore[arg-type]
        is_delta_stream=True,
        participant="alice",
        track="TR_a",
        attributes={"lk.transcription_language": "es"},
    )
    try:
        capture = asyncio.create_task(sink.capture_text("hola"))
        await asyncio.sleep(0)
        # the Spanish stream is still opening when French is set
        sink.set_attributes({"lk.transcription_language": "fr"})
        room.open_gate.set_result(None)
        await capture

        await sink.capture_text("bonjour")
        sink.flush()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        spanish, french = room.streams
        assert spanish["header"]["lk.transcription_language"] == "es"  # type: ignore[index]
        assert spanish["text"] == ["hola"]
        assert spanish["closed_with"]["lk.transcription_language"] == "es"  # type: ignore[index]
        # French opened its own stream rather than reusing the Spanish one
        assert french["header"]["lk.transcription_language"] == "fr"  # type: ignore[index]
        assert french["text"] == ["bonjour"]
    finally:
        await sink.aclose()


async def test_an_unstamped_retarget_before_any_audio_keeps_the_text_waiting_for_it() -> None:
    model = _FakeTranslationModel(retargetable=True)
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        # the translation's text arrives ahead of its audio, which has not come yet
        model.sessions[0].emit("output_transcript", OutputTranscript(text="hola", is_final=False))
        session.set_target_language("fr")
        await asyncio.sleep(0.01)

        # ending the empty segment would rotate the synchronizer and drop "hola", and
        # relabelling now would publish it as French
        assert counts.flushes == 0
        assert labels.languages == []

        # "hola"'s audio comes and its segment ends; only then does the label change
        model.sessions[0].speak()
        model.sessions[0].emit("output_transcript", OutputTranscript(text="", is_final=True))
        await _wait_for(lambda: labels.languages == ["fr"])
        assert counts.flushes == 1
    finally:
        session._room_io = None
        await session.aclose()


async def test_an_unstamped_retarget_ends_speech_held_in_the_resampler() -> None:
    model = _FakeTranslationModel(retargetable=True)
    audio_out = FakeAudioOutput(sample_rate=48000)
    session = TranslationSession(model=model, output_segment_timeout=60)
    session.input.audio = FakeAudioInput()
    session.output.audio = audio_out
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        # a frame too short for the resampler to give anything back yet
        model.sessions[0].speak(_frame(duration=0.001, sample_rate=16000))
        await _wait_for(lambda: session._resampler is not None)
        assert counts.frames == 0

        session.set_target_language("fr")
        await _wait_for(lambda: labels.languages == ["fr"])
        # the held Spanish was drained into its own segment, which ended before the label
        assert counts.flushes == 1
        assert counts.frames >= 1
    finally:
        session._room_io = None
        await session.aclose()


async def test_an_idle_unstamped_retarget_relabels_at_once() -> None:
    model = _FakeTranslationModel(retargetable=True)
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    counts = _Counts(audio_out)
    labels = _LabelRecorder()
    await session.start()
    session._room_io = labels  # type: ignore[assignment]
    try:
        model.sessions[0].emit("output_transcript", OutputTranscript(text="hola", is_final=True))
        model.sessions[0].speak()
        # the Spanish utterance ends, and the session goes quiet
        await _wait_for(lambda: counts.flushes == 1)

        session.set_target_language("fr")
        # nothing of the old language is waiting, so the next utterance starts as French
        await _wait_for(lambda: labels.languages == ["fr"])
        assert counts.flushes == 1
    finally:
        session._room_io = None
        await session.aclose()


async def test_a_deferred_label_does_not_outlive_its_session() -> None:
    model = _FakeTranslationModel(retargetable=True)
    session, _, _, _ = _session(model)
    await session.start()
    model.sessions[0].emit("output_transcript", OutputTranscript(text="hola", is_final=False))
    session.set_target_language("fr")
    await _wait_for(lambda: session._relabel_at_segment_end == "fr")
    await session.aclose()

    # the run it was deferred for is over, so a restart starts clean
    assert session._relabel_at_segment_end is None
    session.set_target_language("de")
    await session.start()
    try:
        assert session._relabel_at_segment_end is None
        assert session._output_language == "de"
    finally:
        await session.aclose()


async def test_the_translation_event_carries_all_of_it() -> None:
    model = _FakeTranslationModel()
    session, *_ = _session(model)
    spoken: list[tuple[str, bool]] = []
    session.on("translation_transcribed", lambda ev: spoken.append((ev.transcript, ev.is_final)))
    await session.start()
    try:
        model_session = model.sessions[0]
        # pieces, then an empty final, as a provider that only sends deltas reports them
        model_session.emit("output_transcript", OutputTranscript(text="hola ", is_final=False))
        model_session.emit("output_transcript", OutputTranscript(text="mundo", is_final=False))
        model_session.emit("output_transcript", OutputTranscript(text="", is_final=True))
        model_session.emit("output_transcript", OutputTranscript(text="adiós", is_final=True))

        assert spoken == [
            ("hola ", False),
            ("hola mundo", False),
            ("hola mundo", True),
            ("adiós", True),
        ]
    finally:
        await session.aclose()


async def test_the_text_segment_ends_with_its_audio_segment() -> None:
    model = _FakeTranslationModel()
    # the transcript goes final well before the speech pauses
    session, _, audio_out, text_out = _session(model, output_segment_timeout=0.1)
    counts = _Counts(audio_out)
    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.emit("output_transcript", OutputTranscript(text="hola", is_final=True))
        model_session.speak()
        await asyncio.sleep(0.02)
        # the text has not ended yet: its audio has not
        assert text_out._messages == []

        await _wait_for(lambda: counts.flushes == 1)
        await _wait_for(lambda: text_out._messages == ["hola"])
    finally:
        await session.aclose()


async def test_text_without_audio_ends_on_its_own_final() -> None:
    model = _FakeTranslationModel()
    session, _, _, text_out = _session(model)
    session.output.audio = None
    await session.start()
    try:
        model.sessions[0].emit("output_transcript", OutputTranscript(text="hola", is_final=True))
        await _wait_for(lambda: text_out._messages == ["hola"])
    finally:
        await session.aclose()


async def test_paced_translation_text_survives_a_late_transcript_final() -> None:
    from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

    model = _FakeTranslationModel()
    audio_sink, text_sink = FakeAudioOutput(), FakeTextOutput()
    synchronizer = TranscriptSynchronizer(
        next_in_chain_audio=audio_sink, next_in_chain_text=text_sink
    )
    session = TranslationSession(model=model, output_segment_timeout=0.05)
    session.input.audio = FakeAudioInput()
    session.output.audio = synchronizer.audio_output
    session.output.transcription = synchronizer.text_output
    await session.start()
    try:
        model_session = model.sessions[0]
        # a provider that sends deltas, and marks the transcript final only after it goes quiet
        model_session.emit("output_transcript", OutputTranscript(text="hola ", is_final=False))
        model_session.emit("output_transcript", OutputTranscript(text="mundo", is_final=False))
        for _ in range(5):
            model_session.speak(_frame(duration=0.02, sample_rate=24000))

        # the silence timer ends the audio segment, and it plays out before the late final
        await asyncio.sleep(0.4)
        model_session.emit("output_transcript", OutputTranscript(text="", is_final=True))

        await _wait_for(lambda: "".join(text_sink._messages).strip() == "hola mundo", timeout=3)
    finally:
        await session.aclose()
        await synchronizer.aclose()


async def test_a_quiet_segment_waits_for_text_that_trails_its_audio() -> None:
    from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

    model = _FakeTranslationModel()
    audio_sink, text_sink = FakeAudioOutput(), FakeTextOutput()
    synchronizer = TranscriptSynchronizer(
        next_in_chain_audio=audio_sink, next_in_chain_text=text_sink
    )
    session = TranslationSession(model=model, output_segment_timeout=0.05)
    session.input.audio = FakeAudioInput()
    session.output.audio = synchronizer.audio_output
    session.output.transcription = synchronizer.text_output
    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.speak(_frame(duration=0.1, sample_rate=24000))
        # the speech goes quiet before any of its transcript arrives
        await asyncio.sleep(0.15)
        model_session.emit("output_transcript", OutputTranscript(text="hola", is_final=False))
        await asyncio.sleep(0.05)
        model_session.emit("output_transcript", OutputTranscript(text="", is_final=True))

        await _wait_for(lambda: "".join(text_sink._messages).strip() == "hola", timeout=3)
        # the next translation starts a segment of its own, rather than taking "hola" in
        model_session.speak(_frame(duration=0.1, sample_rate=24000))
        model_session.emit("output_transcript", OutputTranscript(text="adiós", is_final=True))
        await _wait_for(lambda: text_sink._messages[-1:] == ["adiós"], timeout=3)
    finally:
        await session.aclose()
        await synchronizer.aclose()


async def test_speech_resuming_while_waiting_for_text_carries_the_segment_on() -> None:
    model = _FakeTranslationModel()
    session, _, audio_out, _ = _session(model, output_segment_timeout=0.05)
    counts = _Counts(audio_out)
    await session.start()
    try:
        model_session = model.sessions[0]
        model_session.emit("output_transcript", OutputTranscript(text="hola", is_final=False))
        model_session.speak()
        await asyncio.sleep(0.1)
        # quiet, but its text is still going, so the segment holds
        assert counts.flushes == 0
        model_session.speak()
        model_session.emit("output_transcript", OutputTranscript(text="", is_final=True))
        await _wait_for(lambda: counts.flushes == 1)
    finally:
        await session.aclose()
