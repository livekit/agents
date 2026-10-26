from __future__ import annotations

import asyncio
import time
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from livekit import rtc

from ...job import get_job_context
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import aio, is_given, log_exceptions
from ...voice import io
from ...voice.events import (
    CloseEvent,
    CloseReason,
    ErrorEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)
from .room_io import TranslationRoomIO, TranslationRoomOptions
from .translation_model import (
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    OutputSpeechStopped,
    OutputTranscript,
    TranslationModel,
    TranslationModelError,
    TranslationModelSession,
)

Userdata_T = TypeVar("Userdata_T")

EventTypes = Literal[
    "user_state_changed",
    "user_input_transcribed",
    "translation_transcribed",
    "error",
    "close",
]

# a provider that never reports where its speech ends still has to end a playout segment,
# or the translation plays on with nothing ever marking it finished
DEFAULT_OUTPUT_SEGMENT_TIMEOUT = 0.5


# marks, in the transcript stream, where the audio segment the text belongs to ended
_END_OF_SEGMENT = object()


class TranslationTranscribedEvent(BaseModel):
    """The translation the model spoke, in the language it translates into."""

    type: Literal["translation_transcribed"] = "translation_transcribed"
    transcript: str
    """The translation so far, as a user transcript is: each interim carries all of it, and
    the final one the whole of it."""
    is_final: bool
    language: str | None = None
    created_at: float = Field(default_factory=time.time)


class TranslationSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    """Runs a :class:`TranslationModel` against a room: speech in, translation out.

    Its own primitive rather than an ``AgentSession``: a translator holds no conversation. It
    takes no instructions, keeps no chat context, calls no tools, and never decides to reply —
    so there is no ``Agent`` to supply and no turn to take. What is left it does carry: room
    I/O, both transcripts, user and agent state, and a close that reports why.

    A model reports the speaker's turns itself, so no VAD runs here, and nothing waits for the
    speaker to stop before speaking over them.

    Example:
        ```python
        session = TranslationSession(model=VoiceModel(target_language="es"))
        await session.start(room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        model: TranslationModel,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        output_segment_timeout: float = DEFAULT_OUTPUT_SEGMENT_TIMEOUT,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """
        Args:
            model: The translation model this session runs.
            userdata: Arbitrary state carried on the session.
            output_segment_timeout: Seconds of silence from the model that end a segment of
                its speech. Used only when the model's ``capabilities.output_segmentation``
                is unset; a model that marks its own boundaries ends a segment on the mark,
                so a pause inside a sentence does not split it.
            loop: Event loop to run on.
        """
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()
        self._model = model
        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._output_segment_timeout = output_segment_timeout

        self._input = io.AgentInput(self._on_video_input_changed, self._on_audio_input_changed)
        self._output = io.AgentOutput(
            self._on_video_output_changed,
            self._on_audio_output_changed,
            self._on_text_output_changed,
        )

        self._room_io: TranslationRoomIO | None = None
        self._model_session: TranslationModelSession | None = None

        self._user_state: UserState = "listening"

        self._started = False
        self._closing_task: asyncio.Task[None] | None = None
        self._job_context_cb_registered = False
        self._lock = asyncio.Lock()

        self._forward_input_atask: asyncio.Task[None] | None = None
        self._forward_output_atask: asyncio.Task[None] | None = None
        self._output_transcript_ch: aio.Chan[OutputTranscript | object] | None = None
        # the translation transcribed so far, for the event; the model sends it in pieces
        self._output_transcript_text = ""
        self._output_transcript_atask: asyncio.Task[None] | None = None
        self._segment_timer: asyncio.TimerHandle | None = None
        self._segment_tasks: set[asyncio.Task[None]] = set()
        # serialises capturing a frame against ending a segment, so a tail the resampler
        # holds cannot be flushed out from under a frame being captured
        self._output_lock = asyncio.Lock()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_key: tuple[int, int, int | None] | None = None
        self._resampler_sink: io.AudioOutput | None = None
        # the language the room labels the translation with. It follows the language stamped
        # on the speech itself, which trails a retarget until the provider applies it, and
        # follows the request only for a provider that stamps nothing
        self._output_language: str | None = None
        self._output_language_stamped = False
        # whether the open segment has any audio in it; an empty one is not worth ending
        self._segment_has_audio = False
        # whether translation text has arrived since the last segment ended, which the
        # synchronizer holds for audio still to come
        self._segment_has_text = False
        # a label to apply once the open segment ends: requested while it held text waiting for
        # audio that has not come, which is still in the old language
        self._relabel_at_segment_end: str | None = None

    @property
    def model(self) -> TranslationModel:
        """The translation model this session runs."""
        return self._model

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def room_io(self) -> TranslationRoomIO | None:
        return self._room_io

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("TranslationSession userdata is not set")

        return self._userdata

    @userdata.setter
    def userdata(self, value: Userdata_T) -> None:
        self._userdata = value

    def set_target_language(self, language: str) -> None:
        """Point the model at another language.

        Before :meth:`start` this only changes what the next session is opened with. On a
        running session it reaches the model, which a provider serves only when its
        ``capabilities.retargetable``.

        Raises:
            RuntimeError: If the session is running and its model cannot be retargeted.
        """
        if self._model_session is not None and not self._model.capabilities.retargetable:
            raise RuntimeError(
                f"{self._model.label} translates into a language fixed for the session's "
                "life; start a new session to translate into another"
            )

        # the running session first: it can refuse a language the capability allows in
        # general, and the model setting must not claim a language being spoken
        if self._model_session is not None:
            self._model_session.update_options(target_language=language)

        self._model.update_options(target_language=language)
        if self._model_session is None:
            self._relabel_output(language)
        elif not self._output_language_stamped:
            # nothing stamps the speech with its language, so the label follows the request.
            # The open segment ends first: speech that never pauses would otherwise keep it
            # open, and the old label with it
            task = asyncio.create_task(self._close_segment_and_relabel(language))
            self._segment_tasks.add(task)
            task.add_done_callback(self._segment_tasks.discard)

    async def start(
        self,
        *,
        room: NotGivenOr[rtc.Room] = NOT_GIVEN,
        room_options: NotGivenOr[TranslationRoomOptions] = NOT_GIVEN,
    ) -> None:
        """Start translating.

        Args:
            room: The room to listen to and speak into. Omit it to drive the session's
                ``input``/``output`` yourself.
            room_options: How the session reaches that room.
        """
        async with self._lock:
            if self._started:
                return

            job_ctx = get_job_context(required=False)

            if is_given(room) and self._room_io is None:
                options = room_options if is_given(room_options) else TranslationRoomOptions()

                if self.input.audio is not None and options.audio_input:
                    logger.warning(
                        "room audio input is enabled but input.audio is already set, ignoring.."
                    )
                    options.audio_input = False

                if self.output.audio is not None and options.audio_output:
                    logger.warning(
                        "room audio output is enabled but output.audio is already set, ignoring.."
                    )
                    options.audio_output = False

                self._room_io = TranslationRoomIO(self, room, options=options)
                await self._room_io.start()

                if job_ctx:
                    await job_ctx.connect()

            if job_ctx and not self._job_context_cb_registered:
                job_ctx.add_shutdown_callback(
                    lambda: self._aclose_impl(reason=CloseReason.JOB_SHUTDOWN)
                )
                self._job_context_cb_registered = True

            self._model_session = self._model.session()
            self._output_language = self._model.target_language
            self._output_language_stamped = False
            self._reset_segment_state()
            self._model_session.on("input_speech_started", self._on_input_speech_started)
            self._model_session.on("input_speech_stopped", self._on_input_speech_stopped)
            self._model_session.on("input_transcript", self._on_input_transcript)
            self._model_session.on("output_transcript", self._on_output_transcript)
            self._model_session.on("error", self._on_model_error)

            self._output_transcript_ch = aio.Chan[OutputTranscript | object]()
            self._output_transcript_text = ""
            self._output_transcript_atask = asyncio.create_task(
                self._forward_output_transcript_task(self._output_transcript_ch),
                name="_forward_output_transcript_task",
            )

            self._closing_task = None
            self._started = True

            if self.input.audio is not None:
                self._forward_input_atask = asyncio.create_task(
                    self._forward_input_audio_task(), name="_forward_input_audio_task"
                )

            self._forward_output_atask = asyncio.create_task(
                self._forward_output_audio_task(self._model_session),
                name="_forward_output_audio_task",
            )

    async def aclose(self) -> None:
        """Stop translating and release the model and the room."""
        await self._aclose_impl(reason=CloseReason.USER_INITIATED)

    def shutdown(self) -> None:
        """Close the session without waiting for it."""
        self._close_soon(reason=CloseReason.USER_INITIATED)

    def _close_soon(self, *, reason: CloseReason) -> None:
        if self._closing_task:
            return

        self._closing_task = asyncio.create_task(self._aclose_impl(reason=reason))

    @log_exceptions(logger=logger)
    async def _aclose_impl(self, *, reason: CloseReason) -> None:
        async with self._lock:
            if not self._started:
                return

            self._started = False
            self._cancel_segment_timer()
            if self._segment_tasks:
                await aio.cancel_and_wait(*self._segment_tasks)
                self._segment_tasks.clear()
            # a label deferred for this run's segments must not outlive them
            self._reset_segment_state()

            await aio.cancel_and_wait(
                *[
                    task
                    for task in (self._forward_input_atask, self._forward_output_atask)
                    if task is not None
                ]
            )
            self._forward_input_atask = self._forward_output_atask = None

            if self._output_transcript_ch:
                self._output_transcript_ch.close()
                self._output_transcript_ch = None
            if self._output_transcript_atask:
                await aio.cancel_and_wait(self._output_transcript_atask)
                self._output_transcript_atask = None

            if self._model_session:
                await self._model_session.aclose()
                self._model_session = None

            # detach what room io made, since it closes those on the way out and a restart
            # would otherwise reuse the closed devices. A caller's own stay attached: they
            # were never closed here, and losing the reference would strand them
            if (room_io := self._room_io) is not None:
                if self.input.audio is room_io.audio_input:
                    self.input.audio = None
                if self.output.audio is room_io.audio_output:
                    self.output.audio = None
                if self.output.transcription is room_io.transcription_output:
                    self.output.transcription = None

            self.emit("close", CloseEvent(reason=reason))

            if self._room_io:
                await self._room_io.aclose()
                self._room_io = None

            self._discard_resampler()
            self._user_state = "listening"

    # -- model session --

    @log_exceptions(logger=logger)
    async def _forward_input_audio_task(self) -> None:
        audio_input = self.input.audio
        if audio_input is None:
            return

        async for frame in audio_input:
            if (model_session := self._model_session) is not None:
                model_session.push_audio(frame)

    @log_exceptions(logger=logger)
    async def _forward_output_audio_task(self, model_session: TranslationModelSession) -> None:
        marks_segments = self._model.capabilities.output_segmentation
        try:
            async for ev in model_session.audio_stream:
                if isinstance(ev, OutputSpeechStopped):
                    # read in order behind the audio it closes, so nothing it ended is
                    # still waiting to be captured. Honoured even while the output is
                    # muted: whatever was captured before the mute is still a segment, and
                    # dropping the mark would merge the next translation into it
                    await self._close_output_segment()
                    continue

                if ev.language is not None:
                    self._output_language_stamped = True
                    # the speech says what language it is in, so that drives the label
                    self._relabel_at_segment_end = None
                    if ev.language != self._output_language:
                        # the old language's segment ends before speech in the new one, so
                        # neither is labelled with the other's language
                        await self._close_output_segment()
                        self._relabel_output(ev.language)

                if self.output.audio is None or not self.output.audio_enabled:
                    continue

                async with self._output_lock:
                    if (audio_output := self.output.audio) is None:
                        continue

                    if not self._segment_has_audio:
                        # a segment starts once the one before has played, as the sink
                        # requires; the model's speech waits in its stream meanwhile
                        await audio_output.wait_for_playout()

                    await self._capture_output_frame(ev.frame, audio_output)

                if not marks_segments:
                    self._arm_segment_timer()
        finally:
            self._cancel_segment_timer()
            async with self._output_lock:
                await self._drain_resampler()

    @log_exceptions(logger=logger)
    async def _forward_output_transcript_task(
        self, event_ch: aio.Chan[OutputTranscript | object]
    ) -> None:
        async for ev in event_ch:
            transcription_output = self.output.transcription
            if transcription_output is None or not self.output.transcription_enabled:
                continue

            if ev is _END_OF_SEGMENT:
                transcription_output.flush()
                continue

            assert isinstance(ev, OutputTranscript)
            if ev.text:
                await transcription_output.capture_text(ev.text)
            if ev.is_final and not self._text_follows_audio():
                transcription_output.flush()

    def _text_follows_audio(self) -> bool:
        """Whether the translation's text segments end with its audio segments.

        A synchronizer pairs each audio segment with one text segment, and releases the text
        only once both have ended. Ended on its own schedule, the text would close after its
        audio had played, and the words not yet released would be lost. So while audio is
        going out, the text ends where the audio does, marked in the transcript stream so it
        stays behind the text that came before it.
        """
        return self.output.audio is not None and self.output.audio_enabled

    def _relabel_output(self, language: str) -> None:
        """Label the translation in the room with the language it is now spoken in."""
        self._output_language = language
        if self._room_io is not None:
            self._room_io._on_output_language_changed(language)

    def _on_input_speech_started(self, _: InputSpeechStarted) -> None:
        self._update_user_state("speaking")

    def _on_input_speech_stopped(self, _: InputSpeechStopped) -> None:
        self._update_user_state("listening")

    def _on_input_transcript(self, ev: InputTranscript) -> None:
        self.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(
                transcript=ev.text,
                is_final=ev.is_final,
                language=ev.language,
                speaker_id=ev.speaker_id,
            ),
        )

    def _on_output_transcript(self, ev: OutputTranscript) -> None:
        if ev.text:
            self._segment_has_text = True
        if self._output_transcript_ch is not None:
            self._output_transcript_ch.send_nowait(ev)

        # the model sends pieces; the event carries all of it, so a final is never empty
        self._output_transcript_text += ev.text
        transcript = self._output_transcript_text
        if ev.is_final:
            self._output_transcript_text = ""

        self.emit(
            "translation_transcribed",
            TranslationTranscribedEvent(
                transcript=transcript,
                is_final=ev.is_final,
                language=ev.language if ev.language is not None else self._model.target_language,
            ),
        )

    def _on_model_error(self, ev: TranslationModelError) -> None:
        self.emit("error", ErrorEvent(error=ev, source=self._model))

        if not ev.recoverable:
            logger.error(
                "translation model error, closing the session",
                extra={"label": ev.label, "error": str(ev.error)},
            )
            self._close_soon(reason=CloseReason.ERROR)

    # -- output segments --

    async def _capture_output_frame(
        self, frame: rtc.AudioFrame, audio_output: io.AudioOutput
    ) -> None:
        """Hand one frame to the sink, at the rate that sink plays at.

        The resampler is rebuilt whenever the model's format or the sink changes, since one
        built for the old pair would tag frames with a rate the new one does not play.
        """
        target_rate = audio_output.sample_rate
        key = (frame.sample_rate, frame.num_channels, target_rate)

        if self._resampler is not None and (
            self._resampler_key != key or self._resampler_sink is not audio_output
        ):
            if self._resampler_sink is audio_output:
                # same sink, new format: its tail still belongs there, in this segment
                await self._drain_resampler()
            else:
                # the sink was replaced: its segment ends on the sink that received it
                logger.debug("audio output replaced, closing the segment it was given")
                await self._close_segment_locked()

        if target_rate is None or frame.sample_rate == target_rate:
            await audio_output.capture_frame(frame)
            self._segment_has_audio = True
            return

        if self._resampler is None:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=target_rate,
                num_channels=frame.num_channels,
            )
            self._resampler_key = key
            self._resampler_sink = audio_output

        for resampled in self._resampler.push(frame):
            await audio_output.capture_frame(resampled)
            self._segment_has_audio = True

    async def _drain_resampler(self) -> None:
        """Push what the resampler still holds to the sink it was built for, then drop it.

        Not to whatever sink is current: those frames were converted for the old one, and
        another sink would replay them at its own rate.
        """
        resampler, sink = self._resampler, self._resampler_sink
        self._discard_resampler()
        if resampler is None or sink is None:
            return

        for frame in resampler.flush():
            await sink.capture_frame(frame)
            self._segment_has_audio = True

    def _discard_resampler(self) -> None:
        """Forget the resampler and the pair it was built for, dropping anything it holds.

        Used where that tail cannot be placed, or once :meth:`_drain_resampler` has emptied it.
        """
        self._resampler = None
        self._resampler_key = None
        self._resampler_sink = None

    async def _close_output_segment(self) -> None:
        """End the segment the model has been speaking into."""
        self._cancel_segment_timer()
        async with self._output_lock:
            await self._close_segment_locked()

    async def _close_segment_locked(self) -> None:
        """End the segment on the sink that actually received its audio.

        Drains the resampler first: a partial frame it is still holding belongs to the
        segment being closed, not to whatever the model says next. The sink it was built
        for owns the segment, so replacing ``output.audio`` mid-translation closes the old
        sink's segment rather than flushing the replacement's.
        """
        sink = self._resampler_sink or self.output.audio
        await self._drain_resampler()
        if sink is not None:
            sink.flush()

        had_audio, self._segment_has_audio = self._segment_has_audio, False
        self._segment_has_text = False
        if had_audio and self._output_transcript_ch is not None:
            # the text segment ends with its audio, behind the text already sent
            self._output_transcript_ch.send_nowait(_END_OF_SEGMENT)
        if had_audio and (language := self._relabel_at_segment_end) is not None:
            # the text that was waiting on this segment's audio has gone out with it
            self._relabel_at_segment_end = None
            self._relabel_output(language)

    def _arm_segment_timer(self) -> None:
        """Restart the silence that ends a segment, for a model that marks none itself.

        Armed after every frame, so the segment ends only once the model has gone quiet for
        ``output_segment_timeout``.
        """
        self._cancel_segment_timer()
        self._segment_timer = self._loop.call_later(
            self._output_segment_timeout, self._close_output_segment_soon
        )

    def _cancel_segment_timer(self) -> None:
        """Disarm the silence timer, leaving the current segment open."""
        if self._segment_timer is not None:
            self._segment_timer.cancel()
            self._segment_timer = None

    async def _close_segment_and_relabel(self, language: str) -> None:
        """End the open segment, then label the translation with ``language``.

        For a provider that does not stamp its speech: the segment the old label covers has
        to end, or its playout never finishes and the new label never lands. A segment with
        text but no audio yet is left open, since ending it would discard that text, and the
        label changes once it ends. With neither, the label changes now.
        """
        async with self._output_lock:
            # speech the resampler still holds is the old language's too
            await self._drain_resampler()
            if self._segment_has_audio:
                self._cancel_segment_timer()
                await self._close_segment_locked()
            elif self._segment_has_text:
                # the old-language text waits for its audio; the label waits for both
                self._relabel_at_segment_end = language
                return

        self._relabel_at_segment_end = None
        self._relabel_output(language)

    def _reset_segment_state(self) -> None:
        """Forget what the open segment held, and any label deferred until it ends."""
        self._segment_has_audio = False
        self._segment_has_text = False
        self._relabel_at_segment_end = None

    def _close_output_segment_soon(self) -> None:
        """Close the segment from the timer, which cannot await the drain itself.

        The work runs as a task, tracked so closing the session can wait on it.
        """
        self._segment_timer = None
        task = asyncio.create_task(self._close_output_segment())
        self._segment_tasks.add(task)
        task.add_done_callback(self._segment_tasks.discard)

    # -- state --

    def _update_user_state(self, state: UserState) -> None:
        if self._user_state == state:
            return

        old_state = self._user_state
        self._user_state = state
        self.emit("user_state_changed", UserStateChangedEvent(old_state=old_state, new_state=state))

    # -- io --

    def _on_audio_input_changed(self) -> None:
        if not self._started:
            return

        if self._forward_input_atask is not None:
            self._forward_input_atask.cancel()
            self._forward_input_atask = None

        if self.input.audio is not None:
            self._forward_input_atask = asyncio.create_task(
                self._forward_input_audio_task(), name="_forward_input_audio_task"
            )

    def _on_video_input_changed(self) -> None:
        if self.input.video is not None:
            logger.warning("a translation model takes no video input, ignoring..")

    def _on_audio_output_changed(self) -> None: ...

    def _on_video_output_changed(self) -> None: ...

    def _on_text_output_changed(self) -> None: ...
