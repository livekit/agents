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


class TranslationTranscribedEvent(BaseModel):
    """The translation the model spoke, in the language it translates into."""

    type: Literal["translation_transcribed"] = "translation_transcribed"
    transcript: str
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
        self._output_transcript_ch: aio.Chan[OutputTranscript] | None = None
        self._output_transcript_atask: asyncio.Task[None] | None = None
        self._segment_timer: asyncio.TimerHandle | None = None

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
            self._model_session.on("input_speech_started", self._on_input_speech_started)
            self._model_session.on("input_speech_stopped", self._on_input_speech_stopped)
            self._model_session.on("input_transcript", self._on_input_transcript)
            self._model_session.on("output_transcript", self._on_output_transcript)
            self._model_session.on("error", self._on_model_error)

            self._output_transcript_ch = aio.Chan[OutputTranscript]()
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

            # detach the inputs and outputs: room io closes the ones it made, and a
            # restarted session must not mistake them for sinks the caller supplied
            self.input.audio = None
            self.input.video = None
            self.output.audio = None
            self.output.transcription = None

            self.emit("close", CloseEvent(reason=reason))

            if self._room_io:
                await self._room_io.aclose()
                self._room_io = None

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
        resampler: rtc.AudioResampler | None = None
        marks_segments = self._model.capabilities.output_segmentation

        async def _capture(frame: rtc.AudioFrame, audio_output: io.AudioOutput) -> None:
            nonlocal resampler
            if (
                audio_output.sample_rate is not None
                and frame.sample_rate != audio_output.sample_rate
                and resampler is None
            ):
                resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=audio_output.sample_rate,
                    num_channels=frame.num_channels,
                )

            if resampler is not None:
                for resampled in resampler.push(frame):
                    await audio_output.capture_frame(resampled)
            else:
                await audio_output.capture_frame(frame)

        async def _drain(audio_output: io.AudioOutput) -> None:
            if resampler is not None:
                for resampled in resampler.flush():
                    await audio_output.capture_frame(resampled)

        try:
            async for ev in model_session.audio_stream:
                audio_output = self.output.audio

                if isinstance(ev, OutputSpeechStopped):
                    # read in order behind the audio it closes, so nothing it ended is
                    # still waiting to be captured. Honoured even while the output is
                    # muted: whatever was captured before the mute is still a segment, and
                    # dropping the mark would merge the next translation into it
                    if audio_output is not None:
                        await _drain(audio_output)
                    self._end_output_segment()
                    continue

                if audio_output is None or not self.output.audio_enabled:
                    continue

                await _capture(ev.frame, audio_output)
                if not marks_segments:
                    self._arm_segment_timer()
        finally:
            if (audio_output := self.output.audio) is not None:
                await _drain(audio_output)

    @log_exceptions(logger=logger)
    async def _forward_output_transcript_task(self, event_ch: aio.Chan[OutputTranscript]) -> None:
        async for ev in event_ch:
            transcription_output = self.output.transcription
            if transcription_output is None or not self.output.transcription_enabled:
                continue

            await transcription_output.capture_text(ev.text)
            if ev.is_final:
                transcription_output.flush()

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
        if self._output_transcript_ch is not None:
            self._output_transcript_ch.send_nowait(ev)

        self.emit(
            "translation_transcribed",
            TranslationTranscribedEvent(
                transcript=ev.text,
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

    def _arm_segment_timer(self) -> None:
        self._cancel_segment_timer()
        self._segment_timer = self._loop.call_later(
            self._output_segment_timeout, self._end_output_segment
        )

    def _cancel_segment_timer(self) -> None:
        if self._segment_timer is not None:
            self._segment_timer.cancel()
            self._segment_timer = None

    def _end_output_segment(self) -> None:
        self._cancel_segment_timer()

        if (audio_output := self.output.audio) is not None:
            audio_output.flush()

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
