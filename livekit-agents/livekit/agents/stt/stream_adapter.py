from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from .. import utils
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..vad import VAD, VADEvent, VADEventType, VADStream
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

if TYPE_CHECKING:
    from ..voice.events import ConversationItemAddedEvent

# already a retry mechanism in STT.recognize, don't retry in stream adapter
DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


class _PendingSpeechEnd:
    """Own the opt-in end boundary and its deadline on the stream's event loop.

    Pending clips are counted before recognition starts. Only silence can release
    an end; the deadline may release it before recognition settles. Cancellation
    only closes the timer, whereas a terminal recognition error settles silence.
    """

    def __init__(self, events: utils.aio.Chan[SpeechEvent], max_wait: float) -> None:
        self._events = events
        self._max_wait = max_wait
        self._pending = 0
        self._speaking = False
        self._speech_end_time: float | None = None
        self._deadline = 0.0
        self._timer: asyncio.TimerHandle | None = None

    def speech_started(self) -> None:
        """Suspend end delivery while speech is active."""
        self._speaking = True
        self.close()

    def clip_closed(self, speech_end_time: float) -> None:
        """Count a closed clip and restart the boundary deadline."""
        self._speaking = False
        self._pending += 1
        self._speech_end_time = speech_end_time
        self._deadline = asyncio.get_running_loop().time() + self._max_wait
        self._schedule()

    def recognition_finished(self) -> None:
        """Release a silent boundary after all queued recognition settles."""
        self._pending -= 1
        if self._pending == 0:
            self._emit()

    def speech_reset(self) -> None:
        """Apply an ordered VAD reset and restore the remaining deadline."""
        self._speaking = False
        self._schedule()

    def recognition_failed(self) -> None:
        """Settle silent speech before abandoning queued clips."""
        # Queued clips will be abandoned. Preserve the boundary for text already
        # delivered, but do not invent an end for speech that is still active.
        self._emit()

    def close(self) -> None:
        """Cancel the timer without emitting an end event."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule(self) -> None:
        self.close()
        if self._speaking or self._speech_end_time is None:
            return
        if self._pending == 0:
            self._emit()
        else:
            loop = asyncio.get_running_loop()
            self._timer = loop.call_later(max(0.0, self._deadline - loop.time()), self._emit)

    def _emit(self) -> None:
        if self._speaking or self._speech_end_time is None:
            return
        self._events.send_nowait(
            SpeechEvent(SpeechEventType.END_OF_SPEECH, speech_end_time=self._speech_end_time)
        )
        self._speech_end_time = None
        self.close()


class StreamAdapter(STT):
    def __init__(
        self,
        *,
        stt: STT,
        vad: VAD,
        end_of_speech_after_transcript: bool = False,
        max_transcript_wait: float = 5.0,
    ) -> None:
        """Adapt batch recognition to a VAD-driven stream.

        Args:
            stt: Batch speech recognizer.
            vad: Voice activity detector used to split audio into clips.
            end_of_speech_after_transcript: Hold END_OF_SPEECH until all closed clips
                have been recognized, while continuing to read VAD events. Resumed
                speech holds the event until the next speech end or VAD reset. Empty
                results release it without a FINAL_TRANSCRIPT. Defaults to False,
                preserving immediate VAD end events.
            max_transcript_wait: Maximum seconds to hold END_OF_SPEECH from the latest
                processed VAD END event, when enabled. Must be finite and non-negative.
                Resumed speech cancels the deadline; the next closed clip starts a new
                deadline, so the total hold may exceed this value. A reset without a
                new clip restores the remaining time on the previous deadline.

        Notes:
            The cap does not change recognition timeouts or retries, cancel recognition,
            or bound graceful shutdown. Cap expiry, or speech resumed after an END was
            emitted, can still lead to late text and split turns. Session endpointing
            delays remain in AudioRecognition.

            In this mode, flush ends and drains the current VAD stream before opening
            a fresh one for subsequent audio. This orders the speaking-state reset
            after queued VAD events. Graceful end_input drains clips already closed
            by VAD; open audio may be discarded by the VAD's reset. A terminal
            recognition failure releases a held end only when silent, then propagates
            the error. aclose cancels without releasing a held end.
        """
        if not math.isfinite(max_transcript_wait) or max_transcript_wait < 0:
            raise ValueError("max_transcript_wait must be finite and non-negative")
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=False,
                diarization=False,  # diarization requires streaming STT
                keyterms=stt.capabilities.keyterms,
                chat_context=stt.capabilities.chat_context,
            )
        )
        self._vad = vad
        self._stt = stt
        self._end_of_speech_after_transcript = end_of_speech_after_transcript
        self._max_transcript_wait = max_transcript_wait

        # TODO(theomonnom): The segment_id needs to be populated!
        self._stt.on("metrics_collected", self._on_metrics_collected)

    @property
    def wrapped_stt(self) -> STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, item: ConversationItemAddedEvent) -> None:
        self._stt._push_conversation_item(item)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            self,
            vad=self._vad,
            wrapped_stt=self._stt,
            end_of_speech_after_transcript=self._end_of_speech_after_transcript,
            max_transcript_wait=self._max_transcript_wait,
            language=language,
            conn_options=conn_options,
        )

    def prewarm(self) -> None:
        self._stt.prewarm()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        self._stt.off("metrics_collected", self._on_metrics_collected)


class StreamAdapterWrapper(RecognizeStream):
    def __init__(
        self,
        stt: STT,
        *,
        vad: VAD,
        wrapped_stt: STT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
        end_of_speech_after_transcript: bool = False,
        max_transcript_wait: float = 5.0,
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._vad = vad
        self._wrapped_stt = wrapped_stt
        self._wrapped_stt_conn_options = conn_options
        self._language = language
        self._end_of_speech_after_transcript = end_of_speech_after_transcript
        self._max_transcript_wait = max_transcript_wait

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        async for _ in event_aiter:
            pass

    async def _run(self) -> None:
        if self._end_of_speech_after_transcript:
            await self._run_after_transcript()
            return

        vad_stream = self._vad.stream()

        async def _forward_input() -> None:
            """forward input to vad"""
            async for input in self._input_ch:
                if isinstance(input, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(input)

            vad_stream.end_input()

        async def _recognize() -> None:
            """recognize speech from vad"""
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(SpeechEvent(SpeechEventType.START_OF_SPEECH))
                elif event.type == VADEventType.END_OF_SPEECH:
                    speech_end_time = (
                        time.time() - event.silence_duration - event.inference_duration
                    )
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.END_OF_SPEECH,
                            speech_end_time=speech_end_time,
                        )
                    )

                    merged_frames = utils.merge_frames(event.frames)
                    t_event = await self._wrapped_stt.recognize(
                        buffer=merged_frames,
                        language=self._language,
                        conn_options=self._wrapped_stt_conn_options,
                    )
                    self._mark_provider_progress()

                    if len(t_event.alternatives) == 0:
                        continue
                    elif not t_event.alternatives[0].text:
                        continue

                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[t_event.alternatives[0]],
                            speech_end_time=(
                                t_event.speech_end_time
                                if t_event.speech_end_time is not None
                                else speech_end_time
                            ),
                        )
                    )

        tasks = [
            asyncio.create_task(_forward_input(), name="forward_input"),
            asyncio.create_task(_recognize(), name="recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()

    async def _run_after_transcript(self) -> None:
        """Read VAD independently of ordered recognition and bounded end events."""
        clips = utils.aio.Chan[tuple[VADEvent, float]]()
        pending_end = _PendingSpeechEnd(self._event_ch, self._max_transcript_wait)
        vad_stream: VADStream | None = None

        async def _read_vad(vad_stream: VADStream) -> None:
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    pending_end.speech_started()
                    self._event_ch.send_nowait(SpeechEvent(SpeechEventType.START_OF_SPEECH))
                elif event.type == VADEventType.END_OF_SPEECH:
                    speech_end_time = (
                        time.time() - event.silence_duration - event.inference_duration
                    )
                    pending_end.clip_closed(speech_end_time)
                    clips.send_nowait((event, speech_end_time))
            pending_end.speech_reset()

        async def _forward_segment(vad_stream: VADStream) -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    break
                vad_stream.push_frame(item)
            vad_stream.end_input()

        async def _read_segments() -> None:
            nonlocal vad_stream
            while True:
                vad_stream = self._vad.stream()
                segment_tasks = [
                    asyncio.create_task(_forward_segment(vad_stream), name="forward_segment"),
                    asyncio.create_task(_read_vad(vad_stream), name="read_vad"),
                ]
                try:
                    # A flush may reset VAD without END. End and drain this stream
                    # before resetting our speaking state or forwarding later audio.
                    # This is an ordered reset using only the public VAD interface.
                    await asyncio.gather(*segment_tasks)
                finally:
                    await utils.aio.cancel_and_wait(*segment_tasks)
                    await vad_stream.aclose()
                    vad_stream = None
                if self._input_ch.closed and self._input_ch.qsize() == 0:
                    break
            clips.close()

        async def _transcribe() -> None:
            async for event, speech_end_time in clips:
                try:
                    result = await self._wrapped_stt.recognize(
                        buffer=utils.merge_frames(event.frames),
                        language=self._language,
                        conn_options=self._wrapped_stt_conn_options,
                    )
                except Exception:
                    pending_end.recognition_failed()
                    raise
                if result.alternatives and result.alternatives[0].text:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[result.alternatives[0]],
                            speech_end_time=(
                                result.speech_end_time
                                if result.speech_end_time is not None
                                else speech_end_time
                            ),
                        )
                    )
                pending_end.recognition_finished()

        tasks = [
            asyncio.create_task(_read_segments(), name="read_segments"),
            asyncio.create_task(_transcribe(), name="transcribe"),
        ]
        group = asyncio.gather(*tasks)
        try:
            # Cancel each worker once, so cancellation cannot interrupt the VAD
            # reader's own asynchronous cleanup with a second cancellation.
            await asyncio.shield(group)
        finally:
            pending_end.close()
            await utils.aio.cancel_and_wait(*tasks)
            await asyncio.gather(group, return_exceptions=True)
            # Cancellation may have arrived during a normal segment's cleanup.
            if vad_stream is not None:
                await vad_stream.aclose()
            clips.close()
