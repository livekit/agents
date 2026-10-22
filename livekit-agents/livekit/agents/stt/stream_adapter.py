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
                speech holds the event until the next speech end. Empty recognition
                results also release the event, without emitting a FINAL_TRANSCRIPT.
                Defaults to False, preserving immediate VAD end events.
            max_transcript_wait: Maximum seconds to hold END_OF_SPEECH after the latest
                VAD end event, when enabled. Must be finite and non-negative. The cap
                does not cancel recognition; transcripts may still arrive afterward.
                Session endpointing delays are applied separately by AgentSession.
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
            if self._end_of_speech_after_transcript:
                await self._recognize_after_transcript(vad_stream)
                return

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

    async def _recognize_after_transcript(self, vad_stream: VADStream) -> None:
        """Read VAD independently of ordered recognition and bounded end events."""
        clips = utils.aio.Chan[tuple[VADEvent, float]]()
        pending = 0
        speaking = False
        held_end: float | None = None
        timer: asyncio.TimerHandle | None = None

        def _emit_end() -> None:
            nonlocal held_end, timer
            if speaking or held_end is None:
                return
            self._event_ch.send_nowait(
                SpeechEvent(SpeechEventType.END_OF_SPEECH, speech_end_time=held_end)
            )
            held_end = None
            if timer is not None:
                timer.cancel()
                timer = None

        async def _read_vad() -> None:
            nonlocal pending, speaking, held_end, timer
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    speaking = True
                    if timer is not None:
                        timer.cancel()
                        timer = None
                    self._event_ch.send_nowait(SpeechEvent(SpeechEventType.START_OF_SPEECH))
                elif event.type == VADEventType.END_OF_SPEECH:
                    speaking = False
                    held_end = time.time() - event.silence_duration - event.inference_duration
                    pending += 1
                    clips.send_nowait((event, held_end))
                    if timer is not None:
                        timer.cancel()
                    timer = asyncio.get_running_loop().call_later(
                        self._max_transcript_wait, _emit_end
                    )
            clips.close()

        async def _transcribe() -> None:
            nonlocal pending
            async for event, speech_end_time in clips:
                result = await self._wrapped_stt.recognize(
                    buffer=utils.merge_frames(event.frames),
                    language=self._language,
                    conn_options=self._wrapped_stt_conn_options,
                )
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
                pending -= 1
                if pending == 0:
                    _emit_end()

        tasks = [
            asyncio.create_task(_read_vad(), name="read_vad"),
            asyncio.create_task(_transcribe(), name="transcribe"),
        ]
        try:
            await asyncio.gather(*tasks)
            # Graceful input completion drains recognition before releasing a held end.
            speaking = False
            _emit_end()
        finally:
            if timer is not None:
                timer.cancel()
            await utils.aio.cancel_and_wait(*tasks)
            clips.close()
