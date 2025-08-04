from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .. import utils
from ..log import logger
from ..tokenize import SentenceStream, TokenData
from .tts import AudioEmitter


@dataclass
class StreamPacerOptions:
    min_audio_buffer: float
    max_text_length: int


class SentenceStreamPacer:
    def __init__(self, *, min_audio_buffer: float = 5.0, max_text_length: int = 300) -> None:
        """
        Controls the pacing of text sent to TTS. It buffers text and decides when to flush
        based on audio timing and buffer size. This may reduce waste from interruptions
        and batch larger chunks for better speech quality through increased context.

        Args:
            min_audio_buffer: Minimum audio buffer duration (seconds) before generating next batch.
            max_text_length: Maximum text length sent to TTS at once.
        """
        self._options = StreamPacerOptions(
            min_audio_buffer=min_audio_buffer,
            max_text_length=max_text_length,
        )

    def wrap(self, sent_stream: SentenceStream, audio_emitter: AudioEmitter) -> StreamPacerWrapper:
        return StreamPacerWrapper(
            options=self._options, sent_stream=sent_stream, audio_emitter=audio_emitter
        )


class StreamPacerWrapper(SentenceStream):
    def __init__(
        self,
        sent_stream: SentenceStream,
        audio_emitter: AudioEmitter,
        *,
        options: StreamPacerOptions,
    ) -> None:
        super().__init__()
        self._sent_stream = sent_stream
        self._options = options
        self._audio_emitter = audio_emitter

        self._closing = False
        self._input_ended = False
        self._sentences: list[str] = []
        self._wakeup_event = asyncio.Event()
        self._wakeup_timer: asyncio.TimerHandle | None = None

        self._recv_atask = asyncio.create_task(self._recv_task())
        self._send_atask = asyncio.create_task(self._send_task())
        self._send_atask.add_done_callback(lambda _: self._event_ch.close())

    def push_text(self, text: str) -> None:
        self._sent_stream.push_text(text)

    def flush(self) -> None:
        self._sent_stream.flush()

    def end_input(self) -> None:
        self._sent_stream.end_input()
        self._input_ended = True
        if self._audio_emitter._dst_ch.closed:
            # close the stream if the audio emitter is closed
            self._closing = True
            self._wakeup_event.set()

    async def aclose(self) -> None:
        await self._sent_stream.aclose()
        self._closing = True
        if self._wakeup_timer:
            self._wakeup_timer.cancel()
            self._wakeup_timer = None
        self._wakeup_event.set()

        await utils.aio.cancel_and_wait(self._recv_atask, self._send_atask)

    async def _recv_task(self) -> None:
        try:
            async for ev in self._sent_stream:
                self._sentences.append(ev.token)
                self._wakeup_event.set()
        finally:
            self._input_ended = True

    async def _send_task(self) -> None:
        prev_audio_duration = 0.0
        prev_check_time = 0.0
        generation_started = False
        # mark as stopped if generation started and audio duration is not increasing
        generation_stopped = False

        audio_start_time = 0.0
        last_sent_time = 0.0

        while not self._closing:
            await self._wakeup_event.wait()
            self._wakeup_event.clear()
            if self._wakeup_timer:
                self._wakeup_timer.cancel()
                self._wakeup_timer = None

            if self._closing or (self._input_ended and not self._sentences):
                break

            audio_duration = self._audio_emitter.pushed_duration()
            if audio_duration > 0.0 and audio_start_time == 0.0:
                audio_start_time = time.time()

            # check if audio generation stopped
            if time.time() - prev_check_time >= 0.1:
                if prev_audio_duration < audio_duration:
                    generation_started = True
                elif generation_started:
                    generation_stopped = True
                prev_audio_duration = audio_duration
                prev_check_time = time.time()

            rest_duration = audio_start_time + audio_duration - time.time()
            logger.debug(
                "wakeup",
                extra={
                    "audio_duration": audio_duration,
                    "rest_duration": rest_duration,
                    "audio_recv_started": generation_started,
                    "audio_recv_stopped": generation_stopped,
                },
            )

            if last_sent_time == 0.0 or (
                generation_stopped and rest_duration <= self._options.min_audio_buffer
            ):
                text_buffer = ""  # collect a larger chunk of text for more context
                while self._sentences:
                    ev = self._sentences.pop(0)
                    text_buffer = f"{text_buffer} {ev}"
                    if len(text_buffer) >= self._options.max_text_length:
                        break

                if text_buffer:
                    self._event_ch.send_nowait(TokenData(token=text_buffer))
                    logger.debug(
                        "sent sentence",
                        extra={"text": text_buffer, "len": len(text_buffer)},
                    )
                    generation_started = False
                    generation_stopped = False
                    last_sent_time = time.time()

            # reset wakeup timer
            if generation_started and not generation_stopped:
                wait_time = 0.1
            else:
                wait_time = max(0.5, rest_duration - self._options.min_audio_buffer)
            self._wakeup_timer = asyncio.get_event_loop().call_later(
                wait_time, self._wakeup_event.set
            )
