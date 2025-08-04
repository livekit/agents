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
    min_remaining_audio: float
    max_text_length: int


class SentenceStreamPacer:
    def __init__(self, *, min_remaining_audio: float = 5.0, max_text_length: int = 300) -> None:
        """
        Controls the pacing of text sent to TTS. It buffers sentences and decides when to flush
        based on remaining audio duration. This may reduce waste from interruptions and improve
        speech quality by sending larger chunks of text with more context.

        Args:
            min_remaining_audio: Minimum remaining audio duration (seconds) before sending next batch.
            max_text_length: Maximum text length sent to TTS at once.
        """
        self._options = StreamPacerOptions(
            min_remaining_audio=min_remaining_audio,
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
        audio_start_time = 0.0
        first_sentence = True

        # check if audio generation stopped based on audio duration change
        prev_audio_duration = 0.0
        prev_check_time = 0.0
        generation_started = False
        generation_stopped = False

        while not self._closing:
            await self._wakeup_event.wait()
            self._wakeup_event.clear()
            if self._wakeup_timer:
                self._wakeup_timer.cancel()
                self._wakeup_timer = None

            if self._closing or (self._input_ended and not self._sentences):
                break

            audio_duration = self._audio_emitter.pushed_duration()
            curr_time = time.time()
            if audio_duration > 0.0 and audio_start_time == 0.0:
                audio_start_time = curr_time

            # check if audio generation stopped
            if curr_time - prev_check_time >= 0.1:
                if prev_audio_duration < audio_duration:
                    generation_started = True
                elif generation_started:
                    generation_stopped = True
                prev_audio_duration = audio_duration
                prev_check_time = curr_time

            remaining_audio = (
                audio_start_time + audio_duration - curr_time if audio_start_time > 0.0 else 0.0
            )
            if first_sentence or (
                generation_stopped and remaining_audio <= self._options.min_remaining_audio
            ):
                batch: list[str] = []
                while self._sentences:
                    batch.append(self._sentences.pop(0))
                    if (
                        first_sentence  # send first sentence immediately
                        or sum(len(s) for s in batch) >= self._options.max_text_length
                    ):
                        break

                if batch:
                    text = " ".join(batch)
                    self._event_ch.send_nowait(TokenData(token=text))
                    logger.debug(
                        "sent text to tts",
                        extra={"text": text, "remaining_audio": remaining_audio},
                    )
                    generation_started = False
                    generation_stopped = False
                    first_sentence = False

            # reset wakeup timer
            if generation_started and not generation_stopped:
                wait_time = 0.2  # check more frequently when generation is in progress
            else:
                wait_time = max(0.5, remaining_audio - self._options.min_remaining_audio)
            self._wakeup_timer = asyncio.get_event_loop().call_later(
                wait_time, self._wakeup_event.set
            )
