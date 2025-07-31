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
    min_audio_duration: float
    max_text_length: int


class SentenceStreamPacer:
    def __init__(self, *, min_audio_duration: float = 5.0, max_text_length: int = 300) -> None:
        """
        Controls the pacing of text sent to TTS. It buffers text and decides when to flush
        based on audio timing and buffer size. This may reduce waste from interruptions
        and batch larger chunks for better speech quality through increased context.

        Args:
            min_audio_duration: Minimum audio buffer duration (seconds) before generating next batch.
            max_text_length: Maximum text length sent to TTS at once.
        """
        self._options = StreamPacerOptions(
            min_audio_duration=min_audio_duration,
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

        self._sentences: list[str] = []
        self._text_changed = asyncio.Event()
        self._closing = False
        self._input_ended = False

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
            self._text_changed.set()

    async def aclose(self) -> None:
        await self._sent_stream.aclose()
        self._closing = True
        self._text_changed.set()

        await utils.aio.cancel_and_wait(self._recv_atask, self._send_atask)

    async def _recv_task(self) -> None:
        try:
            async for ev in self._sent_stream:
                self._sentences.append(ev.token)
                self._text_changed.set()
        finally:
            self._input_ended = True

    async def _send_task(self) -> None:
        prev_audio_duration = 0.0
        audio_recv_started = False
        audio_recv_stopped = False

        audio_start_time = 0.0
        last_sent_time = 0.0

        while not self._closing:
            try:
                await asyncio.wait_for(self._text_changed.wait(), timeout=1)
                self._text_changed.clear()
            except asyncio.TimeoutError:
                pass

            if self._closing or (not self._sentences and self._input_ended):
                break

            audio_duration = self._audio_emitter.pushed_duration()
            if audio_duration > 0.0 and audio_start_time == 0.0:
                audio_start_time = time.time()  # rough time when audio generation started

            if prev_audio_duration < audio_duration:
                audio_recv_started = True
            elif audio_recv_started:
                audio_recv_stopped = True
            prev_audio_duration = audio_duration

            rest_duration = audio_start_time + audio_duration - time.time()
            logger.debug(
                "wakeup",
                extra={
                    "audio_duration": audio_duration,
                    "rest_duration": rest_duration,
                    "audio_recv_started": audio_recv_started,
                    "audio_recv_stopped": audio_recv_stopped,
                },
            )

            if last_sent_time == 0.0 or (
                audio_recv_stopped and rest_duration <= self._options.min_audio_duration
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
                    audio_recv_started = False
                    audio_recv_stopped = False
                    last_sent_time = time.time()

            elif time.time() - last_sent_time > 5.0:
                # send empty token to keep the tts connection alive
                logger.debug("sent empty token")
                self._event_ch.send_nowait(TokenData(token=""))
                last_sent_time = time.time()
