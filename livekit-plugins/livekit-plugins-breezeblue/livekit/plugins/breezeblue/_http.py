from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError, tts, utils
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, APIConnectOptions, TimedString

from ._pcm import BYTES_PER_SECOND, PcmPause
from ._text import take_prefix, text_segments

if TYPE_CHECKING:
    from .tts import TTS


class _HttpAudio:
    """Join independent HTTP audio and shift timed words by the actual join padding."""

    def __init__(self, emitter: tts.AudioEmitter, pause_ms: int) -> None:
        self._emitter = emitter
        self._pause = PcmPause(pause_ms)
        self.output_bytes = 0
        self._chunk_offset = 0
        self._previous_text = ""
        self._needs_space = False
        self._words: list[TimedString] = []
        self._partial_sample = b""

    def begin(self, text: str) -> None:
        self._pause.begin()
        self._chunk_offset = self.output_bytes
        self._needs_space = bool(self._previous_text) and (
            self._previous_text[-1].isspace() or text[0].isspace()
        )
        self._previous_text = text

    def add_word(self, word: TimedString) -> None:
        caption = str(word)
        if self._needs_space and caption:
            if not caption[0].isspace():
                word = TimedString(
                    text=" " + caption, start_time=word.start_time, end_time=word.end_time
                )
            self._needs_space = False
        self._words.append(word)

    def push(self, pcm: bytes) -> None:
        pcm = self._partial_sample + pcm
        end = len(pcm) - len(pcm) % 2
        self._partial_sample = pcm[end:]
        if not end:
            return
        gap, audio = self._pause.push(pcm[:end])
        self._write_audio(gap)
        self._write_words()
        self._write_audio(audio)

    def finish(self) -> None:
        if self._partial_sample:
            raise APIStatusError("Breeze returned an incomplete PCM sample", retryable=False)
        audio = self._pause.finish()
        self._write_words()
        self._write_audio(audio)

    def _write_audio(self, pcm: bytes) -> None:
        if pcm:
            self._emitter.push(pcm)
            self.output_bytes += len(pcm)

    def _write_words(self) -> None:
        if self._pause.padding is None:
            return
        offset = (self._chunk_offset + self._pause.padding) / BYTES_PER_SECOND
        for word in self._words:
            start, end = word.start_time, word.end_time
            self._emitter.push_timed_transcript(
                TimedString(
                    text=str(word),
                    start_time=start + offset if utils.is_given(start) else start,
                    end_time=end + offset if utils.is_given(end) else end,
                )
            )
        self._words.clear()


class HttpChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        provider: TTS,
        text: str,
        conn_options: APIConnectOptions,
        voice: str,
        options: dict[str, Any],
    ) -> None:
        super().__init__(tts=provider, input_text=text, conn_options=conn_options)
        self._provider = provider
        self._voice = voice
        self._options = options
        self._reported_terminal_error: Exception | None = None

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        if not recoverable:
            self._reported_terminal_error = api_error
        super()._emit_error(api_error, recoverable)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._provider
        url = f"{provider._base_url}/v1/text-to-speech/{self._voice}/stream"
        timed = provider._word_timestamps
        if timed:
            url += "/with-timestamps"
        timeout = aiohttp.ClientTimeout(
            total=None, connect=self._conn_options.timeout, sock_read=self._conn_options.timeout
        )
        remaining = self._input_text
        audio = _HttpAudio(output_emitter, provider._sentence_pause_ms)
        initialized = False
        try:
            while remaining.strip():
                text, remaining = take_prefix(remaining, provider._max_characters)
                if not text.strip():
                    continue
                async with provider._http_session().post(
                    url,
                    params={"output_format": "pcm"},
                    json={
                        "text": text,
                        **self._options,
                        **({"timestamp_mode": provider._timestamp_mode} if timed else {}),
                    },
                    headers=provider._headers,
                    timeout=timeout,
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        await provider._raise_for_status(response, self._conn_options.timeout)
                    content_types = (
                        ("application/x-ndjson",)
                        if timed
                        else (
                            "audio/pcm",
                            "application/octet-stream",
                        )
                    )
                    if response.content_type not in content_types:
                        raise APIStatusError(
                            f"Unexpected Breeze audio content type {response.content_type}",
                            status_code=response.status,
                            retryable=False,
                        )
                    if not initialized:
                        output_emitter.initialize(
                            request_id=response.headers.get("history-item-id") or utils.shortuuid(),
                            sample_rate=provider.sample_rate,
                            num_channels=provider.num_channels,
                            mime_type="audio/pcm",
                            frame_size_ms=20,
                        )
                        initialized = True
                    audio.begin(text)
                    if timed:
                        await self._push_timed(response, audio)
                    else:
                        async for chunk in response.content.iter_any():
                            audio.push(chunk)
                    audio.finish()
            if initialized:
                output_emitter.flush()
        except asyncio.TimeoutError as error:
            raise APITimeoutError(
                "Breeze TTS timed out", retryable=not audio.output_bytes
            ) from error
        except aiohttp.ClientError:
            raise APIConnectionError(
                "Breeze TTS connection failed", retryable=not audio.output_bytes
            ) from None
        except APIStatusError as error:
            if audio.output_bytes:
                error.retryable = False
            raise

    async def _push_timed(self, response: aiohttp.ClientResponse, audio: _HttpAudio) -> None:
        pending = None
        emitted_index = -1
        async for line in response.content:
            if not line.strip():
                continue
            packet = json.loads(line)
            if packet.get("error"):
                raise APIStatusError(
                    "Breeze timing stream failed", retryable=not audio.output_bytes
                )
            for word in packet.get("word_timestamps", []):
                index = word["index"]
                if index <= emitted_index:
                    continue
                if pending and pending["index"] == index:
                    pending["start"] = min(pending["start"], word["start"])
                    pending["end"] = max(pending["end"], word["end"])
                    continue
                if pending:
                    audio.add_word(_timed_word(pending))
                    emitted_index = pending["index"]
                pending = word.copy()
            if pcm := packet.get("audio_base64"):
                audio.push(base64.b64decode(pcm))
        if pending:
            audio.add_word(_timed_word(pending))


def _timed_word(word: dict[str, Any]) -> TimedString:
    text = word["word"]
    if word["index"] and text and text[0].isalnum():
        text = " " + text
    return TimedString(text=text, start_time=word["start"], end_time=word["end"])


class HttpSynthesizeStream(tts.SynthesizeStream):
    """Synthesize complete HTTP phrases and preserve a natural audio boundary."""

    def __init__(self, provider: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=provider, conn_options=replace(conn_options, max_retry=0))
        self._provider = provider
        self._synthesize_conn_options = conn_options
        self._reported_terminal_error: Exception | None = None

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        # Skip an error already emitted on this provider by the chunked stream.
        if api_error is self._reported_terminal_error:
            self._current_attempt_has_error = True
        else:
            super()._emit_error(api_error, recoverable)

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[tts.SynthesizedAudio]) -> None:
        # The inner HTTP synthesis stream reports metrics for each segment.
        async for _ in event_aiter:
            pass

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )
        audio: _HttpAudio | None = None
        stream: tts.ChunkedStream | None = None
        try:
            async for text in text_segments(
                self._input_ch,
                self._provider._max_characters,
                whole_response=self._provider._input_mode == "whole",
                input_chunk_length_schedule=self._provider._input_chunk_length_schedule,
            ):
                if text is None:
                    if audio is not None:
                        emitter.end_segment()
                        audio = None
                    continue
                if audio is None:
                    emitter.start_segment(segment_id=utils.shortuuid())
                    audio = _HttpAudio(emitter, self._provider._sentence_pause_ms)
                self._mark_started()
                audio.begin(text)
                stream = self._provider.synthesize(text, conn_options=self._synthesize_conn_options)
                async with stream:
                    async for event in stream:
                        frame = event.frame
                        for word in frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, ()):
                            audio.add_word(word)
                        audio.push(bytes(frame.data.cast("B")))
                audio.finish()
        except APIError as error:
            if isinstance(stream, HttpChunkedStream) and stream._reported_terminal_error is error:
                self._reported_terminal_error = error
            if audio is not None and audio.output_bytes:
                error.retryable = False
            raise
        if audio is not None:
            emitter.end_segment()
