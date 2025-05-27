# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModels,
    TTSVoiceEmotion,
    TTSVoiceSpeed,
)

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2025-04-16"

NUM_CHANNELS = 1
BUFFERED_WORDS_COUNT = 10


@dataclass
class WordTimestamps:
    """Word-level timestamps from Cartesia TTS"""

    words: list[str]
    """List of words"""
    start: list[float]
    """Start times for each word in seconds"""
    end: list[float]
    """End times for each word in seconds"""


@dataclass
class PhonemeTimestamps:
    """Phoneme-level timestamps from Cartesia TTS"""

    phonemes: list[str]
    """List of phoneme strings"""
    start: list[float]
    """Start times for each phoneme in seconds"""
    end: list[float]
    """End times for each phoneme in seconds"""


@dataclass
class TimestampEvent:
    """A timestamp event that occurred at a specific time"""

    timestamp: float
    """When this event occurs in the audio stream (seconds)"""
    word_timestamps: WordTimestamps | None = None
    """Word timestamps if available"""
    phoneme_timestamps: PhonemeTimestamps | None = None
    """Phoneme timestamps if available"""


@dataclass
class CartesiaSynthesizedAudio(tts.SynthesizedAudio):
    """Extended SynthesizedAudio with timestamp support"""

    word_timestamps: WordTimestamps | None = None
    """Word-level timestamps that occur during this frame"""
    phoneme_timestamps: PhonemeTimestamps | None = None
    """Phoneme-level timestamps that occur during this frame"""


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    speed: NotGivenOr[TTSVoiceSpeed | float]
    emotion: NotGivenOr[list[TTSVoiceEmotion | str]]
    api_key: str
    language: str
    base_url: str
    add_timestamps: bool
    add_phoneme_timestamps: bool

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "sonic-2",
        language: str = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str | list[float] = TTSDefaultVoiceId,
        speed: NotGivenOr[TTSVoiceSpeed | float] = NOT_GIVEN,
        emotion: NotGivenOr[list[TTSVoiceEmotion | str]] = NOT_GIVEN,
        sample_rate: int = 24000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        add_timestamps: bool = False,
        add_phoneme_timestamps: bool = False,
    ) -> None:
        """
        Create a new instance of Cartesia TTS.

        See https://docs.cartesia.ai/reference/web-socket/stream-speech/stream-speech for more details on the the Cartesia API.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-2".
            language (str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            api_key (str, optional): The Cartesia API key. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Cartesia API. Defaults to "https://api.cartesia.ai".
            add_timestamps (bool, optional): Whether to request word-level timestamps. Defaults to False.
            add_phoneme_timestamps (bool, optional): Whether to request phoneme-level timestamps. Defaults to False.
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        cartesia_api_key = api_key if is_given(api_key) else os.environ.get("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            speed=speed,
            emotion=emotion,
            api_key=cartesia_api_key,
            base_url=base_url,
            add_timestamps=add_timestamps,
            add_phoneme_timestamps=add_phoneme_timestamps,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/tts/websocket?api_key={self._opts.api_key}&cartesia_version={API_VERSION}"
        )
        return await asyncio.wait_for(session.ws_connect(url), self._conn_options.timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str | list[float]] = NOT_GIVEN,
        speed: NotGivenOr[TTSVoiceSpeed | float] = NOT_GIVEN,
        emotion: NotGivenOr[list[TTSVoiceEmotion | str]] = NOT_GIVEN,
        add_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        add_phoneme_timestamps: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language, voice, speed,
        emotion, and timestamps.
        If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-2".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
            add_timestamps (bool, optional): Whether to request word-level timestamps.
            add_phoneme_timestamps (bool, optional): Whether to request phoneme-level timestamps.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(emotion):
            self._opts.emotion = emotion
        if is_given(add_timestamps):
            self._opts.add_timestamps = add_timestamps
        if is_given(add_phoneme_timestamps):
            self._opts.add_phoneme_timestamps = add_phoneme_timestamps

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(
            tts=self,
            pool=self._pool,
            opts=self._opts,
        )

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )

        json = _to_cartesia_options(self._opts)
        json["transcript"] = self._input_text

        headers = {
            API_AUTH_HEADER: self._opts.api_key,
            API_VERSION_HEADER: API_VERSION,
        }

        try:
            async with self._session.post(
                self._opts.get_http_url("/tts/bytes"),
                headers=headers,
                json=json,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                emitter = CartesiaSynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                )
                async for data, _ in resp.content.iter_chunks():
                    for frame in bstream.write(data):
                        emitter.push(frame)

                for frame in bstream.flush():
                    emitter.push(frame)
                emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ):
        super().__init__(tts=tts)
        self._opts, self._pool = opts, pool
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(
            min_sentence_len=BUFFERED_WORDS_COUNT
        ).stream()

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse):
            base_pkt = _to_cartesia_options(self._opts)
            async for ev in self._sent_tokenizer_stream:
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = request_id
                token_pkt["transcript"] = ev.token + " "
                token_pkt["continue"] = True
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = request_id
            end_pkt["transcript"] = " "
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))

        async def _input_task():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )
            emitter = CartesiaSynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                sample_rate=self._opts.sample_rate,
            )

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Cartesia connection closed unexpectedly",
                        request_id=request_id,
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                emitter._segment_id = segment_id

                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    for frame in audio_bstream.write(b64data):
                        emitter.push(frame)
                elif data.get("word_timestamps") and self._opts.add_timestamps:
                    word_timestamps: dict[str, Any] = data["word_timestamps"]
                    if word_timestamps and isinstance(emitter, CartesiaSynthesizedAudioEmitter):
                        word_timestamps = WordTimestamps(
                            words=word_timestamps["words"],
                            start=word_timestamps["start"],
                            end=word_timestamps["end"],
                        )
                        emitter.add_word_timestamps(word_timestamps)
                elif data.get("phoneme_timestamps") and self._opts.add_phoneme_timestamps:
                    phoneme_timestamps: dict[str, Any] = data["phoneme_timestamps"]
                    if phoneme_timestamps and isinstance(emitter, CartesiaSynthesizedAudioEmitter):
                        phoneme_timestamps = PhonemeTimestamps(
                            phonemes=phoneme_timestamps["phonemes"],
                            start=phoneme_timestamps["start"],
                            end=phoneme_timestamps["end"],
                        )
                        emitter.add_phoneme_timestamps(phoneme_timestamps)
                elif data.get("done"):
                    for frame in audio_bstream.flush():
                        emitter.push(frame)
                    emitter.flush()
                    if segment_id == request_id:
                        # we're not going to receive more frames, end stream
                        break
                else:
                    logger.error("unexpected Cartesia message %s", data)

        async with self._pool.connection() as ws:
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_sentence_stream_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)


class CartesiaSynthesizedAudioEmitter:
    """Utility for buffering and emitting audio frames with timestamp tracking.

    This is a Cartesia-specific version of SynthesizedAudioEmitter that properly
    handles timing-based association of word and phoneme timestamps with audio frames.
    """

    def __init__(
        self,
        *,
        event_ch: utils.aio.Chan[CartesiaSynthesizedAudio],
        request_id: str,
        segment_id: str = "",
        sample_rate: int,
    ) -> None:
        self._event_ch = event_ch
        self._frame: rtc.AudioFrame | None = None
        self._request_id = request_id
        self._segment_id = segment_id
        self._sample_rate = sample_rate
        self._current_time = 0.0
        self._timestamp_events: list[TimestampEvent] = []
        self._has_emitted_text = False

    def push(self, frame: rtc.AudioFrame | None):
        """Emits any buffered frame and stores the new frame for later emission.

        The buffered frame is emitted as not final.
        """
        self._emit_frame(is_final=False)
        self._frame = frame

    def add_word_timestamps(self, word_timestamps: WordTimestamps):
        """Add word timestamp events."""
        for i, word in enumerate(word_timestamps.words):
            if i < len(word_timestamps.start):
                # Create a timestamp event for each word's start time
                event = TimestampEvent(
                    timestamp=word_timestamps.start[i],
                    word_timestamps=WordTimestamps(
                        words=[word],
                        start=[word_timestamps.start[i]],
                        end=(
                            [word_timestamps.end[i]]
                            if i < len(word_timestamps.end)
                            else [word_timestamps.start[i]]
                        ),
                    ),
                )
                self._timestamp_events.append(event)

        # Sort events by timestamp
        self._timestamp_events.sort(key=lambda x: x.timestamp)

    def add_phoneme_timestamps(self, phoneme_timestamps: PhonemeTimestamps):
        """Add phoneme timestamp events."""
        for i, phoneme in enumerate(phoneme_timestamps.phonemes):
            if i < len(phoneme_timestamps.start):
                # Create a timestamp event for each phoneme's start time
                event = TimestampEvent(
                    timestamp=phoneme_timestamps.start[i],
                    phoneme_timestamps=PhonemeTimestamps(
                        phonemes=[phoneme],
                        start=[phoneme_timestamps.start[i]],
                        end=(
                            [phoneme_timestamps.end[i]]
                            if i < len(phoneme_timestamps.end)
                            else [phoneme_timestamps.start[i]]
                        ),
                    ),
                )
                self._timestamp_events.append(event)

        # Sort events by timestamp
        self._timestamp_events.sort(key=lambda x: x.timestamp)

    def _get_frame_timestamps(
        self, frame_duration: float
    ) -> tuple[WordTimestamps | None, PhonemeTimestamps | None]:
        """Get timestamps that should be emitted up to the current playback position."""
        playback_end = self._current_time + frame_duration
        frame_word_events = []
        frame_phoneme_events = []
        remaining_events = []

        # Include all events that start before or at the current playback position
        for event in self._timestamp_events:
            if event.timestamp <= playback_end:
                if event.word_timestamps:
                    frame_word_events.append(event.word_timestamps)
                if event.phoneme_timestamps:
                    frame_phoneme_events.append(event.phoneme_timestamps)
            else:
                remaining_events.append(event)

        self._timestamp_events = remaining_events

        word_timestamps = None
        if frame_word_events:
            all_words = []
            all_starts = []
            all_ends = []
            for word_ts in frame_word_events:
                all_words.extend(word_ts.words)
                all_starts.extend(word_ts.start)
                all_ends.extend(word_ts.end)
            word_timestamps = WordTimestamps(words=all_words, start=all_starts, end=all_ends)

        phoneme_timestamps = None
        if frame_phoneme_events:
            all_phonemes = []
            all_starts = []
            all_ends = []
            for phoneme_ts in frame_phoneme_events:
                all_phonemes.extend(phoneme_ts.phonemes)
                all_starts.extend(phoneme_ts.start)
                all_ends.extend(phoneme_ts.end)
            phoneme_timestamps = PhonemeTimestamps(
                phonemes=all_phonemes, start=all_starts, end=all_ends
            )

        return word_timestamps, phoneme_timestamps

    def _emit_frame(self, is_final: bool = False):
        """Sends the buffered frame to the event channel if one exists."""
        if self._frame is None:
            return

        frame_duration = self._frame.duration
        word_timestamps, phoneme_timestamps = self._get_frame_timestamps(frame_duration)

        # Generate delta_text from word timestamps if available
        delta_text = ""
        if word_timestamps and word_timestamps.words:
            joined_words = " ".join(word_timestamps.words)
            # Add leading space if this is not the first text emission
            if self._has_emitted_text and joined_words:
                delta_text = " " + joined_words
            else:
                delta_text = joined_words

            # Mark that we've emitted text
            if joined_words:
                self._has_emitted_text = True

        self._event_ch.send_nowait(
            CartesiaSynthesizedAudio(
                frame=self._frame,
                request_id=self._request_id,
                segment_id=self._segment_id,
                is_final=is_final,
                delta_text=delta_text,
                word_timestamps=word_timestamps,
                phoneme_timestamps=phoneme_timestamps,
            )
        )

        self._current_time += frame_duration
        self._frame = None

    def flush(self):
        """Emits any buffered frame as final."""
        self._emit_frame(is_final=True)


def _to_cartesia_options(opts: _TTSOptions) -> dict[str, Any]:
    voice: dict[str, Any] = {}
    if is_given(opts.voice):
        if isinstance(opts.voice, str):
            voice["mode"] = "id"
            voice["id"] = opts.voice
        else:
            voice["mode"] = "embedding"
            voice["embedding"] = opts.voice

    voice_controls: dict = {}
    if is_given(opts.speed):
        voice_controls["speed"] = opts.speed
    if is_given(opts.emotion):
        voice_controls["emotion"] = opts.emotion

    if voice_controls:
        voice["__experimental_controls"] = voice_controls

    return {
        "model_id": opts.model,
        "voice": voice,
        "output_format": {
            "container": "raw",
            "encoding": opts.encoding,
            "sample_rate": opts.sample_rate,
        },
        "language": opts.language,
        "add_timestamps": opts.add_timestamps,
        "add_phoneme_timestamps": opts.add_phoneme_timestamps,
    }
