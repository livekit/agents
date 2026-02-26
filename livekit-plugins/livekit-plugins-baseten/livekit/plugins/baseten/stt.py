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

"""Baseten STT plugin for LiveKit Agents.

This plugin connects to Baseten's Whisper Streaming WebSocket endpoint
(both truss and chain deployments) for real-time speech-to-text.

Endpoint URL formats:
    - Truss:  ``wss://model-{model_id}.api.baseten.co/environments/production/websocket``
    - Chain:  ``wss://chain-{chain_id}.api.baseten.co/environments/production/run_remote``

The WebSocket protocol is:
    1. Connect with ``Authorization: Api-Key <api_key>`` header.
    2. Send a JSON metadata message matching the ``StreamingWhisperInput`` schema.
    3. Stream raw audio bytes (PCM s16le or µ-law).
    4. Receive JSON transcription results (``StreamingWhisperResult``).

Examples::

    from livekit.plugins import baseten

    # Using a truss model endpoint (explicit URL):
    stt = baseten.STT(api_key="...", model_id="abc123")

    # Using a chain endpoint:
    stt = baseten.STT(api_key="...", chain_id="5qe5d2qo")

    # Or pass a full URL directly:
    stt = baseten.STT(
        api_key="...",
        model_endpoint="wss://model-abc123.api.baseten.co/environments/production/websocket",
    )
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import ssl
import weakref
from dataclasses import dataclass
from typing import Literal

import aiohttp
import numpy as np

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger

STTEncoding = Literal["pcm_s16le", "pcm_mulaw"]

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}

ssl_context = ssl._create_unverified_context()


@dataclass
class STTOptions:
    """Options for the Baseten STT plugin.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        buffer_size_seconds: Size of audio buffer in seconds.
        encoding: Audio encoding format (``pcm_s16le`` or ``pcm_mulaw``).
        language: Language code for transcription (e.g. ``en``, ``es``, ``auto``).
        enable_partial_transcripts: Whether to receive interim (partial) transcripts
            before the speaker finishes.  Highly recommended for LiveKit voice agents.
        partial_transcript_interval_s: How often (in seconds) partial transcripts are
            emitted while the speaker is still talking.
        final_transcript_max_duration_s: Maximum duration (in seconds) of audio before
            the server forces a final transcript.
        show_word_timestamps: Request word-level timestamps from Whisper.
        vad_threshold: Server-side VAD speech probability threshold (0.0–1.0).
        vad_min_silence_duration_ms: Minimum silence duration (ms) to mark end of speech.
        vad_speech_pad_ms: Padding (ms) added around detected speech regions.
    """

    sample_rate: int = 16000
    buffer_size_seconds: float = 0.032
    encoding: str = "pcm_s16le"
    language: str = "en"

    # Streaming params – controls how transcripts are delivered
    enable_partial_transcripts: bool = True
    partial_transcript_interval_s: float = 1.0
    final_transcript_max_duration_s: int = 30

    # Whisper params
    show_word_timestamps: bool = True

    # Server-side VAD params (sent as streaming_vad_config)
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 300
    vad_speech_pad_ms: int = 30


class STT(stt.STT):
    """Baseten Speech-to-Text provider.

    Connects to a Baseten Whisper Streaming WebSocket model for real-time
    transcription.  Works with both **truss** and **chain** deployments.

    There are three ways to specify the endpoint (in priority order):

    1. ``model_endpoint`` – pass the full WebSocket URL directly.
    2. ``model_id`` – auto-constructs a **truss** endpoint URL::

           wss://model-{model_id}.api.baseten.co/environments/production/websocket

    3. ``chain_id`` – auto-constructs a **chain** endpoint URL::

           wss://chain-{chain_id}.api.baseten.co/environments/production/run_remote

    If none of the above are provided, the ``BASETEN_MODEL_ENDPOINT`` environment
    variable is used as a fallback.

    Args:
        api_key: Baseten API key.  Falls back to the ``BASETEN_API_KEY`` env var.
        model_endpoint: Full WebSocket URL of the deployed model.  Takes
            priority over ``model_id`` and ``chain_id``.
        model_id: Baseten **truss** model ID.  The plugin builds the endpoint
            URL automatically.  Ignored when ``model_endpoint`` is given.
        chain_id: Baseten **chain** ID.  The plugin builds the endpoint URL
            automatically.  Ignored when ``model_endpoint`` is given.
        sample_rate: Audio sample rate in Hz (default ``16000``).
        encoding: Audio encoding – ``pcm_s16le`` (default) or ``pcm_mulaw``.
        buffer_size_seconds: Audio buffer size in seconds.
        language: BCP-47 language code (default ``en``).  Use ``auto`` for
            automatic language detection.
        enable_partial_transcripts: Emit interim transcripts while the speaker
            is still talking.  Defaults to ``True``.
        partial_transcript_interval_s: Interval (seconds) between partial
            transcript updates.
        final_transcript_max_duration_s: Maximum seconds of audio before the
            server forces a final transcript.
        show_word_timestamps: Include word-level timestamps in results.
        vad_threshold: Server-side VAD threshold (0.0–1.0).
        vad_min_silence_duration_ms: Minimum silence (ms) to end an utterance.
        vad_speech_pad_ms: Padding (ms) around detected speech.
        http_session: Optional :class:`aiohttp.ClientSession` to reuse.
    """

    # URL templates for auto-constructing endpoints from model/chain IDs.
    _TRUSS_URL_TEMPLATE = (
        "wss://model-{model_id}.api.baseten.co/environments/production/websocket"
    )
    _CHAIN_URL_TEMPLATE = (
        "wss://chain-{chain_id}.api.baseten.co/environments/production/run_remote"
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        sample_rate: int = 16000,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        buffer_size_seconds: float = 0.032,
        language: str = "en",
        enable_partial_transcripts: bool = True,
        partial_transcript_interval_s: float = 1.0,
        final_transcript_max_duration_s: int = 30,
        show_word_timestamps: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 300,
        vad_speech_pad_ms: int = 30,
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            ),
        )

        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        self._api_key = api_key

        # Resolve the WebSocket endpoint URL.
        # Priority: model_endpoint > model_id > chain_id > env var
        endpoint = model_endpoint or os.environ.get("BASETEN_MODEL_ENDPOINT")
        if not endpoint:
            if model_id:
                endpoint = self._TRUSS_URL_TEMPLATE.format(model_id=model_id)
            elif chain_id:
                endpoint = self._CHAIN_URL_TEMPLATE.format(chain_id=chain_id)
            else:
                raise ValueError(
                    "A Baseten endpoint is required.  Provide one of: "
                    "model_endpoint, model_id, or chain_id.  "
                    "Alternatively, set the BASETEN_MODEL_ENDPOINT environment variable."
                )

        self._model_endpoint = endpoint

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            language=language,
            enable_partial_transcripts=enable_partial_transcripts,
            partial_transcript_interval_s=partial_transcript_interval_s,
            final_transcript_max_duration_s=final_transcript_max_duration_s,
            show_word_timestamps=show_word_timestamps,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
        )

        if is_given(encoding):
            self._opts.encoding = encoding

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Baseten"

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            model_endpoint=self._model_endpoint,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            stream.update_options(
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                language=language,
                buffer_size_seconds=buffer_size_seconds,
            )


class SpeechStream(stt.SpeechStream):
    """A streaming speech-to-text session connected to Baseten via WebSocket."""

    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        model_endpoint: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._session = http_session
        self._speech_duration: float = 0

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: list[SpeechEvent] = []
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self._reconnect_event.set()

    async def _run(self) -> None:
        """
        Run a single websocket connection to Baseten and make sure to reconnect
        when something went wrong.
        """

        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            samples_per_buffer = 512

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    if len(frame.data) % 2 != 0:
                        logger.warning("Frame data size not aligned to float32 (multiple of 4)")

                    int16_array = np.frombuffer(frame.data, dtype=np.int16)
                    await ws.send_bytes(int16_array.tobytes())

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    if closing_ws:
                        break
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError(
                        "Baseten connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("Unexpected Baseten message type: %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)

                    # Skip non-transcription messages (e.g. error, status)
                    msg_type = data.get("type")
                    if msg_type and msg_type not in ("transcription",):
                        logger.debug("Ignoring message type: %s", msg_type)
                        continue

                    is_final = data.get("is_final", True)
                    segments = data.get("segments", [])

                    # Build transcript text: prefer top-level "transcript" if present,
                    # otherwise concatenate segment texts (Baseten standard format).
                    text = (
                        data.get("transcript")
                        or " ".join(seg.get("text", "") for seg in segments).strip()
                    )

                    confidence = data.get("confidence", 0.0)

                    # Build timed words – prefer word-level timestamps when available,
                    # fall back to segment-level timing.
                    timed_words: list[TimedString] = []
                    for segment in segments:
                        word_timestamps = segment.get("word_timestamps", [])
                        if word_timestamps:
                            for w in word_timestamps:
                                timed_words.append(
                                    TimedString(
                                        text=w.get("word", ""),
                                        start_time=(
                                            w.get("start_time", 0.0) + self.start_time_offset
                                        ),
                                        end_time=(w.get("end_time", 0.0) + self.start_time_offset),
                                        start_time_offset=self.start_time_offset,
                                    )
                                )
                        else:
                            timed_words.append(
                                TimedString(
                                    text=segment.get("text", ""),
                                    start_time=(
                                        segment.get("start_time", 0.0) + self.start_time_offset
                                    ),
                                    end_time=(
                                        segment.get("end_time", 0.0) + self.start_time_offset
                                    ),
                                    start_time_offset=self.start_time_offset,
                                )
                            )

                    start_time = (
                        segments[0].get("start_time", 0.0) if segments else 0.0
                    ) + self.start_time_offset
                    end_time = (
                        segments[-1].get("end_time", 0.0) if segments else 0.0
                    ) + self.start_time_offset

                    if not is_final:
                        if text:
                            event = stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language="",
                                        text=text,
                                        confidence=confidence,
                                        start_time=start_time,
                                        end_time=end_time,
                                        words=timed_words,
                                    )
                                ],
                            )
                            self._event_ch.send_nowait(event)

                    else:
                        language = data.get("language_code", self._opts.language)

                        if text:
                            event = stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        language=language,
                                        text=text,
                                        confidence=confidence,
                                        start_time=start_time,
                                        end_time=end_time,
                                        words=timed_words,
                                    )
                                ],
                            )
                            self._final_events.append(event)
                            self._event_ch.send_nowait(event)

                except Exception:
                    logger.exception("Failed to process message from Baseten")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (asyncio.gather(*tasks), wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket and send the ``StreamingWhisperInput`` metadata message.

        The metadata schema must match the Baseten server's ``StreamingWhisperInput``
        Pydantic model exactly (which uses ``extra="forbid"``).  Field names are:

        - ``whisper_params``  – Whisper model parameters (language, word timestamps, …)
        - ``streaming_params`` – encoding, sample rate, partial transcript settings
        - ``streaming_vad_config`` – server-side Silero VAD configuration
        - ``streaming_diarization_config`` – diarization settings (empty dict to disable)
        """
        headers = {
            "Authorization": f"Api-Key {self._api_key}",
        }

        ws = await self._session.ws_connect(self._model_endpoint, headers=headers, ssl=ssl_context)

        # Build metadata matching Baseten's StreamingWhisperInput schema.
        # See: https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api
        metadata = {
            "whisper_params": {
                "audio_language": self._opts.language,
                "show_word_timestamps": self._opts.show_word_timestamps,
            },
            "streaming_params": {
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
                "enable_partial_transcripts": self._opts.enable_partial_transcripts,
                "partial_transcript_interval_s": self._opts.partial_transcript_interval_s,
                "final_transcript_max_duration_s": self._opts.final_transcript_max_duration_s,
            },
            "streaming_vad_config": {
                "threshold": self._opts.vad_threshold,
                "min_silence_duration_ms": self._opts.vad_min_silence_duration_ms,
                "speech_pad_ms": self._opts.vad_speech_pad_ms,
            },
            "streaming_diarization_config": {},
        }

        await ws.send_str(json.dumps(metadata))
        return ws
