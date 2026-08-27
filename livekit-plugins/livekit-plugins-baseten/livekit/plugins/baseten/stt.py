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

"""Baseten STT plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import ssl
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
import numpy as np

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from . import qwen3_stt
from .log import logger
from .models import STTModels

STTEncoding = Literal["pcm_s16le", "pcm_mulaw"]

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}

ssl_context = ssl._create_unverified_context()


@dataclass
class STTOptions:
    sample_rate: int = 16000
    buffer_size_seconds: float = 0.032
    encoding: str = "pcm_s16le"
    language: LanguageCode = LanguageCode("en")
    language_options: list[LanguageCode] = field(default_factory=list)

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
    _TRUSS_URL_TEMPLATE = "wss://model-{model_id}.api.baseten.co/environments/production/websocket"
    _CHAIN_URL_TEMPLATE = "wss://chain-{chain_id}.api.baseten.co/environments/production/websocket"

    def __init__(
        self,
        *,
        model: STTModels = "whisper",
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        sample_rate: int = 16000,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        buffer_size_seconds: float = 0.032,
        language: NotGivenOr[str] = NOT_GIVEN,
        language_options: list[str] | None = None,
        enable_partial_transcripts: bool = True,
        partial_transcript_interval_s: NotGivenOr[float] = NOT_GIVEN,
        final_transcript_max_duration_s: int = 30,
        show_word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Baseten Speech-to-Text provider.

        Connects to a Baseten Whisper Streaming WebSocket model for real-time
        transcription.  Works with both **truss** and **chain** deployments.

        There are three ways to specify the endpoint (in priority order):

        1. ``model_endpoint`` – pass the full WebSocket URL directly.
        2. ``model_id`` – auto-constructs a **truss** endpoint URL::

               wss://model-{model_id}.api.baseten.co/environments/production/websocket

        3. ``chain_id`` – auto-constructs a **chain** endpoint URL::

               wss://chain-{chain_id}.api.baseten.co/environments/production/websocket

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
            language: For ``whisper``, a BCP-47 code (default ``en``); use
                ``auto`` for detection. For ``qwen3-asr``, a Qwen3-ASR canonical
                language name (``English``, ``Cantonese``) or ``auto``
                (the default) — forcing one also sets the *output* language and
                can translate, so use it only to pin what is actually spoken.
            language_options: Restrict automatic detection to this set of
                language codes, e.g. ``["en", "de"]``.  Detection across all
                supported languages is unreliable on the one- to two-second
                utterances typical of telephony, so scoping it to the languages
                an agent actually supports is usually more accurate than
                ``auto``.  Requires Baseten Whisper runtime v0.5.0 or newer;
                empty (the default) leaves detection unrestricted.
            enable_partial_transcripts: Emit interim transcripts while the speaker
                is still talking.  Defaults to ``True``.
            partial_transcript_interval_s: Interval (seconds) between partial
                transcript updates. Defaults to 1.0 for ``whisper`` and 0.5 for
                ``qwen3-asr``, matching each deployment's own default.
            final_transcript_max_duration_s: Maximum seconds of audio before the
                server forces a final transcript.
            show_word_timestamps: Include word-level timestamps. Defaults to
                on for ``whisper``; off for ``qwen3-asr``, whose aligner is
                opt-in on the deployment (``STREAM_ALIGNER=mms``).
            vad_threshold: Server-side VAD threshold (0.0–1.0).
            vad_min_silence_duration_ms: Minimum silence (ms) to end an
                utterance. Defaults to 300 (``whisper``) / 500 (``qwen3-asr``).
            vad_speech_pad_ms: Padding (ms) around detected speech. Defaults to
                30 (``whisper``) / 100 (``qwen3-asr``).
            http_session: Optional :class:`aiohttp.ClientSession` to reuse.
        """
        self._model_name = model
        self._qwen3: qwen3_stt._Qwen3Backend | None = None

        if model == "qwen3-asr":
            self._qwen3 = qwen3_stt._Qwen3Backend(
                model_endpoint=model_endpoint,
                model_id=model_id,
                chain_id=chain_id,
                api_key=api_key,
                language=language if is_given(language) else "auto",
                interim_results=enable_partial_transcripts,
                partial_transcript_interval_s=(
                    partial_transcript_interval_s
                    if is_given(partial_transcript_interval_s)
                    else 0.5
                ),
                final_transcript_max_duration_s=final_transcript_max_duration_s,
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=(
                    vad_min_silence_duration_ms if is_given(vad_min_silence_duration_ms) else 500
                ),
                vad_speech_pad_ms=vad_speech_pad_ms if is_given(vad_speech_pad_ms) else 100,
                # Off unless asked for: the aligner is opt-in on the deployment
                # (STREAM_ALIGNER), so defaulting it on would advertise a
                # capability the server may silently not honour.
                word_timestamps=show_word_timestamps if is_given(show_word_timestamps) else False,
                http_session=http_session,
            )
            super().__init__(
                capabilities=stt.STTCapabilities(
                    streaming=True,
                    interim_results=enable_partial_transcripts,
                    aligned_transcript="word" if show_word_timestamps else False,
                    offline_recognize=True,
                )
            )
            self._session = http_session
            self._streams = weakref.WeakSet[stt.SpeechStream]()
            return

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
        endpoint: str | None = None
        if model_endpoint:
            endpoint = model_endpoint
        elif model_id:
            endpoint = self._TRUSS_URL_TEMPLATE.format(model_id=model_id)
        elif chain_id:
            endpoint = self._CHAIN_URL_TEMPLATE.format(chain_id=chain_id)
        else:
            endpoint = os.environ.get("BASETEN_MODEL_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "A Baseten endpoint is required.  Provide one of: "
                "model_endpoint, model_id, or chain_id.  "
                "Alternatively, set the BASETEN_MODEL_ENDPOINT environment variable."
            )

        self._model_endpoint = endpoint

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            language=LanguageCode(language if is_given(language) else "en"),
            language_options=[LanguageCode(lang) for lang in language_options or []],
            enable_partial_transcripts=enable_partial_transcripts,
            partial_transcript_interval_s=(
                partial_transcript_interval_s if is_given(partial_transcript_interval_s) else 1.0
            ),
            final_transcript_max_duration_s=final_transcript_max_duration_s,
            show_word_timestamps=(show_word_timestamps if is_given(show_word_timestamps) else True),
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=(
                vad_min_silence_duration_ms if is_given(vad_min_silence_duration_ms) else 300
            ),
            vad_speech_pad_ms=vad_speech_pad_ms if is_given(vad_speech_pad_ms) else 30,
        )

        if is_given(encoding):
            self._opts.encoding = encoding

        self._session = http_session
        self._streams = weakref.WeakSet[stt.SpeechStream]()

    @property
    def model(self) -> str:
        return self._model_name

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
        if self._qwen3 is not None:
            return await self._qwen3.recognize_via_stream(
                self, buffer, language=language, conn_options=conn_options
            )
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        if self._qwen3 is not None:
            qwen3_stream = self._qwen3.make_stream(
                self, language=language, conn_options=conn_options
            )
            self._streams.add(qwen3_stream)
            return qwen3_stream

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
        language_options: NotGivenOr[list[str]] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if self._qwen3 is not None:
            self._qwen3.update_options(
                language=language,
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )
            # These are Whisper-only; say so rather than dropping them quietly.
            for name, value in (
                ("language_options", language_options),
                ("buffer_size_seconds", buffer_size_seconds),
            ):
                if is_given(value):
                    logger.warning(
                        "%s does not apply to model=%r and was ignored", name, self._model_name
                    )
            return
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(language_options):
            self._opts.language_options = [LanguageCode(lang) for lang in language_options]
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            # Only the Whisper stream carries these knobs; a qwen3-asr stream
            # cannot be in here, since that path returns above.
            if not isinstance(stream, SpeechStream):
                continue
            stream.update_options(
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                language=language,
                language_options=language_options,
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
        language_options: NotGivenOr[list[str]] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(language_options):
            self._opts.language_options = [LanguageCode(lang) for lang in language_options]
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

            try:
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
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("Baseten connection closed unexpectedly") from e

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
                                        language=LanguageCode(""),
                                        text=text,
                                        confidence=confidence,
                                        start_time=start_time,
                                        end_time=end_time,
                                        words=timed_words,
                                    )
                                ],
                            )
                            self._event_ch.send_nowait(event)

                    elif is_final:
                        language = LanguageCode(data.get("language_code", self._opts.language))

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
        """
        headers = {
            "Authorization": f"Api-Key {self._api_key}",
        }

        ws = await self._session.ws_connect(self._model_endpoint, headers=headers, ssl=ssl_context)

        # Build metadata matching Baseten's StreamingWhisperInput schema.
        # See: https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api
        metadata: dict[str, dict[str, Any]] = {
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
        }

        if self._opts.language_options:
            # Omitted unless set: older Whisper runtimes (< v0.5.0) reject
            # unknown whisper_params fields.
            metadata["whisper_params"]["language_options"] = [
                str(lang) for lang in self._opts.language_options
            ]

        await ws.send_str(json.dumps(metadata))
        return ws
