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
import dataclasses
import json
import os
import weakref
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, cast

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.utils.codecs import AudioEncoding as EncoderAudioEncoding, AudioStreamEncoder
from livekit.agents.voice.io import TimedString

from ._utils import PeriodicCollector, _to_deepgram_url
from .log import logger
from .models import DeepgramLanguages, DeepgramModels

AudioEncoding = Literal["linear16", "opus", "mp3"]


@dataclass
class STTOptions:
    language: DeepgramLanguages | str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels | str
    smart_format: bool
    no_delay: bool
    endpointing_ms: int
    enable_diarization: bool
    filler_words: bool
    sample_rate: int
    num_channels: int
    keywords: list[tuple[str, float]]
    keyterms: list[str]
    profanity_filter: bool
    endpoint_url: str
    encoding: AudioEncoding = "linear16"
    vad_events: bool = True
    numerals: bool = False
    mip_opt_out: bool = False
    tags: NotGivenOr[list[str]] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: DeepgramModels | str = "nova-3",
        language: DeepgramLanguages | str = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = False,
        sample_rate: int = 16000,
        no_delay: bool = True,
        endpointing_ms: int = 25,
        enable_diarization: bool = False,
        # enable filler words by default to improve turn detector accuracy
        filler_words: bool = True,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        profanity_filter: bool = False,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.deepgram.com/v1/listen",
        numerals: bool = False,
        mip_opt_out: bool = False,
        vad_events: bool = True,
        encoding: AudioEncoding = "linear16",
        record_audio: bool = False,
    ) -> None:
        """Create a new instance of Deepgram STT.

        Args:
            model: The Deepgram model to use for speech recognition. Defaults to "nova-3".
            language: The language code for recognition. Defaults to "en-US".
            detect_language: Whether to enable automatic language detection. Defaults to False.
            interim_results: Whether to return interim (non-final) transcription results. Defaults to True.
            punctuate: Whether to add punctuations to the transcription. Defaults to True. Turn detector will work better with punctuations.
            smart_format: Whether to apply smart formatting to numbers, dates, etc. Defaults to False.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            no_delay: When smart_format is used, ensures it does not wait for sequence to be complete before returning results. Defaults to True.
            endpointing_ms: Time in milliseconds of silence to consider end of speech. Set to 0 to disable. Defaults to 25.
            filler_words: Whether to include filler words (um, uh, etc.) in transcription. Defaults to True.
            keywords: List of tuples containing keywords and their boost values for improved recognition.
                     Each tuple should be (keyword: str, boost: float). Defaults to None.
                     `keywords` does not work with Nova-3 models. Use `keyterms` instead.
            keyterms: List of key terms to improve recognition accuracy. Defaults to None.
                     `keyterms` is supported by Nova-3 models.
            tags: List of tags to add to the requests for usage reporting. Defaults to NOT_GIVEN.
            profanity_filter: Whether to filter profanity from the transcription. Defaults to False.
            api_key: Your Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for Deepgram API. Defaults to "https://api.deepgram.com/v1/listen".
            numerals: Whether to include numerals in the transcription. Defaults to False.
            mip_opt_out: Whether to take part in the model improvement program
            vad_events: Whether to enable VAD (Voice Activity Detection) events.
                       When enabled, SpeechStarted events are sent when speech is detected. Defaults to True.
            encoding: Audio encoding format to use. Defaults to "linear16" (raw PCM).
                     Supported values: "linear16" (raw PCM), "opus" (Opus in OGG), "mp3".
                     Using "opus" or "mp3" reduces bandwidth but requires encoding overhead.
            record_audio: Whether to record the audio. Defaults to False.
        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the DEEPGRAM_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                diarization=enable_diarization,
                aligned_transcript="word",
            )
        )

        deepgram_api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required")
        self._api_key = deepgram_api_key

        model = _validate_model(model, language)
        _validate_keyterms(model, language, keyterms, keywords)

        self._opts = STTOptions(
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            no_delay=no_delay,
            endpointing_ms=endpointing_ms,
            enable_diarization=enable_diarization,
            filler_words=filler_words,
            sample_rate=sample_rate,
            num_channels=1,
            keywords=keywords if is_given(keywords) else [],
            keyterms=keyterms if is_given(keyterms) else [],
            profanity_filter=profanity_filter,
            numerals=numerals,
            mip_opt_out=mip_opt_out,
            vad_events=vad_events,
            tags=_validate_tags(tags) if is_given(tags) else [],
            endpoint_url=base_url,
            encoding=encoding,
        )
        self._session = http_session
        self._recording = record_audio
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        recognize_config = {
            "model": str(config.model),
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
            "keywords": self._opts.keywords,
            "profanity_filter": config.profanity_filter,
            "numerals": config.numerals,
            "mip_opt_out": config.mip_opt_out,
        }
        if config.enable_diarization:
            logger.warning("speaker diarization is not supported in non-streaming mode, ignoring")

        if config.language:
            recognize_config["language"] = config.language

        try:
            async with self._ensure_session().post(
                url=_to_deepgram_url(recognize_config, self._opts.endpoint_url, websocket=False),
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                return prerecorded_transcription_to_speech_event(
                    config.language,
                    await res.json(),
                )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._opts.endpoint_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[DeepgramModels | str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[AudioEncoding] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(model):
            self._opts.model = _validate_model(model, language)
        if is_given(interim_results):
            self._opts.interim_results = interim_results
        if is_given(punctuate):
            self._opts.punctuate = punctuate
        if is_given(smart_format):
            self._opts.smart_format = smart_format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(no_delay):
            self._opts.no_delay = no_delay
        if is_given(endpointing_ms):
            self._opts.endpointing_ms = endpointing_ms
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(filler_words):
            self._opts.filler_words = filler_words
        if is_given(keywords):
            self._opts.keywords = keywords
        if is_given(keyterms):
            self._opts.keyterms = keyterms
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(vad_events):
            self._opts.vad_events = vad_events
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(encoding):
            self._opts.encoding = cast(AudioEncoding, encoding)

        for stream in self._streams:
            stream.update_options(
                language=language,
                model=model,
                interim_results=interim_results,
                punctuate=punctuate,
                smart_format=smart_format,
                sample_rate=sample_rate,
                no_delay=no_delay,
                endpointing_ms=endpointing_ms,
                filler_words=filler_words,
                keywords=keywords,
                keyterms=keyterms,
                profanity_filter=profanity_filter,
                numerals=numerals,
                mip_opt_out=mip_opt_out,
                vad_events=vad_events,
                endpoint_url=endpoint_url,
                encoding=encoding,
            )

    def _sanitize_options(
        self, *, language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN
    ) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = language

        if config.detect_language:
            config.language = None

        return config


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
    _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
        if opts.detect_language or opts.language is None:
            raise ValueError(
                "language detection is not supported in streaming mode, "
                "please disable it and specify a language"
            )

        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._opts.endpoint_url = base_url
        self._speaking = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

        self._request_id = ""
        self._reconnect_event = asyncio.Event()

    async def _send_pcm(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send raw PCM audio frames to the websocket."""
        import numpy as np

        samples_50ms = self._opts.sample_rate // 20
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            samples_per_channel=samples_50ms,
        )

        has_ended = False
        audio_buffer = []
        try:
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    if self._stt._recording:
                        audio_buffer.append(np.frombuffer(frame.data.tobytes(), dtype=np.int16))
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                    if has_ended:
                        self._audio_duration_collector.flush()
                        await ws.send_str(SpeechStream._FINALIZE_MSG)
                        has_ended = False
        finally:
            try:
                import soundfile as sf

                if self._request_id:
                    logger.info(f"Recording audio to {self._request_id}.wav")
                    sf.write(
                        f"{self._request_id}.wav",
                        np.concatenate(audio_buffer),
                        self._opts.sample_rate,
                    )
                else:
                    logger.info("No request ID, not recording audio")
            except Exception as e:
                logger.exception(f"Failed to record audio: {e}")

    async def _send_encoded(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send encoded audio (opus/mp3) to the websocket."""
        # Map our encoding names to the encoder's format names
        encoder_format: EncoderAudioEncoding = "opus" if self._opts.encoding == "opus" else "mp3"

        encoder = AudioStreamEncoder(
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            format=encoder_format,
        )

        async def encoder_consumer() -> None:
            audio_buffer = []
            try:
                async for chunk in encoder:
                    if self._stt._recording:
                        audio_buffer.append(chunk.data)
                    await ws.send_bytes(chunk.data)
            except BaseException:
                if self._request_id:
                    if encoder_format == "opus":
                        logger.info(f"Recording audio to {self._request_id}.ogg")
                        with open(f"{self._request_id}.ogg", "wb") as f:
                            f.write(b"".join(audio_buffer))
                    else:
                        logger.info(f"Recording audio to {self._request_id}.mp3")
                        with open(f"{self._request_id}.mp3", "wb") as f:
                            f.write(b"".join(audio_buffer))
                raise

        consumer_task = asyncio.create_task(encoder_consumer())

        try:
            has_ended = False
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    self._audio_duration_collector.push(data.duration)
                    encoder.push(data)
                elif isinstance(data, self._FlushSentinel):
                    has_ended = True
                    self._audio_duration_collector.flush()

                if has_ended:
                    encoder.end_input()
                    await consumer_task
                    consumer_task = asyncio.create_task(encoder_consumer())
                    await ws.send_str(SpeechStream._FINALIZE_MSG)
                    encoder = AudioStreamEncoder(
                        sample_rate=self._opts.sample_rate,
                        num_channels=self._opts.num_channels,
                        format=encoder_format,
                    )
                    has_ended = False

            encoder.end_input()
            await consumer_task
        finally:
            await utils.aio.gracefully_cancel(consumer_task)
            await encoder.aclose()

    def update_options(
        self,
        *,
        language: NotGivenOr[DeepgramLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[DeepgramModels | str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        smart_format: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        no_delay: NotGivenOr[bool] = NOT_GIVEN,
        endpointing_ms: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        filler_words: NotGivenOr[bool] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        numerals: NotGivenOr[bool] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        vad_events: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[AudioEncoding] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(model):
            self._opts.model = _validate_model(model, language)
        if is_given(interim_results):
            self._opts.interim_results = interim_results
        if is_given(punctuate):
            self._opts.punctuate = punctuate
        if is_given(smart_format):
            self._opts.smart_format = smart_format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(no_delay):
            self._opts.no_delay = no_delay
        if is_given(endpointing_ms):
            self._opts.endpointing_ms = endpointing_ms
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(filler_words):
            self._opts.filler_words = filler_words
        if is_given(keywords):
            self._opts.keywords = keywords
        if is_given(keyterms):
            self._opts.keyterms = keyterms
        if is_given(profanity_filter):
            self._opts.profanity_filter = profanity_filter
        if is_given(numerals):
            self._opts.numerals = numerals
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(vad_events):
            self._opts.vad_events = vad_events
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(encoding):
            self._opts.encoding = cast(AudioEncoding, encoding)

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # if we want to keep the connection alive even if no audio is sent,
            # Deepgram expects a keepalive message.
            # https://developers.deepgram.com/reference/listen-live#stream-keepalive
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception as e:
                logger.warning(f"Deepgram keepalive task exited: {e}")
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            if self._opts.encoding == "linear16":
                await self._send_pcm(ws)
            else:
                await self._send_encoded(ws)

            # tell deepgram we are done sending audio/inputs
            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # close is expected, see SpeechStream.aclose
                    # or when the agent session ends, the http session is closed
                    if closing_ws or self._session.closed:
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message=f"deepgram connection closed unexpectedly: "
                        f"code={ws.close_code}, reason={msg.extra if msg.extra else 'no reason provided'}"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected deepgram message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process deepgram message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(keepalive_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        live_config: dict[str, Any] = {
            "model": self._opts.model,
            "punctuate": self._opts.punctuate,
            "smart_format": self._opts.smart_format,
            "no_delay": self._opts.no_delay,
            "interim_results": self._opts.interim_results,
            "vad_events": self._opts.vad_events,
            "sample_rate": self._opts.sample_rate,
            "channels": self._opts.num_channels,
            "endpointing": False if self._opts.endpointing_ms == 0 else self._opts.endpointing_ms,
            "filler_words": self._opts.filler_words,
            "profanity_filter": self._opts.profanity_filter,
            "numerals": self._opts.numerals,
            "mip_opt_out": self._opts.mip_opt_out,
        }
        # skip encoding for containerized formats (mp3, ogg-opus)
        if self._opts.encoding == "linear16":
            live_config["encoding"] = "linear16"
        if self._opts.enable_diarization:
            live_config["diarize"] = True
        if self._opts.keywords:
            live_config["keywords"] = self._opts.keywords
        if self._opts.keyterms:
            # the query param is `keyterm`
            # See: https://developers.deepgram.com/docs/keyterm
            live_config["keyterm"] = self._opts.keyterms

        if self._opts.language:
            live_config["language"] = self._opts.language

        if self._opts.tags:
            live_config["tag"] = self._opts.tags

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    _to_deepgram_url(live_config, base_url=self._opts.endpoint_url, websocket=True),
                    headers={"Authorization": f"Token {self._api_key}"},
                ),
                self._conn_options.timeout,
            )
            ws_headers = {
                k: v for k, v in ws._response.headers.items() if k.startswith("dg-") or k == "Date"
            }
            logger.debug(
                "Established new Deepgram STT WebSocket connection:",
                extra={"headers": ws_headers},
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to deepgram") from e
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    def _process_stream_event(self, data: dict) -> None:
        assert self._opts.language is not None

        if data["type"] == "SpeechStarted":
            # This is a normal case. Deepgram's SpeechStarted events
            # are not correlated with speech_final or utterance end.
            # It's possible that we receive two in a row without an endpoint
            # It's also possible we receive a transcript without a SpeechStarted event.
            if self._speaking:
                return

            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_ch.send_nowait(start_event)

        # see this page:
        # https://developers.deepgram.com/docs/understand-endpointing-interim-results#using-endpointing-speech_final
        # for more information about the different types of events
        elif data["type"] == "Results":
            metadata = data["metadata"]
            request_id = metadata["request_id"]
            is_final_transcript = data["is_final"]
            is_endpoint = data["speech_final"]
            self._request_id = request_id

            alts = live_transcription_to_speech_data(
                self._opts.language,
                data,
                is_final=is_final_transcript,
                start_time_offset=self.start_time_offset,
            )
            # If, for some reason, we didn't get a SpeechStarted event but we got
            # a transcript with text, we should start speaking. It's rare but has
            # been observed.
            if len(alts) > 0 and alts[0].text:
                if not self._speaking:
                    self._speaking = True
                    start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    self._event_ch.send_nowait(start_event)

                if is_final_transcript:
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=alts,
                    )
                    self._event_ch.send_nowait(final_event)
                else:
                    interim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=alts,
                    )
                    self._event_ch.send_nowait(interim_event)

            # if we receive an endpoint, only end the speech if
            # we either had a SpeechStarted event or we have a seen
            # a non-empty transcript (deepgram doesn't have a SpeechEnded event)
            if is_endpoint and self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        elif data["type"] == "Metadata":
            pass  # metadata is too noisy
        else:
            logger.warning("received unexpected message from deepgram %s", data)


def live_transcription_to_speech_data(
    language: str, data: dict, *, is_final: bool, start_time_offset: float
) -> list[stt.SpeechData]:
    dg_alts = data["channel"]["alternatives"]

    speech_data = []
    for alt in dg_alts:
        if is_final:
            speakers = [word["speaker"] for word in alt["words"] if "speaker" in word]
            speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
        else:
            # interim result doesn't have correct speaker information?
            speaker = None

        sd = stt.SpeechData(
            language=language,
            start_time=next((word.get("start", 0) for word in alt["words"]), 0) + start_time_offset,
            end_time=next((word.get("end", 0) for word in alt["words"]), 0) + start_time_offset,
            confidence=alt["confidence"],
            text=alt["transcript"],
            speaker_id=f"S{speaker}" if speaker is not None else None,
            words=[
                TimedString(
                    text=word.get("word", ""),
                    start_time=word.get("start", 0) + start_time_offset,
                    end_time=word.get("end", 0) + start_time_offset,
                    start_time_offset=start_time_offset,
                )
                for word in alt["words"]
            ]
            if alt["words"]
            else None,
        )
        if language == "multi" and "languages" in alt:
            sd.language = alt["languages"][0]  # TODO: handle multiple languages
        speech_data.append(sd)
    return speech_data


def prerecorded_transcription_to_speech_event(
    language: str | None,  # language should be None when 'detect_language' is enabled
    data: dict,
) -> stt.SpeechEvent:
    # We only support one channel for now
    request_id = data["metadata"]["request_id"]
    channel: dict = data["results"]["channels"][0]
    dg_alts = channel["alternatives"]

    # Use the detected language if enabled
    # https://developers.deepgram.com/docs/language-detection
    detected_language = channel.get("detected_language", "")

    return stt.SpeechEvent(
        request_id=request_id,
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=language or detected_language,
                start_time=alt["words"][0]["start"] if alt["words"] else 0,
                end_time=alt["words"][-1]["end"] if alt["words"] else 0,
                confidence=alt["confidence"],
                text=alt["transcript"],
                words=[
                    TimedString(
                        text=word.get("word", ""),
                        start_time=word.get("start", 0),
                        end_time=word.get("end", 0),
                    )
                    for word in alt["words"]
                ],
            )
            for alt in dg_alts
        ],
    )


def _validate_model(
    model: DeepgramModels | str, language: NotGivenOr[DeepgramLanguages | str]
) -> DeepgramModels | str:
    en_only_models = {
        "nova-2-meeting",
        "nova-2-phonecall",
        "nova-2-finance",
        "nova-2-conversationalai",
        "nova-2-voicemail",
        "nova-2-video",
        "nova-2-medical",
        "nova-2-drivethru",
        "nova-2-automotive",
    }
    if is_given(language) and language not in ("en-US", "en") and model in en_only_models:
        logger.warning(
            f"{model} does not support language {language}, falling back to nova-2-general"
        )
        return "nova-2-general"
    return model


def _validate_tags(tags: list[str]) -> list[str]:
    for tag in tags:
        if len(tag) > 128:
            raise ValueError("tag must be no more than 128 characters")
    return tags


def _validate_keyterms(
    model: DeepgramModels | str,
    language: NotGivenOr[DeepgramLanguages | str],
    keyterms: NotGivenOr[list[str]],
    keywords: NotGivenOr[list[tuple[str, float]]],
) -> None:
    """
    Validating keyterms and keywords for model compatibility.
    See: https://developers.deepgram.com/docs/keyterm and https://developers.deepgram.com/docs/keywords
    """
    if model.startswith("nova-3") and is_given(keywords):
        raise ValueError(
            "Keywords is only available for use with Nova-2, Nova-1, Enhanced, and "
            "Base speech to text models. For Nova-3, use Keyterm Prompting."
        )

    if is_given(keyterms) and (not model.startswith("nova-3")):
        raise ValueError(
            "Keyterm Prompting is only available for transcription using the Nova-3 Model. "
            "To boost recognition of keywords using another model, use the Keywords feature."
        )
