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
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
import numpy as np
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
from livekit.agents.utils import AudioBuffer

from ._utils import PeriodicCollector
from .log import logger
from .models import DeepgramLanguages, DeepgramModels

BASE_URL = "https://api.deepgram.com/v1/listen"


# This is the magic number during testing that we use to determine if a frame is loud enough
# to possibly contain speech. It's very conservative.
MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(
        self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD
    ):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                # keep speaking during cooldown
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state


@dataclass
class STTOptions:
    language: DeepgramLanguages | str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels
    smart_format: bool
    no_delay: bool
    endpointing_ms: int
    filler_words: bool
    sample_rate: int
    num_channels: int
    keywords: list[Tuple[str, float]]
    profanity_filter: bool
    energy_filter: AudioEnergyFilter | bool = False


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: DeepgramModels = "nova-2-general",
        language: DeepgramLanguages = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        sample_rate: int = 16000,
        no_delay: bool = True,
        endpointing_ms: int = 25,
        # enable filler words by default to improve turn detector accuracy
        filler_words: bool = True,
        keywords: list[Tuple[str, float]] = [],
        profanity_filter: bool = False,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = BASE_URL,
        energy_filter: AudioEnergyFilter | bool = False,
    ) -> None:
        """
        Create a new instance of Deepgram STT.

        ``api_key`` must be set to your Deepgram API key, either using the argument or by setting
        the ``DEEPGRAM_API_KEY`` environmental variable.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )
        self._base_url = base_url

        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")

        model = _validate_model(model, language)

        self._api_key = api_key

        self._opts = STTOptions(
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            no_delay=no_delay,
            endpointing_ms=endpointing_ms,
            filler_words=filler_words,
            sample_rate=sample_rate,
            num_channels=1,
            keywords=keywords,
            profanity_filter=profanity_filter,
            energy_filter=energy_filter,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: DeepgramLanguages | str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        recognize_config = {
            "model": str(config.model),
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
            "keywords": self._opts.keywords,
            "profanity_filter": config.profanity_filter,
        }
        if config.language:
            recognize_config["language"] = config.language

        try:
            async with self._ensure_session().post(
                url=_to_deepgram_url(recognize_config, self._base_url, websocket=False),
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
        language: DeepgramLanguages | str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: DeepgramLanguages | None = None,
        model: DeepgramModels | None = None,
        interim_results: bool | None = None,
        punctuate: bool | None = None,
        smart_format: bool | None = None,
        sample_rate: int | None = None,
        no_delay: bool | None = None,
        endpointing_ms: int | None = None,
        filler_words: bool | None = None,
        keywords: list[Tuple[str, float]] | None = None,
        profanity_filter: bool | None = None,
    ):
        if language is not None:
            self._opts.language = language
        if model is not None:
            self._opts.model = _validate_model(model, language)
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if punctuate is not None:
            self._opts.punctuate = punctuate
        if smart_format is not None:
            self._opts.smart_format = smart_format
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if no_delay is not None:
            self._opts.no_delay = no_delay
        if endpointing_ms is not None:
            self._opts.endpointing_ms = endpointing_ms
        if filler_words is not None:
            self._opts.filler_words = filler_words
        if keywords is not None:
            self._opts.keywords = keywords
        if profanity_filter is not None:
            self._opts.profanity_filter = profanity_filter

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
                profanity_filter=profanity_filter,
            )

    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language

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
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate
        )

        if opts.detect_language and opts.language is None:
            raise ValueError("language detection is not supported in streaming mode")

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._base_url = base_url
        self._speaking = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

        self._audio_energy_filter: Optional[AudioEnergyFilter] = None
        if opts.energy_filter:
            if isinstance(opts.energy_filter, AudioEnergyFilter):
                self._audio_energy_filter = opts.energy_filter
            else:
                self._audio_energy_filter = AudioEnergyFilter()

        self._pushed_audio_duration = 0.0
        self._request_id = ""
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        language: DeepgramLanguages | None = None,
        model: DeepgramModels | None = None,
        interim_results: bool | None = None,
        punctuate: bool | None = None,
        smart_format: bool | None = None,
        sample_rate: int | None = None,
        no_delay: bool | None = None,
        endpointing_ms: int | None = None,
        filler_words: bool | None = None,
        keywords: list[Tuple[str, float]] | None = None,
        profanity_filter: bool | None = None,
    ):
        if language is not None:
            self._opts.language = language
        if model is not None:
            self._opts.model = _validate_model(model, language)
        if interim_results is not None:
            self._opts.interim_results = interim_results
        if punctuate is not None:
            self._opts.punctuate = punctuate
        if smart_format is not None:
            self._opts.smart_format = smart_format
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if no_delay is not None:
            self._opts.no_delay = no_delay
        if endpointing_ms is not None:
            self._opts.endpointing_ms = endpointing_ms
        if filler_words is not None:
            self._opts.filler_words = filler_words
        if keywords is not None:
            self._opts.keywords = keywords
        if profanity_filter is not None:
            self._opts.profanity_filter = profanity_filter

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse):
            # if we want to keep the connection alive even if no audio is sent,
            # Deepgram expects a keepalive message.
            # https://developers.deepgram.com/reference/listen-live#stream-keepalive
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception:
                return

        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            # forward audio to deepgram in chunks of 50ms
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            last_frame: Optional[rtc.AudioFrame] = None
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    state = self._check_energy_state(data)
                    if state in (
                        AudioEnergyFilter.State.START,
                        AudioEnergyFilter.State.SPEAKING,
                    ):
                        if last_frame:
                            frames.extend(
                                audio_bstream.write(last_frame.data.tobytes())
                            )
                            last_frame = None
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif state == AudioEnergyFilter.State.END:
                        # no need to buffer as we have cooldown period
                        frames = audio_bstream.flush()
                        has_ended = True
                    elif state == AudioEnergyFilter.State.SILENCE:
                        # buffer the last silence frame, since it could contain beginning of speech
                        # TODO: improve accuracy by using a ring buffer with longer window
                        last_frame = data
                elif isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                    has_ended = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                    if has_ended:
                        self._audio_duration_collector.flush()
                        await ws.send_str(SpeechStream._FINALIZE_MSG)
                        has_ended = False

            # tell deepgram we are done sending audio/inputs
            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message="deepgram connection closed unexpectedly"
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
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore

                    # propagate exceptions from completed tasks
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
        live_config = {
            "model": self._opts.model,
            "punctuate": self._opts.punctuate,
            "smart_format": self._opts.smart_format,
            "no_delay": self._opts.no_delay,
            "interim_results": self._opts.interim_results,
            "encoding": "linear16",
            "vad_events": True,
            "sample_rate": self._opts.sample_rate,
            "channels": self._opts.num_channels,
            "endpointing": False
            if self._opts.endpointing_ms == 0
            else self._opts.endpointing_ms,
            "filler_words": self._opts.filler_words,
            "keywords": self._opts.keywords,
            "profanity_filter": self._opts.profanity_filter,
        }

        if self._opts.language:
            live_config["language"] = self._opts.language

        ws = await asyncio.wait_for(
            self._session.ws_connect(
                _to_deepgram_url(live_config, base_url=self._base_url, websocket=True),
                headers={"Authorization": f"Token {self._api_key}"},
            ),
            self._conn_options.timeout,
        )
        return ws

    def _check_energy_state(self, frame: rtc.AudioFrame) -> AudioEnergyFilter.State:
        if self._audio_energy_filter:
            return self._audio_energy_filter.update(frame)
        return AudioEnergyFilter.State.SPEAKING

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

            alts = live_transcription_to_speech_data(self._opts.language, data)
            # If, for some reason, we didn't get a SpeechStarted event but we got
            # a transcript with text, we should start speaking. It's rare but has
            # been observed.
            if len(alts) > 0 and alts[0].text:
                if not self._speaking:
                    self._speaking = True
                    start_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH
                    )
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
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )

        elif data["type"] == "Metadata":
            pass  # metadata is too noisy
        else:
            logger.warning("received unexpected message from deepgram %s", data)


def live_transcription_to_speech_data(
    language: str, data: dict
) -> List[stt.SpeechData]:
    dg_alts = data["channel"]["alternatives"]

    return [
        stt.SpeechData(
            language=language,
            start_time=alt["words"][0]["start"] if alt["words"] else 0,
            end_time=alt["words"][-1]["end"] if alt["words"] else 0,
            confidence=alt["confidence"],
            text=alt["transcript"],
        )
        for alt in dg_alts
    ]


def prerecorded_transcription_to_speech_event(
    language: str | None,  # language should be None when 'detect_language' is enabled
    data: dict,
) -> stt.SpeechEvent:
    # We only support one channel for now
    request_id = data["metadata"]["request_id"]
    channel = data["results"]["channels"][0]
    dg_alts = channel["alternatives"]

    # Use the detected language if enabled
    # https://developers.deepgram.com/docs/language-detection
    detected_language = channel.get("detected_language")

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
            )
            for alt in dg_alts
        ],
    )


def _to_deepgram_url(opts: dict, base_url: str, *, websocket: bool) -> str:
    if opts.get("keywords"):
        # convert keywords to a list of "keyword:intensifier"
        opts["keywords"] = [
            f"{keyword}:{intensifier}" for (keyword, intensifier) in opts["keywords"]
        ]

    # lowercase bools
    opts = {k: str(v).lower() if isinstance(v, bool) else v for k, v in opts.items()}

    if websocket and base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    elif not websocket and base_url.startswith("ws"):
        base_url = base_url.replace("ws", "http", 1)

    return f"{base_url}?{urlencode(opts, doseq=True)}"


def _validate_model(
    model: DeepgramModels, language: DeepgramLanguages | str | None
) -> DeepgramModels:
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
    if language not in ("en-US", "en") and model in en_only_models:
        logger.warning(
            f"{model} does not support language {language}, falling back to nova-2-general"
        )
        return "nova-2-general"
    return model
