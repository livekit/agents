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
import io
import json
import os
import wave
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlencode

import aiohttp
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer, merge_frames

from .log import logger
from .models import DeepgramLanguages, DeepgramModels
from .utils import BasicAudioEnergyFilter

BASE_URL = "https://api.deepgram.com/v1/listen"
BASE_URL_WS = "wss://api.deepgram.com/v1/listen"


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
        no_delay: bool = True,
        endpointing_ms: int = 25,
        filler_words: bool = False,
        keywords: list[Tuple[str, float]] = [],
        profanity_filter: bool = False,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
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

        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")

        if language not in ("en-US", "en") and model in (
            "nova-2-meeting",
            "nova-2-phonecall",
            "nova-2-finance",
            "nova-2-conversationalai",
            "nova-2-voicemail",
            "nova-2-video",
            "nova-2-medical",
            "nova-2-drivethru",
            "nova-2-automotive",
        ):
            logger.warning(
                f"{model} does not support language {language}, falling back to nova-2-general"
            )
            model = "nova-2-general"

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
            sample_rate=48000,
            num_channels=1,
            keywords=keywords,
            profanity_filter=profanity_filter,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def recognize(
        self, buffer: AudioBuffer, *, language: DeepgramLanguages | str | None = None
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

        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        data = io_buffer.getvalue()

        async with self._ensure_session().post(
            url=_to_deepgram_url(recognize_config),
            data=data,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "audio/wav",
            },
        ) as res:
            return prerecorded_transcription_to_speech_event(
                config.language, await res.json()
            )

    def stream(
        self, *, language: DeepgramLanguages | str | None = None
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(config, self._api_key, self._ensure_session())

    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language

        if config.detect_language:
            config.language = None

        return config


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})

    def __init__(
        self,
        opts: STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        max_retry: int = 32,
    ) -> None:
        super().__init__()

        if opts.detect_language and opts.language is None:
            raise ValueError("language detection is not supported in streaming mode")

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._max_retry = max_retry
        self._audio_energy_filter = BasicAudioEnergyFilter(cooldown_seconds=1)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._run(self._max_retry)

    async def _run(self, max_retry: int) -> None:
        """
        Run a single websocket connection to Deepgram and make sure to reconnect
        when something went wrong.
        """

        retry_count = 0
        while self._input_ch.qsize() or not self._input_ch.closed:
            try:
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

                headers = {"Authorization": f"Token {self._api_key}"}
                ws = await self._session.ws_connect(
                    _to_deepgram_url(live_config, websocket=True), headers=headers
                )
                retry_count = 0  # connected successfully, reset the retry_count

                await self._run_ws(ws)
            except Exception as e:
                if self._session.closed:
                    break

                if retry_count >= max_retry:
                    logger.exception(
                        f"failed to connect to deepgram after {max_retry} tries"
                    )
                    break

                retry_delay = min(retry_count * 2, 10)  # max 10s
                retry_count += 1  # increment after calculating the delay, the first retry should happen directly

                logger.warning(
                    f"deepgram connection failed, retrying in {retry_delay}s",
                    exc_info=e,
                )
                await asyncio.sleep(retry_delay)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """This method could throw ws errors, these are handled inside the _run method"""

        closing_ws = False

        async def keepalive_task():
            # if we want to keep the connection alive even if no audio is sent,
            # Deepgram expects a keepalive message.
            # https://developers.deepgram.com/reference/listen-live#stream-keepalive
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception:
                return

        async def send_task():
            nonlocal closing_ws

            # forward audio to deepgram in chunks of 100ms
            samples_100ms = self._opts.sample_rate // 10
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_100ms,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    has_audio = self._audio_energy_filter.push_frame(frame)
                    if has_audio:
                        await ws.send_bytes(frame.data.tobytes())

            # tell deepgram we are done sending audio/inputs
            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)

        async def recv_task():
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
                    raise Exception("deepgram connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected deepgram message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process deepgram message")

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
            asyncio.create_task(keepalive_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

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
            is_final_transcript = data["is_final"]
            is_endpoint = data["speech_final"]

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
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=alts
                    )
                    self._event_ch.send_nowait(final_event)
                else:
                    interim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=alts
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
    channel = data["results"]["channels"][0]
    dg_alts = channel["alternatives"]

    # Use the detected language if enabled
    # https://developers.deepgram.com/docs/language-detection
    detected_language = channel.get("detected_language")

    return stt.SpeechEvent(
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


def _to_deepgram_url(opts: dict, *, websocket: bool = False) -> str:
    if opts.get("keywords"):
        # convert keywords to a list of "keyword:intensifier"
        opts["keywords"] = [
            f"{keyword}:{intensifier}" for (keyword, intensifier) in opts["keywords"]
        ]

    # lowercase bools
    opts = {k: str(v).lower() if isinstance(v, bool) else v for k, v in opts.items()}
    base_url = BASE_URL_WS if websocket else BASE_URL
    return f"{base_url}?{urlencode(opts, doseq=True)}"
