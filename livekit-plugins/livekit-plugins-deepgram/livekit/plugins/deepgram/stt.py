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
from contextlib import suppress
from dataclasses import dataclass
from typing import List
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from .log import logger
from .models import DeepgramLanguages, DeepgramModels


@dataclass
class STTOptions:
    language: DeepgramLanguages | str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels
    smart_format: bool
    endpointing: int | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: DeepgramLanguages = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model: DeepgramModels = "nova-2-general",
        api_key: str | None = None,
        min_silence_duration: int = 0,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")
        self._api_key = api_key

        self._opts = STTOptions(
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            endpointing=min_silence_duration,
        )

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: DeepgramLanguages | str | None = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        recognize_config = {
            "model": str(config.model),
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
        }
        if config.language:
            recognize_config["language"] = config.language

        # seems like lower after encoding the parameters is needed
        # otherwise Deepgram returns a bad request
        url = (
            f"https://api.deepgram.com/v1/listen?{urlencode(recognize_config).lower()}"
        )

        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        data = io_buffer.getvalue()

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "audio/wav",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, data=data) as res:
                return prerecorded_transcription_to_speech_event(
                    config.language, await res.json()
                )

    def stream(
        self,
        *,
        language: DeepgramLanguages | str | None = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(config, api_key=self._api_key)

    def _sanitize_options(
        self,
        *,
        language: str | None = None,
    ) -> STTOptions:
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
        sample_rate: int = 16000,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__()

        if opts.detect_language and opts.language is None:
            raise ValueError("language detection is not supported in streaming mode")

        self._opts = opts
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._api_key = api_key
        self._speaking = False

        self._session = aiohttp.ClientSession()
        self._queue = asyncio.Queue[rtc.AudioFrame | str]()
        self._event_queue = asyncio.Queue[stt.SpeechEvent | None]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry))

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: List[stt.SpeechEvent] = []

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        self._queue.put_nowait(SpeechStream._CLOSE_MSG)

        if not wait:
            self._main_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._main_task

        await self._session.close()

    async def _run(self, max_retry: int) -> None:
        """
        Run a single websocket connection to Deepgram and make sure to reconnect
        when something went wrong.
        """

        try:
            retry_count = 0
            while not self._closed:
                try:
                    live_config = {
                        "model": self._opts.model,
                        "punctuate": self._opts.punctuate,
                        "smart_format": self._opts.smart_format,
                        "interim_results": self._opts.interim_results,
                        "encoding": "linear16",
                        "sample_rate": self._sample_rate,
                        "vad_events": True,
                        "channels": self._num_channels,
                        "endpointing": self._opts.endpointing,
                    }

                    if self._opts.language:
                        live_config["language"] = self._opts.language

                    headers = {"Authorization": f"Token {self._api_key}"}

                    url = f"wss://api.deepgram.com/v1/listen?{urlencode(live_config).lower()}"
                    ws = await self._session.ws_connect(url, headers=headers)
                    retry_count = 0  # connected successfully, reset the retry_count

                    await self._run_ws(ws)
                except Exception:
                    # Something went wrong, retry the connection
                    if retry_count >= max_retry:
                        logger.exception(
                            f"failed to connect to deepgram after {max_retry} tries"
                        )
                        break

                    retry_delay = min(retry_count * 2, 10)  # max 10s
                    retry_count += 1  # increment after calculating the delay, the first retry should happen directly

                    logger.warning(
                        f"deepgram connection failed, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)
        except Exception:
            logger.exception("deepgram task failed")
        finally:
            self._event_queue.put_nowait(None)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        This method can throw ws errors, these are handled inside the _run method
        """

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
                pass

        async def send_task():
            nonlocal closing_ws
            # forward inputs to deepgram
            # if we receive a close message, signal it to deepgram and break.
            # the recv task will then make sure to process the remaining audio and stop
            while True:
                data = await self._queue.get()
                self._queue.task_done()

                if isinstance(data, rtc.AudioFrame):
                    # TODO(theomonnom): The remix_and_resample method is low quality
                    # and should be replaced with a continuous resampling
                    frame = data.remix_and_resample(
                        self._sample_rate, self._num_channels
                    )
                    await ws.send_bytes(frame.data.tobytes())
                elif data == SpeechStream._CLOSE_MSG:
                    closing_ws = True
                    await ws.send_str(data)  # tell deepgram we are done with inputs
                    break

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

                    raise Exception(
                        "deepgram connection closed unexpectedly"
                    )  # this will trigger a reconnection, see the _run loop

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected deepgram message type %s", msg.type)
                    continue

                try:
                    # received a message from deepgram
                    data = json.loads(msg.data)
                    self._process_stream_event(data)
                except Exception:
                    logger.exception("failed to process deepgram message")

        await asyncio.gather(send_task(), recv_task(), keepalive_task())

    def _end_speech(self) -> None:
        if not self._speaking:
            logger.warning(
                "trying to commit final events without being in the speaking state"
            )
            return

        if len(self._final_events) == 0:
            return

        self._speaking = False

        # combine all final transcripts since the start of the speech
        sentence = ""
        confidence = 0.0
        for f in self._final_events:
            alt = f.alternatives[0]
            sentence += f"{alt.text.strip()} "
            confidence += alt.confidence

        sentence = sentence.rstrip()
        confidence /= len(self._final_events)  # avg. of confidence

        end_event = stt.SpeechEvent(
            type=stt.SpeechEventType.END_OF_SPEECH,
            alternatives=[
                stt.SpeechData(
                    language=str(self._opts.language),
                    start_time=self._final_events[0].alternatives[0].start_time,
                    end_time=self._final_events[-1].alternatives[0].end_time,
                    confidence=confidence,
                    text=sentence,
                )
            ],
        )
        self._event_queue.put_nowait(end_event)
        self._final_events = []

    def _process_stream_event(self, data: dict) -> None:
        assert self._opts.language is not None

        if data["type"] == "SpeechStarted":
            # This is a normal case. Deepgram's SpeechStarted events
            # are not correlated with speech_final or utterance end.
            # It's poossible that we receive two in a row without an endpoint
            # It's also possible we receive a transcript without a SpeechStarted event.
            if self._speaking:
                return

            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_queue.put_nowait(start_event)

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
                    self._event_queue.put_nowait(start_event)

                if is_final_transcript:
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=alts,
                    )
                    self._final_events.append(final_event)
                    self._event_queue.put_nowait(final_event)
                else:
                    interim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=alts,
                    )
                    self._event_queue.put_nowait(interim_event)

            # if we receive an endpoint, only end the speech if
            # we either had a SpeechStarted event or we have a seen
            # a non-empty transcript
            if is_endpoint and self._speaking:
                self._end_speech()
        elif data["type"] == "Metadata":
            pass
        else:
            logger.warning("received unexpected message from deepgram %s", data)

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def live_transcription_to_speech_data(
    language: str,
    data: dict,
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
