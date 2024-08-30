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
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import List, Optional, Union
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer


ENGLISH = "en"


@dataclass
class STTOptions:
    sample_rate: Optional[int] = None
    word_boost: Optional[str] = None
    encoding: Optional[str] = None  # Allowed values: pcm_s16le, pcm_mulaw
    disable_partial_transcripts: bool = False
    enable_extra_session_information: bool = False
    end_utterance_silence_threshold: Optional[int] = None
    buffer_size_seconds: Optional[int] = None
    token_expires_in: Optional[int] = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: Optional[int] = 16000,
        word_boost: Optional[str] = None,
        encoding: Optional[str] = "pcm_s16le",
        disable_partial_transcripts: bool = False,
        enable_extra_session_information: bool = False,
        end_utterance_silence_threshold: int = 1000,
        http_session: Optional[aiohttp.ClientSession] = None,
        token_expires_in: int = 3600,
        buffer_size_seconds: float = 0.2,
    ):
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if api_key is None:
            raise ValueError("AssemblyAI API key is required. " \
                             "Pass one in via the `api_key` parameter, " \
                             "or set it as the `ASSEMBLYAI_API_KEY` environment variable")
        self._api_key = api_key

        self._opts = STTOptions(
            sample_rate=sample_rate,
            word_boost=word_boost,
            encoding=encoding,
            disable_partial_transcripts=disable_partial_transcripts,
            enable_extra_session_information=enable_extra_session_information,
            buffer_size_seconds=buffer_size_seconds,
            token_expires_in=token_expires_in,
            end_utterance_silence_threshold=end_utterance_silence_threshold,
        )
        self._session = http_session
        self._token_expires_in = token_expires_in

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_session()
        return self._session

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: Optional[str] = None,
    ) -> "SpeechStream":
        config = dataclasses.replace(self._opts)
        return SpeechStream(
            config,
            self._api_key,
            self.session,
        )


class SpeechStream(stt.SpeechStream):
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})

    def __init__(
        self,
        opts: STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        sample_rate: int = 16000,
        num_channels: int = 1,
        max_retry: int = 32,
        token_expires_in: int = 3600,
        buffer_size_seconds: int = 0.2,
        end_utterance_silence_threshold: int = 1000,
    ) -> None:
        super().__init__()

        self._opts = opts
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._api_key = api_key
        self._speaking = False
        self._session = http_session
        self._queue = asyncio.Queue[Union[rtc.AudioFrame, str]]()
        self._event_queue = asyncio.Queue[Optional[stt.SpeechEvent]]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry))

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: List[stt.SpeechEvent] = []

        self._token_expires_in = token_expires_in
        self._token = None

        self._END_UTTERANCE_SILENCE_THRESHOLD_MSG: str = json.dumps({"end_utterance_silence_threshold": end_utterance_silence_threshold })
        
        # Buffer to collect frames until we have 100ms worth of audio
        self._buffer_size_seconds = buffer_size_seconds
        self._buffer = bytearray()
        self._buffer_duration = 0.0  # duration of audio in the buffer in seconds

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

    async def _get_temporary_token(self):
        url = "https://api.assemblyai.com/v2/realtime/token"
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }
        data = json.dumps(
            {
                "expires_in": self._token_expires_in,
            },
        )
        res = await self._session.post(url, headers=headers, data=data)
        res_json = await res.json()
        self._token = res_json["token"]
        if self._token is None:
            raise ValueError("Failed to get temporary token")
        self._token_expires_at = time.time() + self._token_expires_in

    async def _run(self, max_retry: int) -> None:
        """
        Run a single websocket connection to Assembly AI and make sure to reconnect
        when something went wrong.
        """
        if self._token is None or time.time() > self._token_expires_at:
            await self._get_temporary_token()

        try:
            retry_count = 0
            while not self._closed:
                try:
                    live_config = {
                        "sample_rate": self._sample_rate,
                        "word_boost": self._opts.word_boost,
                        "encoding": self._opts.encoding,
                        "disable_partial_transcripts": self._opts.disable_partial_transcripts,
                        "enable_extra_session_information": self._opts.enable_extra_session_information,
                        "token": self._token,  # temporary authentication token
                    }

                    ws_url = "wss://api.assemblyai.com/v2/realtime/ws"
                    url = f"{ws_url}?{urlencode(live_config).lower()}"
                    ws = await self._session.ws_connect(url)
                    retry_count = 0  # connected successfully, reset the retry_count

                    await self._run_ws(ws)
                except Exception:
                    # Something went wrong, retry the connection
                    if retry_count >= max_retry:
                        print(
                            f"failed to connect to Assembly AI after {max_retry} tries"
                        )
                        break

                    retry_delay = min(retry_count * 2, 10)  # max 10s
                    retry_count += 1  # increment after calculating the delay, the first retry should happen directly

                    print(
                        f"Assembly AI connection failed, retrying in {retry_delay}s",
                    )
                    await asyncio.sleep(retry_delay)
        except Exception:
            print("Assembly AI task failed")
        finally:
            self._event_queue.put_nowait(None)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        This method can throw ws errors, these are handled inside the _run method
        """

        closing_ws = False
        self._queue.put_nowait(self._END_UTTERANCE_SILENCE_THRESHOLD_MSG)

        async def send_task():
            nonlocal closing_ws
            # forward inputs to Assembly AI
            # if we receive a close message, signal it to Assembly AI and break.
            # the recv task will then make sure to process the remaining audio and stop
            while True:
                data = await self._queue.get()
                self._queue.task_done()

                if isinstance(data, rtc.AudioFrame):
                    # TODO(theomonnom): The remix_and_resample method is low quality
                    # and should be replaced with a continuous resampling
                    frame = data.remix_and_resample(
                        self._sample_rate,
                        self._num_channels,
                    )
                    self._buffer.extend(frame.data.tobytes())
                    self._buffer_duration += len(frame.data) / (
                        self._sample_rate * self._num_channels
                    )

                    if self._buffer_duration >= self._buffer_size_seconds:
                        await ws.send_bytes(bytes(self._buffer))
                        self._buffer.clear()
                        self._buffer_duration = 0.0
                elif data == SpeechStream._CLOSE_MSG:
                    closing_ws = True
                    await ws.send_str(data)  # tell Assembly AI we are done with inputs
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
                        "Assembly AI connection closed unexpectedly",
                    )  # this will trigger a reconnection, see the _run loop

                if msg.type != aiohttp.WSMsgType.TEXT:
                    print("unexpected Assembly AI message type %s", msg.type)
                    continue

                try:
                    # received a message from Assembly AI
                    data = json.loads(msg.data)
                    self._process_stream_event(data)
                except Exception:
                    print("failed to process Assembly AI message")

        await asyncio.gather(send_task(), recv_task())

    def _end_speech(self) -> None:
        if not self._speaking:
            print(
                "trying to commit final events without being in the speaking state",
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
                    language=ENGLISH,
                    start_time=self._final_events[0].alternatives[0].start_time,
                    end_time=self._final_events[-1].alternatives[0].end_time,
                    confidence=confidence,
                    text=sentence,
                ),
            ],
        )
        self._event_queue.put_nowait(end_event)
        self._final_events = []

    def _process_stream_event(self, data: dict) -> None:
        # see this page:
        # https://www.assemblyai.com/docs/api-reference/streaming/realtime
        # for more information about the different types of events
        if data["message_type"] == "SessionBegins":
            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_queue.put_nowait(start_event)

        elif data["message_type"] == "PartialTranscript":
            self._speaking = True
            alts = live_transcription_to_speech_data(ENGLISH, data)
            if len(alts) > 0 and alts[0].text:
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alts,
                )
                self._event_queue.put_nowait(interim_event)

        elif data["message_type"] == "FinalTranscript":
            alts = live_transcription_to_speech_data(ENGLISH, data)
            if len(alts) > 0 and alts[0].text:
                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alts,
                )
                self._final_events.append(final_event)
                self._event_queue.put_nowait(final_event)
                self._end_speech()
        elif data["message_type"] == "RealtimeError":
            print("Received unexpected error from Assembly AI %s", data)

    async def __anext__(self) -> stt.SpeechEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def live_transcription_to_speech_data(
    language: str,
    data: dict,
) -> List[stt.SpeechData]:
    return [
        stt.SpeechData(
            language=language,
            start_time=data["words"][0]["start"] / 1000 if data["words"] else 0,
            end_time=data["words"][-1]["end"] / 1000 if data["words"] else 0,
            confidence=data["confidence"],
            text=data["text"],
        ),
    ]


def prerecorded_transcription_to_speech_event(
    language: str | None,
    data: dict,
) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=language,  # AssemblyAI doesn't provide language detection in this format
                start_time=data["words"][0]["start"] / 1000 if data["words"] else 0,
                end_time=data["words"][-1]["end"] / 1000 if data["words"] else 0,
                confidence=data["confidence"],
                text=data["text"],
            ),
        ],
    )
