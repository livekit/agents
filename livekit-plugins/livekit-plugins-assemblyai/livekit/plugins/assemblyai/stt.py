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
import contextlib
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer

from .log import logger

ENGLISH = "en"

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}


@dataclass
class STTOptions:
    sample_rate: Optional[int] = None
    word_boost: Optional[List[str]] = None
    encoding: Optional[Literal["pcm_s16le", "pcm_mulaw"]] = None
    disable_partial_transcripts: bool = False
    enable_extra_session_information: bool = False
    end_utterance_silence_threshold: Optional[int] = None
    # Buffer to collect frames until have 100ms worth of audio
    buffer_size_seconds: Optional[float] = None

    def __post_init__(self):
        if self.encoding not in (None, "pcm_s16le", "pcm_mulaw"):
            raise ValueError(f"Invalid encoding: {self.encoding}")


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: Optional[int] = 16000,
        word_boost: Optional[List[str]] = None,
        encoding: Optional[str] = "pcm_s16le",
        disable_partial_transcripts: bool = False,
        enable_extra_session_information: bool = False,
        end_utterance_silence_threshold: int = 1000,
        http_session: Optional[aiohttp.ClientSession] = None,
        buffer_size_seconds: float = 0.2,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "AssemblyAI API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `ASSEMBLYAI_API_KEY` environment variable"
            )
        self._api_key = api_key

        self._opts = STTOptions(
            sample_rate=sample_rate,
            word_boost=word_boost,
            encoding=encoding,
            disable_partial_transcripts=disable_partial_transcripts,
            enable_extra_session_information=enable_extra_session_information,
            buffer_size_seconds=buffer_size_seconds,
            end_utterance_silence_threshold=end_utterance_silence_threshold,
        )
        self._session = http_session

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        *,
        buffer: AudioBuffer,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

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
            stt_=self,
            opts=config,
            api_key=self._api_key,
            http_session=self.session,
        )


class SpeechStream(stt.SpeechStream):
    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})
    # Used to signal end of input
    _END_OF_INPUT_MSG: str = "END_OF_INPUT"

    def __init__(
        self,
        stt_: STT,
        opts: STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__(stt=stt_)

        self._opts = opts
        self._num_channels = num_channels
        self._api_key = api_key
        self._session = http_session
        self._queue = asyncio.Queue[Union[rtc.AudioFrame, str]]()
        self._event_queue = asyncio.Queue[Optional[stt.SpeechEvent]]()
        self._closed = False
        self._max_retry = max_retry

        if self._num_channels != 1:
            raise ValueError(
                f"AssemblyAI only supports mono audio, but a `num_channels` of {self._num_channels} was provided"
            )

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: List[stt.SpeechEvent] = []

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            logger.error(
                "trying to push frames to a closed stream",
            )

        self._queue.put_nowait(frame)

    def end_input(self) -> None:
        self.push_frame(SpeechStream._END_OF_INPUT_MSG)

    async def aclose(self) -> None:
        self.push_frame(SpeechStream._CLOSE_MSG)
        self._closed = True

        self._task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._task

        await self._session.close()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._run(self._max_retry)

    @utils.log_exceptions(logger=logger)
    async def _run(self, max_retry: int) -> None:
        """
        Run a single websocket connection to AssemblyAI and make sure to reconnect
        when something went wrong.
        """
        try:
            retry_count = 0
            while not self._closed:
                try:
                    live_config = {
                        "sample_rate": self._opts.sample_rate,
                        "word_boost": self._opts.word_boost,
                        "encoding": self._opts.encoding,
                        "disable_partial_transcripts": self._opts.disable_partial_transcripts,
                        "enable_extra_session_information": self._opts.enable_extra_session_information,
                    }

                    headers = {
                        "Authorization": self._api_key,
                        "Content-Type": "application/json",
                    }

                    ws_url = "wss://api.assemblyai.com/v2/realtime/ws"
                    filtered_config = {
                        k: v for k, v in live_config.items() if v is not None
                    }
                    url = f"{ws_url}?{urlencode(filtered_config).lower()}"
                    ws = await self._session.ws_connect(url, headers=headers)
                    retry_count = 0  # connected successfully, reset the retry_count

                    await self._run_ws(ws)
                except Exception:
                    # Something went wrong, retry the connection
                    if retry_count >= max_retry:
                        logger.error(
                            f"failed to connect to AssemblyAI after {max_retry} tries"
                        )
                        break

                    retry_delay = min(retry_count * 2, 10)  # max 10s
                    retry_count += 1  # increment after calculating the delay, the first retry should happen directly

                    logger.info(
                        f"AssemblyAI connection failed, retrying in {retry_delay}s",
                    )
                    await asyncio.sleep(retry_delay)
        finally:
            self._event_queue.put_nowait(None)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        This method can throw ws errors, these are handled inside the _run method
        """

        input_ended = False
        closing_ws = False

        END_UTTERANCE_SILENCE_THRESHOLD_MSG = json.dumps(
            {
                "end_utterance_silence_threshold": self._opts.end_utterance_silence_threshold
            }
        )
        if self._opts.end_utterance_silence_threshold is not None:
            self.push_frame(END_UTTERANCE_SILENCE_THRESHOLD_MSG)

        async def send_task():
            nonlocal input_ended, closing_ws

            # Local variables for buffering
            buffer = bytearray()
            buffer_duration = 0.0

            # forward inputs to AssemblyAI
            # if we receive a close message, signal it to AssemblyAI and break.
            # the recv task will then make sure to process the remaining audio and stop
            while True:
                data = await self._queue.get()
                self._queue.task_done()

                if isinstance(data, rtc.AudioFrame):
                    # TODO: The remix_and_resample method is low quality
                    # and should be replaced with a continuous resampling
                    frame = data.remix_and_resample(
                        self._opts.sample_rate,
                        self._num_channels,
                    )
                    buffer.extend(frame.data.tobytes())

                    # Calculate buffer duration
                    total_frames = len(buffer) / bytes_per_frame[self._opts.encoding]
                    samples_per_second = self._opts.sample_rate * self._num_channels
                    buffer_duration = total_frames / samples_per_second

                    if buffer_duration >= self._opts.buffer_size_seconds:
                        await ws.send_bytes(bytes(buffer))
                        buffer.clear()
                        buffer_duration = 0.0
                elif data == END_UTTERANCE_SILENCE_THRESHOLD_MSG:
                    await ws.send_str(data)
                elif data == SpeechStream._END_OF_INPUT_MSG:
                    input_ended = True
                elif data == SpeechStream._CLOSE_MSG:
                    closing_ws = True
                    await ws.send_str(data)  # tell AssemblyAI we are done with inputs
                    break
                else:
                    logger.error("Received unexpected data type: ", type(data))

        async def recv_task():
            nonlocal closing_ws
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=10)
                except asyncio.TimeoutError:
                    if input_ended:
                        # for now, end speech if haven't received anything on
                        # on websocket for 10 seconds and input has ended
                        self._end_speech()
                        await self.aclose()
                        return
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    raise Exception(
                        "AssemblyAI connection closed unexpectedly",
                    )  # this will trigger a reconnection, see the _run loop

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("unexpected AssemblyAI message type %s", msg.type)
                    continue

                try:
                    # received a message from AssemblyAI
                    data = json.loads(msg.data)
                    self._process_stream_event(data, closing_ws)
                except Exception:
                    logger.error("failed to process AssemblyAI message")

        await asyncio.gather(send_task(), recv_task())

    def _end_speech(self) -> None:
        if len(self._final_events) == 0:
            return

        # combine all final transcripts since the start of the speech
        sentence = " ".join(f.alternatives[0].text for f in self._final_events)
        confidence = sum(f.alternatives[0].confidence for f in self._final_events)
        confidence /= len(self._final_events)

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
        self._final_events = []
        self._event_queue.put_nowait(end_event)
        # break async iteration if speech has ended
        self._event_queue.put_nowait(None)

    def _process_stream_event(self, data: dict, closing_ws: bool) -> None:
        # see this page:
        # https://www.assemblyai.com/docs/api-reference/streaming/realtime
        # for more information about the different types of events
        if data["message_type"] == "SessionBegins":
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_queue.put_nowait(start_event)

        elif data["message_type"] == "PartialTranscript":
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

        elif data["message_type"] == "SessionTerminated":
            if closing_ws:
                pass
            else:
                raise Exception("AssemblyAI connection closed unexpectedly")

        elif data["message_type"] == "SessionInformation":
            logger.info("AssemblyAI Session Information: %s", str(data))

        elif data["message_type"] == "RealtimeError":
            logger.error("Received unexpected error from AssemblyAI %s", data)

        else:
            logger.warning(
                "Received unexpected error from AssemblyAI %s", data["message_type"]
            )

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
