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
from typing import Literal
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

ENGLISH = "en"
DEFAULT_ENCODING = "pcm_s16le"
DEFAULT_WORD_FINALIZATION_MAX_WAIT_TIME = 240
DEFAULT_END_OF_TURN_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MIN_END_OF_TURN_SILENCE_WHEN_CONFIDENT = 0
DEFAULT_MAX_TURN_SILENCE = 400
DEFAULT_TOKEN_MODE = False  # Added token_mode default


# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}

@dataclass
class STTOptions:
    sample_rate: int
    buffer_size_seconds: float
    word_boost: NotGivenOr[list[str]] = NOT_GIVEN
    encoding: NotGivenOr[Literal["pcm_s16le", "pcm_mulaw"]] = NOT_GIVEN
    word_finalization_max_wait_time: NotGivenOr[int] = NOT_GIVEN
    end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN
    min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN
    max_turn_silence: NotGivenOr[int] = NOT_GIVEN
    token_mode: NotGivenOr[bool] = NOT_GIVEN  # Added token_mode

    def __post_init__(self):
        if self.encoding not in (NOT_GIVEN, "pcm_s16le", "pcm_mulaw"):
            raise ValueError(f"Invalid encoding: {self.encoding}")


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        word_boost: NotGivenOr[list[str]] = NOT_GIVEN,
        encoding: NotGivenOr[Literal["pcm_s16le", "pcm_mulaw"]] = NOT_GIVEN,
        word_finalization_max_wait_time: NotGivenOr[int] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        token_mode: NotGivenOr[bool] = NOT_GIVEN,  # Added token_mode
        http_session: aiohttp.ClientSession | None = None,
        buffer_size_seconds: float = 0.05,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
        )
        self._api_key = api_key if is_given(api_key) else os.environ.get("ASSEMBLYAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "AssemblyAI API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `ASSEMBLYAI_API_KEY` environment variable"
            )

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            word_boost=word_boost,
            encoding=encoding,
            word_finalization_max_wait_time=word_finalization_max_wait_time,
            end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident=min_end_of_turn_silence_when_confident,
            max_turn_silence=max_turn_silence,
            token_mode=token_mode,  # Ensure token_mode is passed
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        word_boost: NotGivenOr[list[str]] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        word_finalization_max_wait_time: NotGivenOr[int] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
    ):
        if is_given(word_boost):
            self._opts.word_boost = word_boost
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(word_finalization_max_wait_time):
            self._opts.word_finalization_max_wait_time = word_finalization_max_wait_time
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_end_of_turn_silence_when_confident):
            self._opts.min_end_of_turn_silence_when_confident = min_end_of_turn_silence_when_confident
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence

        for stream in self._streams:
            stream.update_options(
                word_boost=word_boost,
                buffer_size_seconds=buffer_size_seconds,
                word_finalization_max_wait_time=word_finalization_max_wait_time,
                end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
                min_end_of_turn_silence_when_confident=min_end_of_turn_silence_when_confident,
                max_turn_silence=max_turn_silence,
            )


class SpeechStream(stt.SpeechStream):
    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speech_duration: float = 0
        self._utterance_mapping = {}

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: list[SpeechEvent] = []
        self._last_seen_words: dict[str, str] = {}  # For token_mode
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        word_boost: NotGivenOr[list[str]] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        word_finalization_max_wait_time: NotGivenOr[int] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
    ):
        if is_given(word_boost):
            self._opts.word_boost = word_boost
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(word_finalization_max_wait_time):
            self._opts.word_finalization_max_wait_time = word_finalization_max_wait_time
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_end_of_turn_silence_when_confident):
            self._opts.min_end_of_turn_silence_when_confident = min_end_of_turn_silence_when_confident
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence

        self._reconnect_event.set()

    async def _run(self) -> None:
        """
        Run a single websocket connection to AssemblyAI and make sure to reconnect
        when something went wrong.
        """

        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            samples_per_buffer = self._opts.sample_rate // round(1 / self._opts.buffer_size_seconds)
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            # forward inputs to AssemblyAI
            # if we receive a close message, signal it to AssemblyAI and break.
            # the recv task will then make sure to process the remaining audio and stop
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
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
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    raise APIStatusError(
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
                    logger.exception("failed to process AssemblyAI message")

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
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore
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
        self._last_seen_words.clear()  # Reset for new connection
        live_config = {
            "sample_rate": self._opts.sample_rate,
            "word_boost": json.dumps(self._opts.word_boost)
            if is_given(self._opts.word_boost)
            else None,
            "encoding": self._opts.encoding if is_given(self._opts.encoding) else DEFAULT_ENCODING,
            "word_finalization_max_wait_time": self._opts.word_finalization_max_wait_time
            if is_given(self._opts.word_finalization_max_wait_time)
            else DEFAULT_WORD_FINALIZATION_MAX_WAIT_TIME,
            "end_of_turn_confidence_threshold": self._opts.end_of_turn_confidence_threshold
            if is_given(self._opts.end_of_turn_confidence_threshold)
            else DEFAULT_END_OF_TURN_CONFIDENCE_THRESHOLD,
            "min_end_of_turn_silence_when_confident": self._opts.min_end_of_turn_silence_when_confident
            if is_given(self._opts.min_end_of_turn_silence_when_confident)
            else DEFAULT_MIN_END_OF_TURN_SILENCE_WHEN_CONFIDENT,
            "max_turn_silence": self._opts.max_turn_silence
            if is_given(self._opts.max_turn_silence)
            else DEFAULT_MAX_TURN_SILENCE,
            "token_mode": self._opts.token_mode if is_given(self._opts.token_mode) else DEFAULT_TOKEN_MODE,  # Added token_mode
        }

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": f"AssemblyAI/1.0 (integration=Livekit/1.0.22)"
        }

        ws_url = "wss://streaming.assemblyai.com/v3/ws"
        filtered_config = {k: v for k, v in live_config.items() if v is not None}
        url = f"{ws_url}?{urlencode(filtered_config).lower()}"
        ws = await self._session.ws_connect(url, headers=headers)
        return ws

    def _process_stream_event(self, data: dict, closing_ws: bool) -> None:
        # see this page:
        # https://www.assemblyai.com/docs/api-reference/streaming/realtime
        # for more information about the different types of events
        if "error" in data:
            logger.error("Received error from AssemblyAI: %s", data["error"])
            return

        message_type = data.get("type")

        if message_type == "Begin":
            logger.debug("AssemblyAI session started: %s", str(data))

        elif message_type == "Turn":
            logger.debug("AssemblyAI turn received: %s", str(data))
            alts = live_transcription_to_speech_data(ENGLISH, data)

            if self._opts.token_mode:
                # Get all words from the message
                all_words = data.get("words", [])
                end_of_turn = data.get("end_of_turn")
            
                if end_of_turn == True: 
                    self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)) 
                
                for i, word in enumerate(all_words):
                    word_id = f"{data.get('turn_order', 0)}_{i}"  # Create a unique ID for this word position
                    word_text = word.get("text", "")
                
                # Check if this is a new or updated word
                if word_id in self._last_seen_words:
                    # This position has been seen before - check if the word has changed/expanded
                    last_text = self._last_seen_words[word_id]
                    
                    if word_text != last_text and word_text.startswith(last_text):
                        # The word has expanded (e.g., "medi" -> "medicine")
                        # Emit just the new part as a final transcript
                        new_part = word_text[len(last_text):]
                        
                        # Only emit if there's actually new content
                        if new_part:
                            delta_speech_data = stt.SpeechData(
                                language="en-US",  # TODO: Revert to self._opts.language once STTOptions issue is resolved
                                text=new_part,
                                start_time=word.get("start", 0) / 1000.0,
                                end_time=word.get("end", 0) / 1000.0,
                                confidence=word.get("confidence", 0.0)
                            )
                            final_event = stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT, 
                                alternatives=[delta_speech_data]
                            )
                            self._event_ch.send_nowait(final_event)                    
                    # Note: We don't handle complete word changes/corrections since AssemblyAI
                    # only emits immutable transcripts where words only grow and don't get replaced
                
                else:
                    # This is a completely new word - emit it as a final transcript
                    delta_speech_data = stt.SpeechData(
                        language="en-US",  
                        text=word_text,
                        start_time=word.get("start", 0) / 1000.0,
                        end_time=word.get("end", 0) / 1000.0,
                        confidence=word.get("confidence", 0.0)
                    )
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT, 
                        alternatives=[delta_speech_data]
                    )
                    self._event_ch.send_nowait(final_event)
                
                # Update our tracking dictionary with this word's current text
                self._last_seen_words[word_id] = word_text
                
            else:
                end_of_turn = data.get("end_of_turn")
            
                if end_of_turn == True:  
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=alts,
                    )
                    self._final_events.append(final_event)
                    self._event_ch.send_nowait(final_event)

                    self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                    return
                
                if self._speech_duration > 0:
                    usage_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        alternatives=[],
                        recognition_usage=stt.RecognitionUsage(audio_duration=self._speech_duration),
                    )
                    self._event_ch.send_nowait(usage_event)
                    self._speech_duration = 0

        elif message_type == "Termination":
            if closing_ws:
                pass
            else:
                raise Exception("AssemblyAI connection closed unexpectedly")

        else:
            logger.warning(
                "Received unexpected message type from AssemblyAI: %s",
                message_type or "No message_type field",
            )

def live_transcription_to_speech_data(
    language: str,
    data: dict,
) -> list[stt.SpeechData]:
    return [
        stt.SpeechData(
            language=language,
            start_time=data["words"][0]["start"] / 1000 if data["words"] else 0,
            end_time=data["words"][-1]["end"] / 1000 if data["words"] else 0,
            confidence=sum(word["confidence"] for word in data["words"]) / len(data["words"]),
            text=data["transcript"],
        ),
    ]
