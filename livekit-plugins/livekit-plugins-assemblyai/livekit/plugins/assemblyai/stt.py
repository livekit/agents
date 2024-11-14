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
from dataclasses import dataclass
from typing import List, Literal, Optional
from urllib.parse import urlencode

import aiohttp
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
    # Buffer to collect frames to send to AssemblyAI
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
        end_utterance_silence_threshold: Optional[int] = 500,
        http_session: Optional[aiohttp.ClientSession] = None,
        buffer_size_seconds: float = 0.05,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
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

    def __init__(
        self,
        stt_: STT,
        opts: STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        num_channels: int = 1,
        max_retry: int = 32,
    ) -> None:
        super().__init__(stt=stt_, sample_rate=opts.sample_rate)

        self._opts = opts
        self._num_channels = num_channels
        self._api_key = api_key
        self._session = http_session
        self._max_retry = max_retry
        self._speech_duration = 0

        if self._num_channels != 1:
            raise ValueError(
                f"AssemblyAI only supports mono audio, but a `num_channels` of {self._num_channels} was provided"
            )

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: List[stt.SpeechEvent] = []

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        await self._run(self._max_retry)

    @utils.log_exceptions(logger=logger)
    async def _run(self, max_retry: int) -> None:
        """
        Run a single websocket connection to AssemblyAI and make sure to reconnect
        when something went wrong.
        """
        retry_count = 0
        while self._input_ch.qsize() or not self._input_ch.closed:
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

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        This method can throw ws errors, these are handled inside the _run method
        """
        closing_ws = False

        async def send_task():
            nonlocal closing_ws

            if self._opts.end_utterance_silence_threshold:
                await ws.send_str(
                    json.dumps(
                        {
                            "end_utterance_silence_threshold": self._opts.end_utterance_silence_threshold
                        }
                    )
                )

            samples_per_buffer = self._opts.sample_rate // round(
                1 / self._opts.buffer_size_seconds
            )
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._num_channels,
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

        async def recv_task():
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
                    logger.exception("failed to process AssemblyAI message")

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def _process_stream_event(self, data: dict, closing_ws: bool) -> None:
        # see this page:
        # https://www.assemblyai.com/docs/api-reference/streaming/realtime
        # for more information about the different types of events
        if data["message_type"] == "SessionBegins":
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_ch.send_nowait(start_event)

        elif data["message_type"] == "PartialTranscript":
            alts = live_transcription_to_speech_data(ENGLISH, data)
            if len(alts) > 0 and alts[0].text:
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alts,
                )
                self._event_ch.send_nowait(interim_event)

        elif data["message_type"] == "FinalTranscript":
            alts = live_transcription_to_speech_data(ENGLISH, data)
            if len(alts) > 0 and alts[0].text:
                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alts,
                )
                self._final_events.append(final_event)
                self._event_ch.send_nowait(final_event)

            # log metrics
            if self._speech_duration > 0:
                usage_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    # request_id="",
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=self._speech_duration
                    ),
                )
                self._event_ch.send_nowait(usage_event)
                self._speech_duration = 0

        elif data["message_type"] == "SessionTerminated":
            if closing_ws:
                pass
            else:
                raise Exception("AssemblyAI connection closed unexpectedly")

        elif data["message_type"] == "SessionInformation":
            logger.debug("AssemblyAI Session Information: %s", str(data))

        elif data["message_type"] == "RealtimeError":
            logger.error("Received unexpected error from AssemblyAI %s", data)

        else:
            logger.warning(
                "Received unexpected error from AssemblyAI %s", data["message_type"]
            )


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
