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
import json
import os
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from ._utils import PeriodicCollector, _to_deepgram_url
from .log import logger
from .models import V2Models


@dataclass
class STTOptions:
    model: V2Models | str
    sample_rate: int
    keyterm: str | Sequence[str]
    endpoint_url: str
    language: str = "en"
    eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN
    eot_threshold: NotGivenOr[float] = NOT_GIVEN
    eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN
    mip_opt_out: bool = False
    tags: NotGivenOr[list[str]] = NOT_GIVEN


class STTv2(stt.STT):
    def __init__(
        self,
        *,
        model: V2Models | str = "flux-general-en",
        sample_rate: int = 16000,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "wss://api.deepgram.com/v2/listen",
        mip_opt_out: bool = False,
        # deprecated
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Deepgram STT.

        Args:
            model: The Deepgram model to use for speech recognition. Defaults to "flux-general-en".
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            eager_eot_threshold: The threshold for eager end of turn to enable preemptive generation. Disabled by default. Set to 0.3-0.9 to enable preemptive generation.
            eot_threshold: The threshold for end of speech detection. Defaults to 0.7.
            eot_timeout_ms: The timeout for end of speech detection. Defaults to 3000.
            keyterm: str or list of str of key terms to improve recognition accuracy. Defaults to None.
            tags: List of tags to add to the requests for usage reporting. Defaults to NOT_GIVEN.
            api_key: Your Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for Deepgram API. Defaults to "https://api.deepgram.com/v1/listen".
            mip_opt_out: Whether to take part in the model improvement program

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the DEEPGRAM_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            )
        )

        deepgram_api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required")
        self._api_key = deepgram_api_key

        if is_given(keyterms):
            logger.warning(
                "`keyterms` is deprecated, use `keyterm` instead for consistency with Deepgram API."
            )
            keyterm = keyterms

        self._opts = STTOptions(
            model=model,
            sample_rate=sample_rate,
            keyterm=keyterm if is_given(keyterm) else [],
            mip_opt_out=mip_opt_out,
            tags=_validate_tags(tags) if is_given(tags) else [],
            eager_eot_threshold=eager_eot_threshold,
            eot_threshold=eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
            endpoint_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamv2]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "V2 API does not support non-streaming recognize. Use with a StreamAdapter"
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        stream = SpeechStreamv2(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._opts.endpoint_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[V2Models | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        # deprecated
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(eot_timeout_ms):
            self._opts.eot_timeout_ms = eot_timeout_ms
        if is_given(keyterms):
            logger.warning(
                "`keyterms` is deprecated, use `keyterm` instead for consistency with Deepgram API."
            )
            keyterm = keyterms
        if is_given(keyterm):
            self._opts.keyterm = keyterm
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(eager_eot_threshold):
            self._opts.eager_eot_threshold = eager_eot_threshold

        for stream in self._streams:
            stream.update_options(
                model=model,
                sample_rate=sample_rate,
                eot_threshold=eot_threshold,
                eot_timeout_ms=eot_timeout_ms,
                keyterm=keyterm,
                mip_opt_out=mip_opt_out,
                endpoint_url=endpoint_url,
                tags=tags,
                eager_eot_threshold=eager_eot_threshold,
            )


class SpeechStreamv2(stt.SpeechStream):
    # _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
    # _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})

    def __init__(
        self,
        *,
        stt: STTv2,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
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

    def update_options(
        self,
        *,
        model: NotGivenOr[V2Models | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[str | list[str]] = NOT_GIVEN,
        mip_opt_out: NotGivenOr[bool] = NOT_GIVEN,
        tags: NotGivenOr[list[str]] = NOT_GIVEN,
        endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        # deprecated
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(eot_timeout_ms):
            self._opts.eot_timeout_ms = eot_timeout_ms
        if is_given(keyterms):
            logger.warning(
                "`keyterms` is deprecated, use `keyterm` instead for consistency with Deepgram API."
            )
            keyterm = keyterms
        if is_given(keyterm):
            self._opts.keyterm = keyterm
        if is_given(mip_opt_out):
            self._opts.mip_opt_out = mip_opt_out
        if is_given(tags):
            self._opts.tags = _validate_tags(tags)
        if is_given(endpoint_url):
            self._opts.endpoint_url = endpoint_url
        if is_given(eager_eot_threshold):
            self._opts.eager_eot_threshold = eager_eot_threshold

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        # async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
        #     # if we want to keep the connection alive even if no audio is sent,
        #     # Deepgram expects a keepalive message.
        #     # https://developers.deepgram.com/reference/listen-live#stream-keepalive
        #     try:
        #         while True:
        #             await ws.send_str(SpeechStream._KEEPALIVE_MSG)
        #             await asyncio.sleep(5)
        #     except Exception:
        #         return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio to deepgram in chunks of 50ms
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                    if has_ended:
                        self._audio_duration_collector.flush()
                        has_ended = False

            # tell deepgram we are done sending audio/inputs
            closing_ws = True
            await ws.send_str(SpeechStreamv2._CLOSE_MSG)

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
                    raise APIStatusError(message="deepgram connection closed unexpectedly")

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
                    # asyncio.create_task(keepalive_task(ws)),
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
            "sample_rate": self._opts.sample_rate,
            "encoding": "linear16",
            "mip_opt_out": self._opts.mip_opt_out,
        }

        if self._opts.eager_eot_threshold:
            live_config["eager_eot_threshold"] = self._opts.eager_eot_threshold

        if self._opts.eot_threshold:
            live_config["eot_threshold"] = self._opts.eot_threshold

        if self._opts.eot_timeout_ms:
            live_config["eot_timeout_ms"] = self._opts.eot_timeout_ms

        if self._opts.keyterm:
            live_config["keyterm"] = self._opts.keyterm

        if self._opts.tags:
            live_config["tag"] = self._opts.tags

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    _to_deepgram_url(live_config, base_url=self._opts.endpoint_url, websocket=True),
                    headers={"Authorization": f"Token {self._api_key}"},
                    heartbeat=30.0,
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

    def _send_transcript_event(self, event_type: stt.SpeechEventType, data: dict) -> None:
        alts = _parse_transcription(self._opts.language, data, self.start_time_offset)
        if alts:
            event = stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=alts,
            )
            self._event_ch.send_nowait(event)

    def _process_stream_event(self, data: dict) -> None:
        assert self._opts.language is not None

        if request_id := data.get("request_id"):
            self._request_id = request_id

        if data["type"] == "TurnInfo":
            event_type = data["event"]

            if event_type == "StartOfTurn":
                if self._speaking:
                    return

                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)

                self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)

            elif event_type == "Update":
                if not self._speaking:
                    return

                self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)

            elif event_type == "EagerEndOfTurn":
                # technically, a pause in speech is detected. for lifecycle purposes,
                # we are assuming the user is still speaking and sending a preflight event to
                # start preemptive synthesis.
                if not self._speaking:
                    return

                self._send_transcript_event(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, data)

            elif event_type == "TurnResumed":
                # sending interim transcript will abort eager end of turn
                self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)

            elif event_type == "EndOfTurn":
                if not self._speaking:
                    return

                self._speaking = False

                self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, data)

                end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                self._event_ch.send_nowait(end_event)

        elif data["type"] == "Error":
            logger.warning("deepgram sent an error", extra={"data": data})
            desc = data.get("description") or "unknown error from deepgram"
            code = -1
            raise APIStatusError(message=desc, status_code=code)


def _parse_transcription(
    language: str, data: dict[str, Any], start_time_offset: float
) -> list[stt.SpeechData]:
    transcript = data.get("transcript")
    words = data.get("words")
    if not words:
        return []
    confidence = sum(word["confidence"] for word in words) / len(words) if words else 0

    sd = stt.SpeechData(
        language=language,
        start_time=data.get("audio_window_start", 0) + start_time_offset,
        end_time=data.get("audio_window_end", 0) + start_time_offset,
        confidence=confidence,
        text=transcript or "",
        words=[
            TimedString(
                text=word.get("word", ""),
                start_time=word.get("start", 0) + start_time_offset,
                end_time=word.get("end", 0) + start_time_offset,
                start_time_offset=start_time_offset,
            )
            for word in words
        ],
    )
    return [sd]


def _validate_tags(tags: list[str]) -> list[str]:
    for tag in tags:
        if len(tag) > 128:
            raise ValueError("tag must be no more than 128 characters")
    return tags
