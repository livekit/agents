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
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTLanguages, STTModels
from .tts import API_AUTH_HEADER

# each model is served from its own region
_BASE_URLS = {
    "finch_indic_1.0": "https://in.api.murf.ai",
    "finch_global_1.0": "https://us-east.api.murf.ai",
}
_SAMPLE_RATES = (8000, 12000, 16000, 24000, 32000, 44100, 48000)
# invalid key, invalid config, no credits, credits ran out, key revoked
_FATAL_CLOSE_CODES = (4001, 4002, 4003, 4005, 4006)


@dataclass
class _STTOptions:
    api_key: str
    model: STTModels | str
    language: STTLanguages | str
    sample_rate: int
    keywords: list[str] | None
    text_formatting: bool
    bvc: bool
    base_url: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: STTModels | str = "finch_global_1.0",
        language: STTLanguages | str = "en",
        sample_rate: int = 16000,
        keywords: list[str] | None = None,
        text_formatting: bool = False,
        bvc: bool = False,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Murf AI STT.

        See https://murf.ai/api/docs/speech-to-text/streaming for more details on the the Murf AI API.

        Args:
            api_key (str | None, optional): The Murf AI API key. If not provided, it will be read from the MURF_API_KEY environment variable.
            model (STTModels | str, optional): The Murf AI STT model to use: "finch_global_1.0" for English and European languages, "finch_indic_1.0" for Indian languages. Defaults to "finch_global_1.0".
            language (STTLanguages | str, optional): The language to transcribe (e.g., "en", "hi"), "auto" to detect it, or a comma-separated list to detect from. Defaults to "en".
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 16000.
            keywords (list[str] | None, optional): Words or phrases to boost recognition of.
            text_formatting (bool, optional): Format numbers and similar terms in Hindi and English (India) transcripts. Defaults to False.
            bvc (bool, optional): Remove background voices before transcription. Defaults to False.
            base_url (str, optional): The base URL for the Murf AI API. Defaults to the region that serves the model.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """  # noqa: E501
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
                # a flush sends Finalize, which ends the turn at once
                manual_flush=True,
            )
        )

        murf_api_key = api_key or os.environ.get("MURF_API_KEY")
        if not murf_api_key:
            raise ValueError("MURF_API_KEY must be set")

        if sample_rate not in _SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {_SAMPLE_RATES}")

        if not is_given(base_url):
            if model not in _BASE_URLS:
                raise ValueError(f"base_url must be set for model {model}")
            base_url = _BASE_URLS[model]

        self._opts = _STTOptions(
            api_key=murf_api_key,
            model=model,
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            text_formatting=text_formatting,
            bvc=bvc,
            base_url=base_url,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Murf"

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
        raise NotImplementedError("Murf AI STT does not support non-streaming recognize")

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        return SpeechStream(stt=self, opts=opts, conn_options=conn_options)


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = stt._ensure_session()
        self._speaking = False
        self._request_id = ""
        # transcripts don't carry the detected language
        detect = opts.language == "auto" or "," in opts.language
        self._language = LanguageCode("" if detect else opts.language)

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Murf closes the connection when it receives neither audio nor KeepAlive
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio to Murf in chunks of 50ms
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 20,
            )

            try:
                async for data in self._input_ch:
                    if isinstance(data, rtc.AudioFrame):
                        frames = audio_bstream.write(data.data.tobytes())
                    else:
                        frames = audio_bstream.flush()

                    for frame in frames:
                        await ws.send_bytes(frame.data.tobytes())

                    if isinstance(data, self._FlushSentinel):
                        await ws.send_str(SpeechStream._FINALIZE_MSG)

                # Murf sends the final transcript of the open turn, then closes
                closing_ws = True
                await ws.send_str(SpeechStream._CLOSE_MSG)
            except (aiohttp.ClientError, ConnectionError):
                # the server closed the connection; recv_task reports why
                return

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # Murf closes with 1000 after CloseStream, with an error code otherwise
                    if (closing_ws and ws.close_code == 1000) or self._session.closed:
                        return

                    raise APIStatusError(
                        message="Murf AI STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=self._request_id,
                        body=f"{msg.data=} {msg.extra=}",
                        retryable=ws.close_code not in _FATAL_CLOSE_CODES,
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Murf AI STT message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process Murf AI STT message")

        ws = await self._connect_ws()
        self._input_ch.restart_provider_clock()
        tasks = [
            asyncio.create_task(send_task(ws)),
            asyncio.create_task(recv_task(ws)),
        ]
        keepalive = asyncio.create_task(keepalive_task(ws))
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks, keepalive)
            await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params: dict[str, Any] = {"model": self._opts.model, "language": self._opts.language}
        if self._opts.keywords:
            params["keywords"] = ",".join(self._opts.keywords)
        if self._opts.text_formatting:
            params["text_formatting"] = "true"
        if self._opts.bvc:
            params["bvc"] = "true"

        url = self._opts.base_url.replace("http", "ws", 1)
        url = f"{url}/v1/transcription/stream-input?{urlencode(params)}"

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    url, headers={API_AUTH_HEADER: self._opts.api_key}, autoping=False
                ),
                self._conn_options.timeout,
            )
            await ws.send_str(
                json.dumps(
                    {
                        "type": "AudioConfig",
                        "codec": "pcm_s16le",
                        "sample_rate": self._opts.sample_rate,
                        "channels": 1,
                    }
                )
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to Murf AI STT") from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to Murf AI STT ({type(e).__name__})"
            ) from None

        self._report_connection_acquired(time.perf_counter() - t0, False)
        return ws

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")
        if msg_type == "Error":
            logger.error("Murf AI STT error: %s", data.get("message"))
            return

        if msg_type not in ("partial", "final"):
            return

        raw_event = data.get("rawEvent", {})
        self._request_id = raw_event.get("request_id", self._request_id)
        self._mark_provider_progress()

        text = data.get("transcript", "")
        # Murf finals carry no word timestamps today
        speech_data = stt.SpeechData(language=self._language, text=text) if text else None
        if speech_data and not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                )
            )

        if msg_type == "partial":
            if speech_data:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    )
                )
            return

        # a final follows each Finalize, with an empty transcript if no speech was pending
        if speech_data:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[speech_data],
                )
            )

        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                    alternatives=[speech_data] if speech_data else [],
                )
            )
