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
from dataclasses import dataclass

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

from .log import logger
from .rtzrapi import DEFAULT_SAMPLE_RATE, RTZRConnectionError, RTZROpenAPIClient, RTZRStatusError

_DEFAULT_CHUNK_MS = 100


@dataclass
class _STTOptions:
    model_name: str = "sommers_ko"  # sommers_ko: "ko", sommers_ja: "ja"
    language: str = "ko"  # ko, ja, en
    sample_rate: int = DEFAULT_SAMPLE_RATE
    encoding: str = "LINEAR16"  # or "OGG_OPUS" in future
    domain: str = "CALL"  # CALL, MEETING
    epd_time: float = 0.3
    noise_threshold: float = 0.60
    active_threshold: float = 0.80
    use_punctuation: bool = False


class STT(stt.STT):
    """RTZR Streaming STT over WebSocket.

    Uses RTZROpenAPIClient for authentication and WebSocket connection.
    Audio frames streamed to `/v1/transcribe:streaming` endpoint.
    Server performs endpoint detection (EPD), final messages carry `final=true`.
    Stream is finalized by sending the string `EOS`.
    """

    def __init__(
        self,
        *,
        model: str = "sommers_ko",
        language: str = "ko",
        sample_rate: int = 8000,
        domain: str = "CALL",
        epd_time: float = 0.3,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
        use_punctuation: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        self._params = _STTOptions(
            model_name=model,
            language=language,
            sample_rate=sample_rate,
            domain=domain,
            epd_time=epd_time,
            noise_threshold=noise_threshold,
            active_threshold=active_threshold,
            use_punctuation=use_punctuation,
        )
        self._client = RTZROpenAPIClient(http_session=http_session)

    async def aclose(self) -> None:
        """Close the RTZR client and cleanup resources."""
        await self._client.close()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Single-shot recognition is not supported; use stream().")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        config = self._stt._client.build_config(
            model_name=self._stt._params.model_name,
            domain=self._stt._params.domain,
            sample_rate=self._stt._params.sample_rate,
            encoding=self._stt._params.encoding,
            epd_time=self._stt._params.epd_time,
            noise_threshold=self._stt._params.noise_threshold,
            active_threshold=self._stt._params.active_threshold,
            use_punctuation=self._stt._params.use_punctuation,
        )

        try:
            ws = await asyncio.wait_for(
                self._stt._client.connect_websocket(config), timeout=self._conn_options.timeout
            )
            return ws
        except asyncio.TimeoutError as e:
            raise APITimeoutError("WebSocket connection timeout") from e
        except RTZRStatusError as e:
            logger.error("RTZR API status error: %s", e)
            raise APIStatusError(
                message=e.message,
                status_code=e.status_code or 500,
                request_id=None,
                body=None,
            ) from e
        except RTZRConnectionError as e:
            logger.error("RTZR API connection error: %s", e)
            raise APIConnectionError("RTZR API connection failed") from e

    async def _run(self) -> None:
        while True:
            try:
                self._ws = await self._connect_ws()
                send_task = asyncio.create_task(self._send_audio_task())
                recv_task = asyncio.create_task(self._recv_task())
                try:
                    await asyncio.gather(send_task, recv_task)
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)
            except asyncio.TimeoutError as e:
                logger.error("RTZR STT connection timeout: %s", e)
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                logger.error("RTZR STT status error: %s %s", e.status, e.message)
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e
            except aiohttp.ClientError as e:
                logger.error("RTZR STT client error: %s", e)
                raise APIConnectionError() from e
            except Exception as e:
                logger.exception("RTZR STT unexpected error: %s", e)
                raise
            finally:
                if self._ws:
                    await self._ws.close()
                    self._ws = None
            break

    @utils.log_exceptions(logger=logger)
    async def _send_audio_task(self) -> None:
        assert self._ws
        # Bundle audio into appropriate chunks using AudioByteStream
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._stt._params.sample_rate,
            num_channels=1,
            samples_per_channel=self._stt._params.sample_rate // (1000 // _DEFAULT_CHUNK_MS),
        )
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                # Write audio frame data to the byte stream
                frames = audio_bstream.write(data.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                # Flush any remaining audio data
                frames = audio_bstream.flush()
            else:
                frames = []

            for frame in frames:
                await self._ws.send_bytes(frame.data.tobytes())

        await self._ws.send_str("EOS")
        logger.info("Sent EOS to close audio stream")

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self) -> None:
        assert self._ws
        in_speech = False
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON text from RTZR STT: %s", msg.data)
                    continue

                # Expected schema from reference: {"alternatives":[{"text": "..."}], "final": bool}
                if "alternatives" in data and data["alternatives"]:
                    text = data["alternatives"][0].get("text", "")
                    is_final = bool(data.get("final", False))
                    if text:
                        # Send START_OF_SPEECH if this is the first transcript in a sequence
                        if not in_speech:
                            in_speech = True
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                            )

                        # Send transcript event
                        event_type = (
                            stt.SpeechEventType.FINAL_TRANSCRIPT
                            if is_final
                            else stt.SpeechEventType.INTERIM_TRANSCRIPT
                        )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=event_type,
                                alternatives=[
                                    stt.SpeechData(text=text, language=self._stt._params.language)
                                ],
                            )
                        )

                        if is_final:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                            )
                            in_speech = False

                # Handle error messages
                if "error" in data:
                    error_msg = data["error"]
                    raise APIStatusError(
                        message=f"Server error: {error_msg}",
                        status_code=500,
                        request_id=None,
                        body=None,
                    )
                elif data.get("type") == "error" and "message" in data:
                    error_msg = data["message"]
                    raise APIStatusError(
                        message=f"Server error: {error_msg}",
                        status_code=500,
                        request_id=None,
                        body=None,
                    )

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket error: %s", self._ws.exception())
                raise APIConnectionError("WebSocket error occurred")
            else:
                logger.debug("Ignored WebSocket message type: %s", msg.type)
