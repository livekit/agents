"""
*   Telnyx TTS API documentation:
    <https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone>.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
from pydantic import BaseModel, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .common import NUM_CHANNELS, SAMPLE_RATE, TTS_ENDPOINT, SessionManager, get_api_key
from .log import logger


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    base_url: str


class _AudioMessage(BaseModel):
    audio: str | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "Telnyx.NaturalHD.astra",
        api_key: str | None = None,
        base_url: str = TTS_ENDPOINT,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            voice=voice,
            api_key=get_api_key(api_key),
            base_url=base_url,
        )
        self._session_manager = SessionManager(http_session)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.voice

    @property
    def provider(self) -> str:
        return "telnyx"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            ensure_session=self._session_manager.ensure_session,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._session_manager.close()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        ensure_session: Callable[[], aiohttp.ClientSession],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._ensure_session = ensure_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._segments_ch = utils.aio.Chan[str]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _collect_segments() -> None:
            segment_text = ""
            async for input_data in self.input:
                if isinstance(input_data, str):
                    segment_text += input_data
                else:
                    if segment_text:
                        self._segments_ch.send_nowait(segment_text)
                        segment_text = ""
            self._segments_ch.close()

        async def _run_segments() -> None:
            async for text in self._segments_ch:
                await self._run_ws(text, output_emitter)

        tasks = [
            asyncio.create_task(_collect_segments()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIConnectionError:
            raise
        except (APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        url = f"{self._opts.base_url}?voice={self._opts.voice}"
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}

        decoder = utils.media.AudioStreamDecoder(
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            format="audio/mp3",
        )

        read_deadline = self._core_attempts

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await ws.send_str(json.dumps({"text": " "}))
            self.mark_started()
            await ws.send_str(json.dumps({"text": text}))
            await ws.send_str(json.dumps({"text": ""}))
            read_deadline.expect_response(final=True)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = _AudioMessage.model_validate_json(msg.data)
                        audio_data = data.audio
                        if audio_data:
                            audio_bytes = base64.b64decode(audio_data)
                            if audio_bytes:
                                read_deadline.progress()
                                decoder.push(audio_bytes)
                    except ValidationError as exc:
                        raise APIConnectionError(
                            "Invalid Telnyx audio message", retryable=False
                        ) from exc

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Telnyx TTS WebSocket error: %s", ws.exception())
                    break

            decoder.end_input()

        async def decode_task() -> None:
            async for frame in decoder:
                output_emitter.push(frame.data.tobytes())

        try:
            ws = await utils.http_context.ws_connect(
                self._ensure_session(),
                url,
                headers=headers,
                conn_options=self._conn_options,
            )
            async with ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(decode_task()),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await decoder.aclose()
            output_emitter.end_segment()
