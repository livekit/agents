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

from pydantic import BaseModel, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .common import NUM_CHANNELS, SAMPLE_RATE, TTS_ENDPOINT, SessionManager, get_api_key


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
        http_session: http.HttpSession | None = None,
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

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
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
        ensure_session: Callable[[], http.HttpSession],
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
            segment: list[str] = []
            async for input_data in self.input:
                if isinstance(input_data, tts.SpeechChunk):
                    segment.append(input_data.text)
                elif segment:
                    self._segments_ch.send_nowait(" ".join(segment))
                    segment = []
            self._segments_ch.close()

        async def _run_segments() -> None:
            async for text in self._segments_ch:
                await self._run_ws(text, output_emitter)

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(_collect_segments()), tasks.spawn(_run_segments()))
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

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

        async def send_task(ws: http.WebSocket) -> None:
            await ws.send_str(json.dumps({"text": " "}))
            self.mark_started()
            await ws.send_str(json.dumps({"text": text}))
            await ws.send_str(json.dumps({"text": ""}))
            read_deadline.expect_response(final=True)

        async def recv_task(ws: http.WebSocket) -> None:
            async for msg in ws:
                if msg.type is http.WebSocketMessageType.TEXT:
                    try:
                        data = _AudioMessage.model_validate_json(msg.data)
                        audio_data = data.audio
                        if audio_data:
                            audio_bytes = base64.b64decode(audio_data)
                            if audio_bytes:
                                read_deadline.progress()
                                decoder.push(audio_bytes)
                    except ValidationError as exc:
                        raise APIError(
                            "Invalid Telnyx audio message", kind="protocol", detail=str(msg.data)
                        ) from exc

            decoder.end_input()

        async def decode_task() -> None:
            async for frame in decoder:
                output_emitter.push(frame.data.tobytes())

        try:
            ws = await self._ensure_session().ws_connect(
                url, headers=headers, conn_options=self._conn_options
            )
            try:
                async with utils.aio.TaskSet() as tasks:
                    await asyncio.gather(
                        tasks.spawn(send_task(ws)),
                        tasks.spawn(recv_task(ws)),
                        tasks.spawn(decode_task()),
                    )
            finally:
                await ws.close()

        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await decoder.aclose()
            output_emitter.end_segment()
