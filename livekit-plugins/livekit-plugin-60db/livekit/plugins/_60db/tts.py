from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import websockets
from dotenv import load_dotenv
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
    utils,
)

from .client import _get_default_api_key
from .log import logger

# Load .env.local from current directory
load_dotenv(Path(".env.local"))

_DEFAULT_WS_URL = "wss://api.60db.ai/ws/tts"

NUM_CHANNELS = 1


class TTS(tts.TTS):
    """60db.ai WebSocket-based TTS provider for LiveKit Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        ws_url: str | None = None,
        voice_id: str = "fbb75ed2-975a-40c7-9e06-38e30524a9a1",
        encoding: str = "LINEAR16",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key or _get_default_api_key() or os.getenv("SIXTY_DB_API_KEY", "")
        self._ws_url = ws_url or os.getenv("SIXTY_DB_TTS_URL", "") or _DEFAULT_WS_URL
        self._voice_id = voice_id
        self._encoding = encoding
        self._sample_rate = sample_rate

        if not self._api_key:
            raise ValueError(
                "60db API key is required. Set SIXTY_DB_API_KEY env var or pass api_key argument."
            )
        if not self._ws_url:
            raise ValueError(
                "60db TTS WebSocket URL is required. Set SIXTY_DB_TTS_URL env var or pass ws_url argument."
            )

        logger.info("60db TTS: initialized with ws_url=%s", self._ws_url)

    @property
    def label(self) -> str:
        return "60db.TTS"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Single-text TTS synthesis using the 60db WebSocket protocol."""

    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()

        try:
            url = f"{self._tts._ws_url}?apiKey={self._tts._api_key}"
            logger.info("60db TTS ChunkedStream: connecting to %s", self._tts._ws_url)

            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as ws:
                # Wait for connection_established
                msg = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(msg)
                if not data.get("connection_established"):
                    raise APIConnectionError(
                        "60db TTS: expected connection_established"
                    )
                logger.info("60db TTS ChunkedStream: connection established")

                # Create context
                create_msg = {
                    "create_context": {
                        "context_id": context_id,
                        "voice_id": self._tts._voice_id,
                        "audio_config": {
                            "audio_encoding": self._tts._encoding,
                            "sample_rate_hertz": self._tts._sample_rate,
                        },
                    }
                }
                await ws.send(json.dumps(create_msg))

                msg = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(msg)
                if not data.get("context_created"):
                    raise APIConnectionError(
                        "60db TTS: expected context_created"
                    )
                logger.info("60db TTS ChunkedStream: context created")

                # Initialize output emitter
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts._sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                # Send text
                await ws.send(
                    json.dumps(
                        {
                            "send_text": {
                                "context_id": context_id,
                                "text": self._input_text,
                            }
                        }
                    )
                )

                # Flush to trigger audio generation
                await ws.send(
                    json.dumps({"flush_context": {"context_id": context_id}})
                )

                # Receive audio chunks
                while True:
                    msg = await asyncio.wait_for(
                        ws.recv(), timeout=self._conn_options.timeout
                    )
                    data = json.loads(msg)

                    if "audio_chunk" in data:
                        audio_b64 = data["audio_chunk"].get("audioContent", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            output_emitter.push(audio_bytes)

                    if data.get("flush_completed"):
                        output_emitter.flush()
                        break

                # Close context
                await ws.send(
                    json.dumps({"close_context": {"context_id": context_id}})
                )
                msg = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(msg)
                if data.get("context_closed"):
                    logger.info("60db TTS ChunkedStream: context closed")

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIConnectionError:
            raise
        except websockets.exceptions.ConnectionClosed as e:
            logger.info("60db TTS ChunkedStream: WebSocket closed: %s", e)
        except Exception as e:
            raise APIConnectionError(
                f"60db TTS ChunkedStream: connection error: {e}"
            ) from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS synthesis using the 60db WebSocket protocol."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        request_id = utils.shortuuid()

        try:
            url = f"{self._tts._ws_url}?apiKey={self._tts._api_key}"
            logger.info("60db TTS SynthesizeStream: connecting to %s", self._tts._ws_url)

            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as ws:
                # Wait for connection_established
                msg = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(msg)
                if not data.get("connection_established"):
                    raise APIConnectionError(
                        "60db TTS: expected connection_established"
                    )

                # Create context
                await ws.send(
                    json.dumps(
                        {
                            "create_context": {
                                "context_id": context_id,
                                "voice_id": self._tts._voice_id,
                                "audio_config": {
                                    "audio_encoding": self._tts._encoding,
                                    "sample_rate_hertz": self._tts._sample_rate,
                                },
                            }
                        }
                    )
                )

                msg = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(msg)
                if not data.get("context_created"):
                    raise APIConnectionError(
                        "60db TTS: expected context_created"
                    )
                logger.info("60db TTS SynthesizeStream: context created")

                # Initialize output emitter in streaming mode
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts._sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                    stream=True,
                )

                segment_id = utils.shortuuid()
                output_emitter.start_segment(segment_id=segment_id)

                # Send and receive tasks running in parallel
                async def send_task() -> None:
                    async for input in self._input_ch:
                        if isinstance(input, str):
                            await ws.send(
                                json.dumps(
                                    {
                                        "send_text": {
                                            "context_id": context_id,
                                            "text": input,
                                        }
                                    }
                                )
                            )
                        elif isinstance(input, self._FlushSentinel):
                            await ws.send(
                                json.dumps(
                                    {
                                        "flush_context": {
                                            "context_id": context_id
                                        }
                                    }
                                )
                            )

                    # Input ended — close the context
                    await ws.send(
                        json.dumps({"close_context": {"context_id": context_id}})
                    )

                async def recv_task() -> None:
                    while True:
                        msg = await asyncio.wait_for(
                            ws.recv(), timeout=self._conn_options.timeout
                        )
                        data = json.loads(msg)

                        if "audio_chunk" in data:
                            audio_b64 = data["audio_chunk"].get(
                                "audioContent", ""
                            )
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                output_emitter.push(audio_bytes)

                        if data.get("flush_completed"):
                            output_emitter.end_segment()
                            # Start a new segment for the next batch
                            segment_id = utils.shortuuid()
                            output_emitter.start_segment(
                                segment_id=segment_id
                            )

                        if data.get("context_closed"):
                            logger.info(
                                "60db TTS SynthesizeStream: context closed"
                            )
                            break

                tasks = [
                    asyncio.create_task(send_task()),
                    asyncio.create_task(recv_task()),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)

                # End the final segment
                output_emitter.end_segment()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIConnectionError:
            raise
        except websockets.exceptions.ConnectionClosed as e:
            logger.info("60db TTS SynthesizeStream: WebSocket closed: %s", e)
        except Exception as e:
            raise APIConnectionError(
                f"60db TTS SynthesizeStream: connection error: {e}"
            ) from e
