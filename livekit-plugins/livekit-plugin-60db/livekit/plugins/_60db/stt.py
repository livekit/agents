from __future__ import annotations

import asyncio
import audioop
import json
import os
from pathlib import Path

import websockets
from dotenv import load_dotenv

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

from .client import _get_default_api_key
from .log import logger

# Load .env.local from current directory
load_dotenv(Path(".env.local"))

_DEFAULT_WS_URL = "wss://api.60db.ai/ws/stt"


class STT(stt.STT):
    """60db.ai WebSocket-based STT provider for LiveKit Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        ws_url: str | None = None,
        languages: list[str] | None = None,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        continuous_mode: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        self._api_key = api_key or _get_default_api_key() or os.getenv("SIXTY_DB_API_KEY", "")
        self._ws_url = ws_url or os.getenv("SIXTY_DB_STT_URL", "") or _DEFAULT_WS_URL
        self._languages = languages or ["en"]
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._continuous_mode = continuous_mode

        if not self._api_key:
            raise ValueError(
                "60db API key is required. Set SIXTY_DB_API_KEY env var or pass api_key argument."
            )
        if not self._ws_url:
            raise ValueError(
                "60db STT WebSocket URL is required. Set SIXTY_DB_STT_URL env var or pass ws_url argument."
            )

        logger.info("60db STT: initialized with ws_url=%s", self._ws_url)

    @property
    def label(self) -> str:
        return "60db.STT"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Non-streaming recognition is not supported for WebSocket-based STT."""
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a streaming STT session."""
        return SpeechStream(
            stt=self,
            ws_url=self._ws_url,
            api_key=self._api_key,
            languages=[language] if language else self._languages,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            continuous_mode=self._continuous_mode,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    """WebSocket-based streaming STT implementation for 60db.ai."""

    def __init__(
        self,
        *,
        stt: STT,
        ws_url: str,
        api_key: str,
        languages: list[str],
        encoding: str,
        sample_rate: int,
        continuous_mode: bool,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)

        self._ws_url = ws_url
        self._api_key = api_key
        self._languages = languages
        self._encoding = encoding
        self._target_sample_rate = sample_rate
        self._continuous_mode = continuous_mode

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._session_started = False

        # Audio resampling state
        self._resample_state: tuple | None = None
        self._input_sample_rate: int | None = None
        self._input_channels: int | None = None

        logger.info("60db STT Stream: created - encoding=%s, sample_rate=%d", encoding, sample_rate)

    async def _run(self) -> None:
        """Main run loop: connect WebSocket, handshake, stream audio, receive transcriptions."""
        try:
            url = f"{self._ws_url}?apiKey={self._api_key}"
            logger.info("60db STT: connecting to %s", self._ws_url)

            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as ws:
                self._ws = ws
                logger.info("60db STT: WebSocket connected")

                # Step 2: Wait for connection_established
                msg = await asyncio.wait_for(ws.recv(), timeout=self._conn_options.timeout)
                data = json.loads(msg)
                if (
                    not data.get("connection_established")
                    and data.get("type") != "connection_established"
                ):
                    logger.error("60db STT: expected connection_established, got: %s", data)
                    raise APIConnectionError("60db STT: failed to establish connection")
                logger.info("60db STT: connection established")

                # Step 3: Send start message
                start_msg = {
                    "type": "start",
                    "languages": self._languages,
                    "config": {
                        "encoding": self._encoding,
                        "sample_rate": self._target_sample_rate,
                        "continuous_mode": self._continuous_mode,
                    },
                }
                await ws.send(json.dumps(start_msg))
                logger.info("60db STT: sent start message")

                # Step 4: Wait for connected
                msg = await asyncio.wait_for(ws.recv(), timeout=self._conn_options.timeout)
                data = json.loads(msg)
                if data.get("type") != "connected":
                    logger.error("60db STT: expected 'connected', got: %s", data)
                    raise APIConnectionError("60db STT: session start failed")
                self._session_started = True
                logger.info("60db STT: session connected")

                # Step 5: Start receive loop
                receive_task = asyncio.create_task(self._receive_loop())

                try:
                    # Step 6-8: Read audio from input channel, convert, send
                    async for frame in self._input_ch:
                        if isinstance(frame, self._FlushSentinel):
                            continue
                        if hasattr(frame, "sample_rate"):
                            self._input_sample_rate = frame.sample_rate
                        if hasattr(frame, "num_channels"):
                            self._input_channels = frame.num_channels

                        pcm = frame.data.tobytes() if hasattr(frame.data, "tobytes") else frame.data
                        audio_data = self._convert_audio(pcm)
                        await ws.send(audio_data)

                except Exception as e:
                    logger.error("60db STT: audio streaming error: %s", e)
                finally:
                    # Send stop on close
                    if self._session_started and self._ws:
                        try:
                            await ws.send(json.dumps({"type": "stop"}))
                            logger.info("60db STT: sent stop message")
                        except Exception:
                            pass

                    try:
                        await asyncio.wait_for(receive_task, timeout=self._conn_options.timeout)
                    except asyncio.TimeoutError:
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except websockets.exceptions.ConnectionClosed as e:
            logger.info("60db STT: WebSocket closed: %s", e)
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError(f"60db STT: connection error: {e}") from e
        finally:
            self._ws = None
            self._session_started = False

    async def _receive_loop(self) -> None:
        """Receive transcription messages from WebSocket."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("60db STT: failed to parse message")
                    continue

                msg_type = data.get("type")

                if msg_type == "transcription":
                    self._handle_transcription(data)
                elif msg_type == "session_stopped":
                    billing = data.get("billing_summary", {})
                    logger.info(
                        "60db STT: session stopped, cost=%s",
                        billing.get("total_cost", "unknown"),
                    )
                    self._session_started = False
                    break
                elif msg_type == "error":
                    logger.error("60db STT: server error: %s", data.get("error"))
                else:
                    logger.debug("60db STT: unknown message type '%s': %s", msg_type, data)

        except websockets.exceptions.ConnectionClosed:
            logger.info("60db STT: receive loop - connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("60db STT: receive loop error: %s", e)

    def _handle_transcription(self, data: dict) -> None:
        """Emit SpeechEvent for a transcription message."""
        text = data.get("text", "")
        is_final = data.get("is_final", False)

        if not text or not text.strip():
            return

        language = data.get("language", self._languages[0] if self._languages else "en")

        event_type = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=language,
                    )
                ],
            )
        )

        if is_final:
            logger.info("60db STT: final transcript: '%s'", text)

    def _convert_audio(self, pcm: bytes) -> bytes:
        """Convert PCM audio to the target format (default: mulaw 8kHz)."""
        # Convert to mono if needed
        if self._input_channels and self._input_channels > 1:
            pcm = audioop.tomono(pcm, 2, 1.0, 1.0)

        # Resample to target rate if needed
        if self._input_sample_rate and self._input_sample_rate != self._target_sample_rate:
            pcm, self._resample_state = audioop.ratecv(
                pcm,
                2,
                1,
                self._input_sample_rate,
                self._target_sample_rate,
                self._resample_state,
            )

        # Encode to mulaw if that's the configured encoding
        if self._encoding == "mulaw":
            pcm = audioop.lin2ulaw(pcm, 2)

        return pcm
