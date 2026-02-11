"""FonadaLabs TTS plugin for LiveKit - WebSocket implementation"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import os
from typing import Literal

import aiohttp
from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    APIStatusError,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
    utils,
)

from .log import logger

FONADALABS_TTS_BASE_URL = "https://api.fonada.ai"
FONADALABS_TTS_WS_URL = "wss://api.fonada.ai/tts/generate-audio-ws"

# Available voices - Only Vaanee is supported
FonadaLabsVoices = Literal["Vaanee"]

# Supported languages
FonadaLabsLanguages = Literal["Hindi", "English", "Tamil", "Telugu"]


class ConnectionState(enum.Enum):
    """WebSocket connection states for TTS."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


class TTS(tts.TTS):
    """FonadaLabs Text-to-Speech implementation for LiveKit.

    This class provides text-to-speech functionality using the FonadaLabs API.
    FonadaLabs specializes in high-quality TTS for Indian languages.

    Args:
        api_key: FonadaLabs API key (required)
        voice: Voice to use for synthesis (default: "Vaanee", only Vaanee is supported)
        language: Language name (default: "Hindi"). Supported: Hindi, English, Tamil, Telugu
        api_url: API base URL (optional, defaults to production)
        sample_rate: Audio sample rate in Hz (default: 24000)
        num_channels: Number of audio channels (default: 1, mono)
        http_session: Optional aiohttp.ClientSession instance to use (for advanced use)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: FonadaLabsVoices | str = "Vaanee",
        language: FonadaLabsLanguages | str = "Hindi",
        api_url: str | None = None,
        sample_rate: int = 24000,
        num_channels: int = 1,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("FONADALABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FonadaLabs API key is required. Provide it directly or set FONADALABS_API_KEY env var."
            )

        # Force voice to be Vaanee (only supported voice)
        if voice and voice.strip() and voice.strip() != "Vaanee":
            logger.warning(f"Voice '{voice}' is not supported. Using 'Vaanee' instead.")
        self._voice = "Vaanee"

        # Validate language
        supported_languages = ["Hindi", "English", "Tamil", "Telugu"]
        if not language or not language.strip():
            raise ValueError(f"Language is required. Supported languages: {', '.join(supported_languages)}")

        language_normalized = language.strip()
        if language_normalized not in supported_languages:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported languages: {', '.join(supported_languages)}"
            )

        self._language = language_normalized
        self._base_url = (api_url or FONADALABS_TTS_BASE_URL).rstrip("/")

        # Build WebSocket URL
        if api_url:
            ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
            if not ws_url.endswith("/tts/generate-audio-ws"):
                ws_url = ws_url.rstrip("/") + "/tts/generate-audio-ws"
        else:
            ws_url = FONADALABS_TTS_WS_URL

        self._ws_url = ws_url
        self._http_session = http_session
        self._own_session = http_session is None
        if self._own_session:
            self._http_session = aiohttp.ClientSession()

    async def synthesize(self, text: str) -> SynthesizeStream:
        """Synthesize speech from text."""
        return SynthesizeStream(
            tts=self,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            text=text,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> SynthesizeStream:
        """Create a streaming TTS session - required by LiveKit agents framework.

        This method is called by LiveKit agents framework to get a streaming TTS instance.
        The text will be provided to the stream via the input channel.

        Args:
            conn_options: Connection options for the stream

        Returns:
            SynthesizeStream instance that can accept text via its input channel
        """
        # The SynthesizeStream expects the text argument as a keyword, not as a positional arg.
        # This is correct because LiveKit agents framework expects a streaming TTS object
        # which receives text segments asynchronously via an input channel, not at instantiation.
        # By passing 'text=None' as a keyword argument, the SynthesizeStream knows to expect text on its input channel.
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
            text=None,  # Text will be provided via input channel
        )
    async def aclose(self) -> None:
        """Close the TTS instance and cleanup resources."""
        if self._own_session and self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """SynthesizeStream implementation for FonadaLabs TTS."""

    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        text: str | None = None,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._text = text  # Can be None if text will be provided via input channel
        self._session_id = f"fonadalabs_{id(self)}"
        self._connection_state = ConnectionState.DISCONNECTED
        self._ws_conn = None
        self._send_task = None
        self._recv_task = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Main synthesis loop."""
        import uuid

        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Send TTS request."""
            try:
                # Get text - must be provided either via __init__ or input channel
                text = self._text
                if text is None:
                    # Read text from input channel (for stream() pattern used by LiveKit agents)
                    # LiveKit agents framework provides text segments via input_ch
                    logger.info("Text is None, reading from input channel...", extra=self._build_log_context())
                    text_segments = []
                    try:
                        # Check if _input_ch exists (it's a private attribute)
                        if not hasattr(self, '_input_ch'):
                            logger.error("_input_ch not found on SynthesizeStream", extra=self._build_log_context())
                            raise ValueError("_input_ch not available on SynthesizeStream")

                        logger.debug(f"_input_ch type: {type(self._input_ch)}", extra=self._build_log_context())

                        # Read all segments from input channel until it's closed
                        segment_count = 0
                        async for segment in self._input_ch:
                            segment_count += 1
                            logger.debug(f"Received segment {segment_count}: type={type(segment)}, value={str(segment)[:50]}...", extra=self._build_log_context())

                            # Handle different segment types
                            if isinstance(segment, str):
                                text_segments.append(segment)
                            elif hasattr(segment, 'text'):
                                text_segments.append(segment.text)
                            elif hasattr(segment, 'content'):
                                content = segment.content
                                if isinstance(content, str):
                                    text_segments.append(content)
                                elif isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, str):
                                            text_segments.append(item)
                            else:
                                # Try to convert to string
                                text_segments.append(str(segment))

                        logger.info(f"Finished reading from input channel: {segment_count} segments, {len(text_segments)} text segments", extra=self._build_log_context())
                    except asyncio.CancelledError:
                        logger.warning("Reading from input channel was cancelled", extra=self._build_log_context())
                        raise
                    except Exception as e:
                        logger.error(f"Error reading from input channel: {e}", exc_info=True, extra=self._build_log_context())
                        raise ValueError(f"Failed to read text from input channel: {e}") from e

                    if not text_segments:
                        logger.error("No text segments received from input channel", extra=self._build_log_context())
                        raise ValueError(
                            "No text provided. Use synthesize(text) or provide text via input channel."
                        )

                    # Concatenate text segments (join with space)
                    text = " ".join(text_segments)
                    logger.info(f"Read text from input channel ({len(text_segments)} segments, {len(text)} chars): {text[:100]}...", extra=self._build_log_context())

                request_data = {
                    "api_key": self._tts._api_key,
                    "text": text,
                    "voice_id": self._tts._voice,
                    "language": self._tts._language,
                }
                await ws.send_str(json.dumps(request_data))
                logger.debug(
                    f"TTS request sent: {json.dumps({**request_data, 'api_key': '***'})}",
                    extra=self._build_log_context()
                )
            except Exception as e:
                logger.error(f"Error sending request: {e}", extra=self._build_log_context())
                raise

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive and process WebSocket messages."""
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Log raw message for debugging
                        logger.debug(f"Raw WebSocket message: {msg.data}", extra=self._build_log_context())
                        should_continue = await self._handle_websocket_message(msg.data, output_emitter)
                        if not should_continue:
                            break
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # Binary audio data - push directly (FonadaLabs sends binary chunks)
                        output_emitter.push(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise APIConnectionError(f"WebSocket error: {ws.exception()}")
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        logger.debug("WebSocket closed by server", extra=self._build_log_context())
                        output_emitter.end_input()
                        break
            except asyncio.CancelledError:
                logger.debug("Receive task cancelled", extra=self._build_log_context())
                raise
            except Exception as e:
                logger.error(f"Error receiving messages: {e}", extra=self._build_log_context())
                raise

        try:
            self._connection_state = ConnectionState.CONNECTING

            # Create WebSocket connection using aiohttp
            timeout = aiohttp.ClientTimeout(total=self._conn_options.timeout)
            ws = await self._tts._http_session.ws_connect(
                self._tts._ws_url,
                timeout=timeout,
            )
            self._ws_conn = ws
            self._connection_state = ConnectionState.CONNECTED

            logger.info("WebSocket connected successfully", extra=self._build_log_context())

            try:

                self._send_task = asyncio.create_task(send_task(ws))
                self._recv_task = asyncio.create_task(recv_task(ws))

                tasks = [self._send_task, self._recv_task]

                try:
                    await asyncio.gather(*tasks)
                    logger.info(
                        "WebSocket session completed successfully", extra=self._build_log_context()
                    )
                except Exception as e:
                    logger.error(
                        f"WebSocket session failed: {e}",
                        extra=self._build_log_context(),
                        exc_info=True,
                    )
                    raise
                finally:
                    # Gracefully cancel tasks
                    await utils.aio.gracefully_cancel(*tasks)
                    self._send_task = None
                    self._recv_task = None
            finally:
                # Close WebSocket connection
                if ws and not ws.closed:
                    await ws.close()

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            self._connection_state = ConnectionState.FAILED
            logger.error(f"Connection failed: {e}", extra=self._build_log_context())
            raise APIConnectionError(f"Failed to connect to TTS WebSocket: {e}") from e
        except Exception as e:
            self._connection_state = ConnectionState.FAILED
            logger.error(
                f"Unexpected error in WebSocket session: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(f"TTS WebSocket session failed: {e}") from e
        finally:
            self._connection_state = ConnectionState.DISCONNECTED
            self._ws_conn = None

    async def _handle_websocket_message(
        self, msg_data: str, output_emitter: tts.AudioEmitter
    ) -> bool:
        """Handle WebSocket message with proper error handling.
        Returns:
            True if processing should continue, False if stream should end
        """
        try:
            resp = json.loads(msg_data)
            msg_type = resp.get("type")
            status = resp.get("status")  # FonadaLabs uses "status" field

            logger.debug(
                f"Processing message: type={msg_type}, status={status}, response={resp}",
                extra=self._build_log_context()
            )

            # Check for completion first (FonadaLabs sends {"status": "complete"})
            if status == "complete" or msg_type == "complete":
                logger.debug("Generation complete event received", extra=self._build_log_context())
                output_emitter.end_input()
                return False  # Stop processing

            # Check for errors - check multiple indicators
            if status == "error" or msg_type == "error" or resp.get("error") or resp.get("error_message"):
                await self._handle_error_message(resp)
                return False  # Stop processing on error

            # Handle other message types
            if msg_type == "audio":
                return await self._handle_audio_message(resp, output_emitter)
            elif msg_type == "event" or msg_type == "final":
                return await self._handle_event_message(resp, output_emitter)
            elif status == "streaming" or msg_type == "streaming" or msg_type == "status":
                # Streaming started - continue
                logger.debug("Streaming started", extra=self._build_log_context())
                return True
            else:
                logger.debug(f"Unknown message: {resp}", extra=self._build_log_context())
                return True

        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in WebSocket message: {e}",
                extra={**self._build_log_context(), "raw_data": msg_data[:200]},
            )
            return True  # Continue processing
        except Exception as e:
            logger.error(
                f"Error processing WebSocket message: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(f"Message processing error: {e}") from e

    async def _handle_audio_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        """Handle audio message with proper error handling."""
        try:
            # Check if audio is in data.audio (base64) or direct binary
            audio_data = resp.get("data", {}).get("audio", "")
            if not audio_data:
                logger.debug("Received empty audio data", extra=self._build_log_context())
                return True

            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            output_emitter.push(audio_bytes)

            return True

        except Exception as e:  # base64 decode error
            logger.error(f"Invalid base64 audio data: {e}", extra=self._build_log_context())
            # Don't stop processing for audio decode errors
            return True

    async def _handle_error_message(self, resp: dict) -> None:
        """Handle error messages from the API."""
        # Log the full response for debugging
        logger.debug(f"Error response received: {resp}", extra=self._build_log_context())

        # Check multiple possible locations for error message
        error_data = resp.get("data", {})
        if not isinstance(error_data, dict):
            error_data = {}

        # Try to get error message from various locations
        error_msg = (
            error_data.get("message") or
            error_data.get("error") or
            resp.get("message") or
            resp.get("error") or
            resp.get("error_message") or
            error_data.get("detail") or
            resp.get("detail") or
            (str(error_data) if error_data and error_data != {} else None) or
            "Unknown error"
        )

        error_code = error_data.get("code") or resp.get("code") or resp.get("error_code", "unknown")
        error_type = error_data.get("error") or error_data.get("type") or resp.get("type") or resp.get("error_type", "")

        # If we still have "Unknown error", log the full response
        if error_msg == "Unknown error" or not error_msg:
            logger.warning(
                f"Could not extract error message, full response: {resp}",
                extra={**self._build_log_context(), "full_response": resp}
            )
            # Try to create a more descriptive error message
            if resp:
                error_msg = f"API error: {json.dumps(resp)}"
            else:
                error_msg = "Unknown API error"

        logger.error(
            f"TTS API error: {error_msg}",
            extra={
                **self._build_log_context(),
                "error_code": error_code,
                "error_message": error_msg,
                "error_type": error_type,
                "full_response": resp,
            },
        )

        # Determine if error is recoverable based on error code/type
        recoverable_errors = ["rate_limit", "temporary_unavailable", "timeout", "connection"]
        is_recoverable = any(err in str(error_msg).lower() for err in recoverable_errors)

        # Check for specific error types
        if error_type == "credits_exhausted" or "credits" in str(error_msg).lower():
            raise APIStatusError(message=f"Credits exhausted: {error_msg}", status_code=402)
        elif error_type == "rate_limit_exceeded" or "rate limit" in str(error_msg).lower():
            raise APIStatusError(message=f"Rate limit exceeded: {error_msg}", status_code=429)
        elif "invalid api key" in str(error_msg).lower() or "authentication" in str(error_msg).lower():
            raise APIStatusError(message=f"Authentication failed: {error_msg}. Please check your API key.", status_code=401)

        if is_recoverable:
            raise APIConnectionError(f"Recoverable TTS API error: {error_msg}")
        else:
            raise APIStatusError(message=f"TTS API error: {error_msg}", status_code=500)

    async def _handle_event_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        """Handle event messages from the API."""
        event_data = resp.get("data", {})
        event_type = event_data.get("event_type") or resp.get("type")

        if event_type == "final" or event_type == "complete":
            logger.debug("Generation complete event received", extra=self._build_log_context())
            output_emitter.end_input()
            return False  # Stop processing
        else:
            logger.debug(f"Unknown event type: {event_type}", extra=self._build_log_context())
            return True

    def _build_log_context(self) -> dict:
        """Build consistent logging context."""
        return {
            "session_id": self._session_id,
            "connection_state": self._connection_state.value,
            "voice": self._tts._voice,
            "language": self._tts._language,
        }

    async def aclose(self) -> None:
        """Close the stream and cleanup resources."""
        logger.debug("Starting TTS stream cleanup", extra=self._build_log_context())

        self._connection_state = ConnectionState.DISCONNECTED

        # Cancel running tasks first
        tasks_to_cancel = []
        for task_attr in ["_send_task", "_recv_task"]:
            task = getattr(self, task_attr, None)
            if task and not task.done():
                tasks_to_cancel.append(task)

        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error cancelling tasks: {e}", extra=self._build_log_context())

        # Close WebSocket connection
        if self._ws_conn and not self._ws_conn.closed:
            try:
                await self._ws_conn.close()
                logger.debug("WebSocket connection closed", extra=self._build_log_context())
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}", extra=self._build_log_context())

        # Call parent cleanup
        try:
            await super().aclose()
        except Exception as e:
            logger.warning(f"Error in parent cleanup: {e}", extra=self._build_log_context())
