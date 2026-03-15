from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)

from .log import logger

# Talkscriber TTS WebSocket API endpoint
# Support environment variables for flexible deployment
# Default to Talkscriber API server (not localhost) as per reference implementation
TTS_SERVER_HOST = os.environ.get("TTS_SERVER_HOST", "api.talkscriber.com")
TTS_SERVER_PORT = int(os.environ.get("TTS_SERVER_PORT", "9099"))
TTS_SERVER_USE_SSL = os.environ.get("TTS_SERVER_USE_SSL", "true").lower() == "true"

# Build URLs based on environment
_protocol = "wss" if TTS_SERVER_USE_SSL else "ws"
_http_protocol = "https" if TTS_SERVER_USE_SSL else "http"

BASE_URL = f"{_protocol}://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}"
BASE_REST_URL = f"{_http_protocol}://{TTS_SERVER_HOST}:40123/api/tts"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    voice: str
    encoding: str
    sample_rate: int
    language: str
    speed: float
    pitch: float
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "alloy",
        model: str | None = None,  # Backward compatibility with Deepgram-style API
        encoding: str = "linear16",
        sample_rate: int = 24000,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 0.0,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        base_rest_url: str = BASE_REST_URL,
        use_streaming: bool = True,
        word_tokenizer: tokenize.WordTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
        audio_buffer_size: int = 10,
    ) -> None:
        """
        Create a new instance of Talkscriber TTS.

        Args:
            voice (str): TTS voice to use. Defaults to "alloy".
            model (str): Alias for voice (for Deepgram compatibility). If provided, overrides voice.
            encoding (str): Audio encoding to use. Defaults to "linear16".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            language (str): Language code for synthesis. Defaults to "en".
            speed (float): Speech speed (0.25 to 4.0). Defaults to 1.0.
            pitch (float): Speech pitch (-20.0 to 20.0). Defaults to 0.0.
            api_key (str): Talkscriber API key. If not provided, will look for TALKSCRIBER_API_KEY in environment.
            base_url (str): Base WebSocket URL for Talkscriber TTS API. Defaults to "ws://localhost:9099".
            base_rest_url (str): Base REST URL for Talkscriber TTS API. Defaults to "http://localhost:40123/api/tts".
            use_streaming (bool): Whether to use WebSocket-based streaming instead of the REST API. Defaults to True.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
            audio_buffer_size (int): Number of audio packets to buffer before streaming. 0 means no buffering (immediate streaming). Defaults to 0.

        """
        # Handle backward compatibility: model parameter maps to voice
        if model is not None:
            voice = model
            logger.info(f"Using model parameter '{model}' as voice")

        # Initialize word_tokenizer if not provided
        if word_tokenizer is None:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        logger.info(
            f"TTS initialized with voice: {voice}, encoding: {encoding}, sample_rate: {sample_rate}, language: {language}, speed: {speed}, pitch: {pitch}"
        )

        api_key = api_key or os.environ.get("TALKSCRIBER_API_KEY")
        if not api_key:
            raise ValueError(
                "Talkscriber API key required. Set TALKSCRIBER_API_KEY or provide api_key."
            )

        self._opts = _TTSOptions(
            voice=voice,
            encoding=encoding,
            sample_rate=sample_rate,
            language=language,
            speed=speed,
            pitch=pitch,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
        self._api_key = api_key
        self._base_url = base_url
        self._base_rest_url = base_rest_url
        self._use_streaming = use_streaming
        self._audio_buffer_size = audio_buffer_size
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

        logger.info(
            f"TTS initialized with voice: {voice}, encoding: {encoding}, sample_rate: {sample_rate}, language: {language}, speed: {speed}, pitch: {pitch}"
        )
        logger.info(f"API key: {api_key}")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"Base REST URL: {base_rest_url}")
        logger.info(f"Use streaming: {use_streaming}")
        logger.info(f"Word tokenizer: {word_tokenizer}")
        logger.info(f"HTTP session: {http_session}")

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            logger.info(f"Connecting to WebSocket at {self._base_url}, is this local?")

            # Use the provided timeout for initial connection
            connect_timeout = timeout

            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._base_url,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=60, connect=30),
                ),
                connect_timeout,
            )
            logger.info("WebSocket connection established successfully")
            return ws
        except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
            logger.error(f"Failed to connect to WebSocket at {self._base_url}: {e}")
            logger.warning(
                "Consider setting TTS_SERVER_HOST environment variable if running in containers"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to WebSocket: {e}")
            raise

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: str | None = None,
        model: str | None = None,  # Backward compatibility
        sample_rate: int | None = None,
        language: str | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> None:
        """
        Update TTS options.

        Args:
            voice (str): TTS voice to use.
            model (str): Alias for voice (for Deepgram compatibility). If provided, overrides voice.
            sample_rate (int): Sample rate of audio.
            language (str): Language code for synthesis.
            speed (float): Speech speed (0.25 to 4.0).
            pitch (float): Speech pitch (-20.0 to 20.0).
        """
        # Handle backward compatibility: model parameter maps to voice
        if model is not None:
            voice = model
            logger.info(f"Update: using model parameter '{model}' as voice")

        if voice is not None:
            self._opts.voice = voice
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if language is not None:
            self._opts.language = language
        if speed is not None:
            self._opts.speed = speed
        if pitch is not None:
            self._opts.pitch = pitch

        # Talkscriber sets options upon connection, so we need to invalidate the pool
        # to get a new connection with the updated options
        self._pool.invalidate()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        # Use default conn_options if not provided
        if conn_options is None:
            conn_options = APIConnectOptions()

        return ChunkedStream(
            tts=self,
            input_text=text,
            base_rest_url=self._base_rest_url,
            api_key=self._api_key,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(self, *, conn_options: APIConnectOptions | None = None) -> SynthesizeStream:
        if not self._use_streaming:
            raise ValueError("Streaming is disabled. Use synthesize() for chunked synthesis.")

        # Check if we can reach the WebSocket server
        logger.debug(f"Creating TTS stream connecting to {self._base_url}")

        # Use default conn_options if not provided
        if conn_options is None:
            conn_options = APIConnectOptions()

        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            pool=self._pool,
            opts=self._opts,
            api_key=self._api_key,
            audio_buffer_size=self._audio_buffer_size,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._use_streaming:
            self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        base_rest_url: str,
        api_key: str,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._base_rest_url = base_rest_url
        self._api_key = api_key
        self._conn_options = conn_options
        logger.info(
            f"ChunkedStream initialized with voice: {self._opts.voice}, encoding: {self._opts.encoding}, sample_rate: {self._opts.sample_rate}, language: {self._opts.language}, speed: {self._opts.speed}, pitch: {self._opts.pitch}"
        )
        logger.info(f"Base REST URL: {base_rest_url}")
        logger.info(f"text input: {input_text}")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        # Initialize the output emitter
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        # Start the audio segment
        output_emitter.start_segment(segment_id=segment_id)

        try:
            payload = {
                "text": self._input_text,
                "voice": self._opts.voice,
                "language": self._opts.language,
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
                "speed": self._opts.speed,
                "pitch": self._opts.pitch,
            }

            async with self._session.post(
                self._base_rest_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._conn_options.timeout,
            ) as res:
                if res.status != 200:
                    error_body = None
                    try:
                        error_body = await res.json()
                    except Exception as e:
                        logger.debug(f"Failed to parse error response as JSON: {e}")

                    raise APIStatusError(
                        message=res.reason or "Unknown error occurred.",
                        status_code=res.status,
                        request_id=request_id,
                        body=error_body,
                    )

                async for bytes_data, _ in res.content.iter_chunks():
                    output_emitter.push(bytes_data)

                # End the segment and flush
                output_emitter.end_segment()
                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        api_key: str,
        audio_buffer_size: int = 0,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._pool = pool
        self._api_key = api_key
        self._audio_buffer_size = audio_buffer_size
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        self._session_id = utils.shortuuid()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        # Initialize the output emitter with stream parameters
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        @utils.log_exceptions(logger=logger)
        async def _accumulate_and_process():
            # Use a single WebSocket connection for the entire session
            async with self._pool.connection(timeout=30.0) as ws:
                # Send authentication once at the start
                # Match the reference implementation format (see ts-client/tts/client/tts_client.py)
                # NOTE: Server requires "text" field in auth message, even if empty for streaming mode
                auth_message = {
                    "uid": self._session_id,
                    "auth": self._api_key,
                    "type": "tts",
                    "speaker_name": self._opts.voice,  # Server expects "speaker_name" not "voice"
                    "text": "",  # Required field - empty for streaming mode
                }
                await ws.send_str(json.dumps(auth_message))
                logger.debug(
                    f"Sent TTS authentication for session with speaker: {self._opts.voice}"
                )

                # Wait for authentication confirmation before proceeding
                auth_confirmed = False
                while not auth_confirmed:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        resp = json.loads(msg.data)
                        if resp.get("type") == "authenticated":
                            logger.debug("TTS authentication confirmed by server")
                            auth_confirmed = True
                        elif resp.get("type") == "error":
                            error_msg = resp.get("message", "Authentication failed")
                            raise APIStatusError(
                                f"Talkscriber TTS auth error: {error_msg}",
                                request_id=self._session_id,
                            )
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIConnectionError("WebSocket closed during authentication")

                # Use the provided output_emitter
                emitter = output_emitter

                # Start audio receiving task
                audio_task = asyncio.create_task(
                    self._receive_audio_continuous(ws, emitter, request_id, self._audio_buffer_size)
                )

                try:
                    # Accumulate text with time-based sending
                    accumulated_text = ""
                    input_iterator = self._input_ch.__aiter__()

                    while True:
                        try:
                            # Wait for input with 0.5 second timeout
                            input = await asyncio.wait_for(input_iterator.__anext__(), timeout=0.5)

                            if isinstance(input, str):
                                logger.debug(f"Received text chunk: '{input}'")
                                text_msg = {
                                    "type": "chunk_speak",
                                    "text": input,
                                    "speaker": self._opts.voice,  # Include speaker for consistency
                                }
                                await ws.send_str(json.dumps(text_msg))
                                # Continue to next iteration to reset timeout
                            elif isinstance(input, self._FlushSentinel):
                                # Send any remaining text on flush
                                if accumulated_text.strip():
                                    text_msg = {
                                        "type": "speak",
                                        "text": accumulated_text.strip(),
                                        "speaker": self._opts.voice,  # Match reference implementation format
                                    }
                                    await ws.send_str(json.dumps(text_msg))
                                    logger.debug(
                                        f"Sent final text chunk: '{accumulated_text.strip()[:50]}...'"
                                    )
                                    accumulated_text = ""

                                # Send flush to complete synthesis
                                flush_msg = {"type": "flush"}
                                await ws.send_str(json.dumps(flush_msg))
                                logger.debug("Sent flush message")

                                # Wait for audio completion
                                break

                        except asyncio.TimeoutError:
                            logger.debug("Timeout error")

                        except StopAsyncIteration:
                            # No more input available
                            break

                    # Wait for audio processing to complete
                    await audio_task

                except Exception as e:
                    logger.error(f"Error in _accumulate_and_process: {e}")
                    audio_task.cancel()
                    raise

        tasks = [
            asyncio.create_task(_accumulate_and_process()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _receive_audio_continuous(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        emitter: tts.AudioEmitter,
        request_id: str,
        buffer_size: int = 0,
    ):
        """Receive and process audio from WebSocket continuously"""
        try:
            # Start the audio segment before pushing any frames
            segment_id = utils.shortuuid()
            emitter.start_segment(segment_id=segment_id)

            # Buffer for collecting audio packets before streaming
            audio_buffer = []
            buffering_active = buffer_size > 0

            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.debug(f"WebSocket closed: {msg.type}")
                    break

                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Raw audio data
                    data = msg.data

                    if buffering_active:
                        # Buffer the audio packet
                        audio_buffer.append(data)
                        logger.debug(f"Buffered audio packet {len(audio_buffer)}/{buffer_size}")

                        # Once we have enough packets, flush the buffer and switch to direct streaming
                        if len(audio_buffer) >= buffer_size:
                            logger.debug(
                                f"Buffer full ({len(audio_buffer)} packets), flushing and switching to direct streaming"
                            )
                            for buffered_data in audio_buffer:
                                emitter.push(buffered_data)
                            audio_buffer.clear()
                            buffering_active = False  # Switch to direct streaming
                    else:
                        # Direct streaming (after buffer has been flushed)
                        emitter.push(data)

                elif msg.type == aiohttp.WSMsgType.TEXT:
                    resp = json.loads(msg.data)
                    msg_type = resp.get("type")

                    if msg_type == "audio_complete":
                        # Synthesis complete - flush any remaining buffered audio first
                        if audio_buffer:
                            logger.debug(f"Flushing {len(audio_buffer)} remaining buffered packets")
                            for buffered_data in audio_buffer:
                                emitter.push(buffered_data)
                            audio_buffer.clear()

                        # End the segment and flush
                        emitter.end_segment()
                        emitter.flush()
                        logger.debug("Audio synthesis session completed")
                        break

                    elif msg_type == "error":
                        error_msg = resp.get("message", "Unknown TTS error")
                        logger.error(f"Talkscriber TTS error: {error_msg}")
                        raise APIStatusError(
                            f"Talkscriber TTS error: {error_msg}", request_id=request_id
                        )

                    elif msg_type == "server_ready":
                        logger.debug("Talkscriber TTS server ready")
                        continue

                    elif msg_type == "warning":
                        logger.warning("Talkscriber TTS warning: %s", resp.get("message"))

                    else:
                        logger.debug("Unknown TTS message type: %s", resp)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    raise APIStatusError(
                        f"WebSocket error: {ws.exception()}",
                        request_id=request_id,
                    )

        except asyncio.CancelledError:
            logger.debug("Audio receive task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            raise


def configure_for_server(
    host: str = "localhost", port: int = 9099, use_ssl: bool = False
) -> tuple[str, str]:
    """
    Configure TTS URLs for a specific server.

    Args:
        host (str): Server hostname or IP address. Defaults to "localhost".
        port (int): WebSocket port. Defaults to 9099.
        use_ssl (bool): Whether to use WSS/HTTPS. Defaults to False.

    Returns:
        tuple[str, str]: (websocket_url, rest_url)

    Examples:
        # Local development
        ws_url, rest_url = configure_for_server()

        # Remote server
        ws_url, rest_url = configure_for_server("192.168.1.100", 9099)

        # Public server with SSL
        ws_url, rest_url = configure_for_server("your-domain.com", 9099, use_ssl=True)
    """
    protocol = "wss" if use_ssl else "ws"
    http_protocol = "https" if use_ssl else "http"

    websocket_url = f"{protocol}://{host}:{port}"
    rest_url = f"{http_protocol}://{host}:40123/api/tts"  # Assuming REST is always on 40123

    return websocket_url, rest_url
