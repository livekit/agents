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
import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, AsyncGenerator

import aiohttp
import websockets
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import OutputFormat, Precision


RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
NUM_CHANNELS = 1


@dataclass
class _Options:
    voice_uuid: str
    project_uuid: str
    sample_rate: int
    precision: Precision
    output_format: OutputFormat
    binary_response: bool
    no_audio_header: bool


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_uuid: str,
        project_uuid: str,
        sample_rate: int = 44100,
        precision: Union[Precision, str] = Precision.PCM_16,
        output_format: Union[OutputFormat, str] = OutputFormat.WAV,
        binary_response: bool = False,
        no_audio_header: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        # Validate and set API key
        self._api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Resemble API key is required, either as argument or set RESEMBLE_API_KEY environment variable"
            )
        
        # Convert string enum values to their proper types if needed
        if isinstance(precision, str):
            precision = Precision(precision)
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)
            
        # Set options
        self._opts = _Options(
            voice_uuid=voice_uuid,
            project_uuid=project_uuid,
            sample_rate=sample_rate,
            precision=precision,
            output_format=output_format,
            binary_response=binary_response,
            no_audio_header=no_audio_header,
        )
        
        # HTTP session for REST API calls
        self._session = http_session
        self._private_session = None
        self._owns_session = False
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure a HTTP session is available."""
        if self._session:
            return self._session
        
        try:
            # Try to use the LiveKit utility if available in an agent context
            self._session = utils.http_context.http_session()
            return self._session
        except RuntimeError:
            # If not in an agent context, create our own session
            if not self._private_session:
                self._private_session = aiohttp.ClientSession()
                self._owns_session = True
            return self._private_session

    def update_options(
        self,
        *,
        voice_uuid: str | None = None,
        project_uuid: str | None = None,
        precision: Union[Precision, str, None] = None,
        output_format: Union[OutputFormat, str, None] = None,
        **kwargs,
    ) -> None:
        """Update TTS options."""
        if voice_uuid:
            self._opts.voice_uuid = voice_uuid
        if project_uuid:
            self._opts.project_uuid = project_uuid
        if precision:
            if isinstance(precision, str):
                precision = Precision(precision)
            self._opts.precision = precision
        if output_format:
            if isinstance(output_format, str):
                output_format = OutputFormat(output_format)
            self._opts.output_format = output_format

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> "ChunkedStream":
        """Synthesize text into speech using Resemble AI."""
        return ChunkedStream(
            tts=self,
            input_text=text,
            opts=self._opts,
            conn_options=conn_options,
            api_key=self._api_key,
            session=self._ensure_session(),
            should_close_session=False,  # Let TTS handle session lifecycle
        )

    def stream(
        self, *, conn_options: Optional[APIConnectOptions] = None
    ) -> "SynthesizeStream":
        """Create a streaming synthesis connection to Resemble AI."""
        stream = SynthesizeStream(
            tts=self,
            opts=self._opts,
            conn_options=conn_options,
            api_key=self._api_key,
        )
        self._streams.add(stream)
        return stream
    
    async def __aenter__(self) -> "TTS":
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and clean up resources."""
        await self.aclose()
        
    async def aclose(self) -> None:
        """Clean up resources."""
        # Close all active streams
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        
        # Close the private session if we own it
        if self._owns_session and self._private_session:
            await self._private_session.close()
            self._private_session = None
        
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text into speech in one go using Resemble AI's REST API."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _Options,
        conn_options: Optional[APIConnectOptions] = None,
        api_key: str | None = None,
        session: aiohttp.ClientSession,
        should_close_session: bool = False,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._session = session
        self._should_close_session = should_close_session
        self._segment_id = utils.shortuuid()

    async def _run(self) -> None:
        """Run the synthesis process using REST API."""
        request_id = utils.shortuuid()
        
        # Create request headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",  # Expect JSON response
        }
        
        # Create request payload
        payload = {
            "voice_uuid": self._opts.voice_uuid,
            "data": self._input_text,
            "sample_rate": self._opts.sample_rate,
            "output_format": self._opts.output_format,
        }
        
        # Add project_uuid if provided (optional for some accounts)
        if hasattr(self._opts, "project_uuid") and self._opts.project_uuid:
            payload["project_uuid"] = self._opts.project_uuid
        
        # Create decoder for audio processing
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        logger.debug(f"Making Resemble API request to {RESEMBLE_REST_API_URL}")
        logger.debug(f"Payload: {payload}")
        
        try:
            # Make the HTTP request with explicit timeout
            async with self._session.post(
                RESEMBLE_REST_API_URL, 
                headers=headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=30,  # 30 seconds total timeout
                    sock_connect=self._conn_options.timeout,
                ),
            ) as response:
                logger.debug(f"Received response from Resemble API: {response.status}")
                
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Resemble API error: {error_text}")
                    raise APIStatusError(
                        message=f"Resemble API error: {error_text}",
                        status_code=response.status,
                        request_id=request_id,
                        body=error_text,
                    )
                
                # Parse the JSON response
                response_json = await response.json()
                logger.debug(f"Response JSON received with keys: {response_json.keys()}")
                
                # Check for success
                if not response_json.get("success", False):
                    issues = response_json.get("issues", ["Unknown error"])
                    error_msg = "; ".join(issues)
                    logger.error(f"Resemble API returned success=false: {error_msg}")
                    raise APIStatusError(
                        message=f"Resemble API returned failure: {error_msg}",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )
                
                # Extract base64-encoded audio content
                audio_content_b64 = response_json.get("audio_content")
                if not audio_content_b64:
                    logger.error("No audio_content found in Resemble API response")
                    raise APIStatusError(
                        message="No audio content in response",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )
                
                # Decode base64 to get raw audio bytes
                audio_bytes = base64.b64decode(audio_content_b64)
                logger.debug(f"Decoded {len(audio_bytes)} bytes of audio data")
                
                # Create audio emitter
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=self._segment_id,
                )
                
                # Push audio data to decoder
                decoder.push(audio_bytes)
                decoder.end_input()
                
                # Emit audio frames
                async for frame in decoder:
                    emitter.push(frame)
                
                # Final flush of the emitter
                emitter.flush()
                
        except aiohttp.ClientResponseError as e:
            # Handle HTTP errors (4xx, 5xx)
            logger.error(f"Resemble API error: HTTP {e.status} - {e.message}")
            raise APIStatusError(
                message=f"Resemble API error: {e.message}",
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except asyncio.TimeoutError as e:
            logger.error("Timeout while connecting to Resemble API")
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            logger.error(f"Connection error to Resemble API: {e}")
            raise APIConnectionError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during synthesis: {e}")
            raise APIConnectionError(f"Error during synthesis: {e}") from e
        finally:
            await decoder.aclose()
            # Clean up session if we own it
            if self._should_close_session:
                await self._session.close()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Resemble AI WebSocket API.
    
    This implementation connects to Resemble's WebSocket API for real-time streaming
    synthesis. Note that this requires a Business plan subscription with Resemble AI.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _Options,
        conn_options: Optional[APIConnectOptions] = None,
        api_key: str | None = None,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._request_id = 0
        self._running = False
        self._websocket = None
        
        # Channels for communication between components
        self._text_ch = asyncio.Queue()
        self._audio_ch = asyncio.Queue()
        
        # Tasks for processing
        self._websocket_task = None
        self._processing_task = None
        self._closed = False

    async def synthesize_text(self, text: str) -> None:
        """Queue text for synthesis."""
        if self._closed:
            raise RuntimeError("Stream is closed")
            
        await self._text_ch.put(text)
        if not self._running:
            # Start processing if not already running
            self._running = True
            self._processing_task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        """Close the stream and clean up resources."""
        self._closed = True
        
        # Close the text channel to signal the end
        if self._running:
            await self._text_ch.put(None)  # Signal end of input
        
        # Cancel any running tasks
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket connection if open
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            
        await super().aclose()

    async def _connect_websocket(self) -> websockets.WebSocketClientProtocol:
        """Connect to the Resemble WebSocket API."""
        logger.debug(f"Connecting to Resemble WebSocket API: {RESEMBLE_WEBSOCKET_URL}")
        return await websockets.connect(
            RESEMBLE_WEBSOCKET_URL,
            extra_headers={"Authorization": f"Bearer {self._api_key}"},
            ping_interval=5,
            ping_timeout=10,
        )

    async def _run(self) -> None:
        """Main processing loop for the streaming synthesis."""
        
        # Initialize decoder for audio processing
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        try:
            # Connect to the WebSocket
            self._websocket = await self._connect_websocket()
            request_id = utils.shortuuid()
            segment_id = utils.shortuuid()
            
            # Create audio emitter
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )
            
            # Process text input and WebSocket messages
            while not self._closed:
                # Wait for text to synthesize
                text = await self._text_ch.get()
                
                # None signals end of input
                if text is None:
                    logger.debug("Received end of input signal")
                    break
                
                # Prepare request payload
                logger.debug(f"Synthesizing text: {text}")
                self._mark_started()
                
                payload = {
                    "voice_uuid": self._opts.voice_uuid,
                    "project_uuid": self._opts.project_uuid,
                    "data": text,
                    "request_id": self._request_id,
                    "binary_response": self._opts.binary_response,
                    "output_format": self._opts.output_format,
                    "sample_rate": self._opts.sample_rate,
                    "precision": self._opts.precision,
                    "no_audio_header": self._opts.no_audio_header,
                }
                
                # Send synthesis request
                await self._websocket.send(json.dumps(payload))
                self._request_id += 1
                
                # Process responses until audio_end is received
                first_audio = True
                while True:
                    message = await self._websocket.recv()
                    
                    # Handle binary response (when binary_response=true)
                    if isinstance(message, bytes):
                        logger.debug(f"Received binary audio data: {len(message)} bytes")
                        decoder.push(message)
                        
                        # Process decoded audio frames
                        async for frame in decoder:
                            emitter.push(frame)
                        continue
                    
                    # Handle JSON response
                    try:
                        data = json.loads(message)
                        
                        # Debug log the first message
                        if first_audio:
                            logger.debug(f"Received first JSON message: {data.get('type')}")
                            first_audio = False
                        
                        # Handle audio data
                        if data.get("type") == "audio":
                            # Decode base64 audio content
                            audio_data = base64.b64decode(data["audio_content"])
                            
                            # Push to decoder
                            decoder.push(audio_data)
                            
                            # Process decoded audio frames
                            async for frame in decoder:
                                emitter.push(frame)
                        
                        # Handle end of audio
                        elif data.get("type") == "audio_end":
                            logger.debug("Received audio_end message")
                            # Complete current segment
                            emitter.flush()
                            break
                        
                        # Handle errors
                        elif data.get("type") == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.error(f"Resemble WebSocket API error: {error_msg}")
                            raise APIStatusError(
                                message=f"Resemble API error: {error_msg}",
                                status_code=data.get("status_code", 500),
                                request_id=str(request_id),
                                body=None,
                            )
                    except json.JSONDecodeError:
                        logger.error("Failed to decode JSON response")
                
                # Mark the text as processed
                self._text_ch.task_done()
            
        except asyncio.CancelledError:
            logger.debug("Streaming synthesis was cancelled")
            raise
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
            raise APIConnectionError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            logger.error(f"Error during streaming synthesis: {e}")
            raise APIConnectionError(f"Error during streaming synthesis: {e}") from e
        finally:
            # Clean up resources
            await decoder.aclose()
            
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            
            self._running = False 