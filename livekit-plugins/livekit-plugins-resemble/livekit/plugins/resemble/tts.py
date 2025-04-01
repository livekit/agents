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
import time
import weakref
from dataclasses import dataclass
from typing import Optional, Union

import aiohttp
import websockets
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger

RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
NUM_CHANNELS = 1
DEFAULT_VOICE_UUID = '55592656'

@dataclass
class _Options:
    voice_uuid: str
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_uuid: str | None = DEFAULT_VOICE_UUID,
        sample_rate: int = 44100,
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
            
        # Set options
        self._opts = _Options(
            voice_uuid=voice_uuid,
            sample_rate=sample_rate,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        
        # Create a connection pool for WebSockets
        self._pool = utils.ConnectionPool[websockets.WebSocketClientProtocol](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self) -> websockets.WebSocketClientProtocol:
        """Connect to the Resemble WebSocket API."""
        return await websockets.connect(
            RESEMBLE_WEBSOCKET_URL,
            extra_headers={"Authorization": f"Bearer {self._api_key}"},
            ping_interval=5,
            ping_timeout=10,
        )
    
    async def _close_ws(self, ws: websockets.WebSocketClientProtocol):
        """Close the WebSocket connection."""
        await ws.close()

    def update_options(
        self,
        *,
        voice_uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Update TTS options."""
        if voice_uuid:
            self._opts.voice_uuid = voice_uuid

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
            session=self._session,
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
            pool=self._pool,
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
        
        # Close the WebSocket connection pool
        await self._pool.aclose()
        
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
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._session = session
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
        }
        
        # Create decoder for audio processing
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
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
                if not response.ok:
                    error_text = await response.text()
                    raise APIStatusError(
                        message=f"Resemble API error: {error_text}",
                        status_code=response.status,
                        request_id=request_id,
                        body=error_text,
                    )
                
                # Parse the JSON response
                response_json = await response.json()
                
                # Check for success
                if not response_json.get("success", False):
                    issues = response_json.get("issues", ["Unknown error"])
                    error_msg = "; ".join(issues)
                    raise APIStatusError(
                        message=f"Resemble API returned failure: {error_msg}",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )
                
                # Extract base64-encoded audio content
                audio_content_b64 = response_json.get("audio_content")
                if not audio_content_b64:
                    raise APIStatusError(
                        message="No audio content in response",
                        status_code=response.status,
                        request_id=request_id,
                        body=json.dumps(response_json),
                    )
                
                # Decode base64 to get raw audio bytes
                audio_bytes = base64.b64decode(audio_content_b64)
                
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
        pool: utils.ConnectionPool[websockets.WebSocketClientProtocol],
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._request_id = 0
        self._running = False
        self._websocket = None
        self._pool = pool
        
        # Channels for communication between components
        self._text_ch = asyncio.Queue()
        self._audio_ch = asyncio.Queue()
        
        # Tasks for processing
        self._websocket_task = None
        self._processing_task = None
        self._closed = False
        
        # Create a task to monitor the base class's input channel
        self._input_monitor_task = asyncio.create_task(self._monitor_input_channel())
        
    async def _monitor_input_channel(self) -> None:
        """Monitor the input channel from the base class and forward to our text channel."""
        try:
            buffer = ""
            word_count = 0
            MIN_WORDS_TO_BUFFER = 5  # Buffer at least this many words before sending
            
            async for item in self._input_ch:
                
                if isinstance(item, self._FlushSentinel):
                    # When we get a flush sentinel, send any buffered text
                    if buffer:
                        await self._text_ch.put(buffer)
                        buffer = ""
                        word_count = 0
                    # Signal end of input
                    await self._text_ch.put(None)
                    continue
                else:
                    # It's a text token, add to buffer
                    buffer += item
                    
                    # Count words in the buffer
                    if item.strip() and (item.endswith(' ') or item.endswith('\n')):
                        word_count += 1
                    
                    # Send buffer when we have enough words or hit sentence-ending punctuation
                    if word_count >= MIN_WORDS_TO_BUFFER or any(buffer.rstrip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                        await self._text_ch.put(buffer)
                        buffer = ""
                        word_count = 0
                
            # End of input - send any remaining text in buffer
            if buffer:
                await self._text_ch.put(buffer)
        except Exception as e:
            logger.error(f"Error in input channel monitor: {e}")
        finally:
            if not self._closed:
                # Signal end of input if our monitor is shutting down unexpectedly
                await self._text_ch.put(None)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before sending to Resemble API.
        
        This ensures punctuation is properly handled by combining it with adjacent words.
        """
        # Skip if text is empty or None
        if not text or not text.strip():
            return text
            
        # If text is just punctuation, add a space before it to avoid errors
        if text.strip() in ",.!?;:":
            return " " + text
            
        return text

    async def synthesize_text(self, text: str) -> None:
        """Queue text for synthesis."""
        if self._closed:
            raise RuntimeError("Stream is closed")
            
        # Preprocess text before sending
        processed_text = self._preprocess_text(text)
        await self._text_ch.put(processed_text)
        
        if not self._running:
            # Start processing if not already running
            self._running = True
            self._processing_task = asyncio.create_task(self._run())
        
        # Wait for the text to be processed
        await self._text_ch.join()
        
        # Signal end of input - this will close the channel
        # Note: We don't call flush() here because it's already done in end_input()
        self.end_input()

    async def aclose(self) -> None:
        """Close the stream and clean up resources."""
        self._closed = True
        
        # Close the text channel to signal the end
        if self._running:
            await self._text_ch.put(None)  # Signal end of input
        
        # Cancel the input monitor task
        if self._input_monitor_task and not self._input_monitor_task.done():
            self._input_monitor_task.cancel()
            try:
                await self._input_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel any running tasks
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            
        await super().aclose()

    async def _run(self) -> None:
        """Main processing loop for the streaming synthesis."""
        
        # Initialize decoder for audio processing
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        try:
            request_id = utils.shortuuid()
            segment_id = utils.shortuuid()
            
            # Create audio emitter
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )
            
            # Track pending requests to ensure all responses are received
            pending_requests = set()
            
            async with self._pool.connection() as websocket:
                # Start a separate task to handle WebSocket messages
                async def _ws_recv_task():
                    try:
                        while not self._closed:
                            message = await websocket.recv()
                            
                            # Handle JSON response
                            try:
                                data = json.loads(message)
                                
                                # Handle audio data
                                if data.get("type") == "audio":
                                    # Decode base64 audio content
                                    audio_data = base64.b64decode(data["audio_content"])
                                    
                                    try:
                                        # For PCM_16, each sample is 2 bytes (16 bits)
                                        bytes_per_sample = 2
                                        samples_per_channel = len(audio_data) // bytes_per_sample
                                        
                                        # Create audio frame directly from the PCM data
                                        frame = rtc.AudioFrame(
                                            data=audio_data,
                                            samples_per_channel=samples_per_channel,
                                            sample_rate=self._opts.sample_rate,
                                            num_channels=NUM_CHANNELS,
                                        )

                                        emitter.push(frame)
                                        
                                        emitter.flush()
                                        
                                    except Exception as e:
                                        logger.error(f"Error processing audio data: {e}", exc_info=True)
                                
                                # Handle end of audio
                                elif data.get("type") == "audio_end":
                                    # Complete current segment
                                    emitter.flush()
                                    
                                    # Mark request as completed if request_id is present
                                    if "request_id" in data:
                                        req_id = data["request_id"]
                                        if req_id in pending_requests:
                                            pending_requests.remove(req_id)
                                    
                                # Handle errors
                                elif data.get("type") == "error":
                                    error_msg = data.get("message", "Unknown error")
                                    logger.error(f"Resemble WebSocket API error: {error_msg}")
                                    
                                    # Don't raise an error for punctuation-only inputs
                                    if "would not generate any audio" in error_msg and data.get("request_id") in pending_requests:
                                        req_id = data.get("request_id")
                                        pending_requests.remove(req_id)
                                    else:
                                        raise APIStatusError(
                                            message=f"Resemble API error: {error_msg}",
                                            status_code=data.get("status_code", 500),
                                            request_id=str(request_id),
                                            body=None,
                                        )
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON response: {message}")
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"WebSocket connection closed: {e}")
                        if not self._closed:
                            raise APIConnectionError(f"WebSocket connection closed unexpectedly: {e}")
                    except Exception as e:
                        logger.error(f"Error in WebSocket receive task: {e}")
                        if not self._closed:
                            raise
                
                # Start WebSocket receive task
                ws_task = asyncio.create_task(_ws_recv_task())
                
                # Process text input
                try:
                    while not self._closed:
                        # Wait for text to synthesize
                        text = await self._text_ch.get()
                        
                        # None signals end of input
                        if text is None:
                            break
                        
                        if not text.strip():
                            self._text_ch.task_done()
                            continue
                        
                        # Preprocess text before sending
                        text = self._preprocess_text(text)
                        
                        self._mark_started()
                        
                        payload = {
                            "voice_uuid": self._opts.voice_uuid,
                            "data": text,
                            "request_id": self._request_id,
                            "sample_rate": self._opts.sample_rate,
                            "precision": "PCM_16",
                            "no_audio_header": True,
                        }
                        
                        # Add request to pending set
                        pending_requests.add(self._request_id)
                        
                        # Send synthesis request
                        await websocket.send(json.dumps(payload))
                        self._request_id += 1
                        
                        # Mark the text as processed
                        self._text_ch.task_done()
                    
                    # Wait for all pending requests to complete
                    if pending_requests:
                        # Wait with a timeout to avoid hanging indefinitely
                        wait_start = time.time()
                        while pending_requests and (time.time() - wait_start) < 5.0:
                            await asyncio.sleep(0.1)
                        
                        if pending_requests:
                            logger.warning(f"Timed out waiting for {len(pending_requests)} audio responses")
                    
                finally:
                    # Cancel WebSocket task
                    if not ws_task.done():
                        ws_task.cancel()
                        try:
                            await ws_task
                        except asyncio.CancelledError:
                            pass
            
        except asyncio.CancelledError:
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
            
            self._running = False 