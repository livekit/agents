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
from .models import OutputFormat, Precision

RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
NUM_CHANNELS = 1


@dataclass
class _Options:
    voice_uuid: str
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
        sample_rate: int = 44100,
        precision: Union[Precision, str] = Precision.PCM_16,
        output_format: Union[OutputFormat, str] = OutputFormat.WAV,
        binary_response: bool = False,
        no_audio_header: bool = True,
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
        precision: Union[Precision, str, None] = None,
        output_format: Union[OutputFormat, str, None] = None,
        **kwargs,
    ) -> None:
        """Update TTS options."""
        if voice_uuid:
            self._opts.voice_uuid = voice_uuid
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
        
        # Create a lock to prevent multiple coroutines from accessing the WebSocket simultaneously
        self._ws_lock = asyncio.Lock()

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
            logger.debug("Input channel monitor started")
            buffer = ""
            word_count = 0
            MIN_WORDS_TO_BUFFER = 5  # Buffer at least this many words before sending
            
            async for item in self._input_ch:
                logger.debug(f"Received item from input channel: {type(item)}")
                
                if isinstance(item, self._FlushSentinel):
                    # When we get a flush sentinel, send any buffered text
                    if buffer:
                        logger.debug(f"Flushing buffered text: {buffer}")
                        await self._text_ch.put(buffer)
                        buffer = ""
                        word_count = 0
                    # Signal end of input
                    logger.debug("Signaling end of input")
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
                        logger.debug(f"Sending buffered text: {buffer}")
                        await self._text_ch.put(buffer)
                        buffer = ""
                        word_count = 0
                
            # End of input - send any remaining text in buffer
            if buffer:
                logger.debug(f"Sending final buffered text: {buffer}")
                await self._text_ch.put(buffer)
        except Exception as e:
            logger.error(f"Error in input channel monitor: {e}")
        finally:
            if not self._closed:
                # Signal end of input if our monitor is shutting down unexpectedly
                logger.debug("Signaling end of input")
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
        
        # Close websocket connection if open
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            
        await super().aclose()

    async def _connect_websocket(self) -> websockets.WebSocketClientProtocol:
        """Connect to the Resemble WebSocket API."""
        logger.debug(f"Connecting to Resemble WebSocket API: {RESEMBLE_WEBSOCKET_URL}")
        
        # Use the lock to ensure only one coroutine can connect at a time
        async with self._ws_lock:
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
            
            # Create a lock to prevent multiple coroutines from receiving simultaneously
            self._ws_lock = asyncio.Lock()
            
            # Track pending requests to ensure all responses are received
            pending_requests = set()
            
            # Start a separate task to handle WebSocket messages
            async def _ws_recv_task():
                try:
                    while not self._closed and self._websocket:
                        # Use a lock to ensure only one coroutine can call recv() at a time
                        async with self._ws_lock:
                            message = await self._websocket.recv()
                        
                        # Handle binary response (when binary_response=true)
                        if isinstance(message, bytes):
                            logger.info(f"Received binary audio data: {len(message)} bytes")
                            decoder.push(message)
                            
                            # Process decoded audio frames
                            frame_count = 0
                            async for frame in decoder:
                                frame_count += 1
                                logger.info(f"Emitting audio frame {frame_count}: {frame.samples_per_channel} samples")
                                emitter.push(frame)
                            
                            # If we didn't get any frames, log a warning
                            if frame_count == 0:
                                logger.warning("No audio frames were decoded from the audio data")
                            else:
                                logger.info(f"Successfully decoded {frame_count} frames from audio data")
                                # Immediately flush after each chunk to ensure frames are available
                                emitter.flush()
                            continue
                        
                        # Handle JSON response
                        try:
                            data = json.loads(message)
                            logger.debug(f"Received JSON message: {data.get('type')}")
                            
                            # Handle audio data
                            if data.get("type") == "audio":
                                # Decode base64 audio content
                                audio_data = base64.b64decode(data["audio_content"])
                                logger.info(f"Received audio data: {len(audio_data)} bytes")
                                
                                try:
                                    # When no_audio_header is True, we get raw PCM samples
                                    # Convert the raw bytes directly to an audio frame
                                    if self._opts.no_audio_header:
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
                                        
                                        logger.info(f"Created direct audio frame with {samples_per_channel} samples from raw PCM data")
                                        emitter.push(frame)
                                    else:
                                        # If we have an audio header (WAV format)
                                        data_chunk_pos = audio_data.find(b'data')
                                        if data_chunk_pos >= 0:
                                            # Skip the 'data' marker (4 bytes) and the chunk size (4 bytes)
                                            pcm_data_start = data_chunk_pos + 8
                                            # Get the chunk size (4 bytes little-endian)
                                            chunk_size = int.from_bytes(audio_data[data_chunk_pos+4:data_chunk_pos+8], byteorder='little')
                                            logger.info(f"Found WAV data chunk at position {data_chunk_pos}, size: {chunk_size}, data starts at {pcm_data_start}")
                                            pcm_data = audio_data[pcm_data_start:pcm_data_start+chunk_size]
                                        else:
                                            # If we can't find the data chunk, assume standard 44-byte header
                                            pcm_data = audio_data[44:]
                                            logger.info("Using standard 44-byte WAV header offset")
                                        
                                        # Create an audio frame from the PCM data
                                        samples_per_channel = len(pcm_data) // (2 * NUM_CHANNELS)  # 2 bytes per sample (16-bit PCM)
                                        frame = rtc.AudioFrame(
                                            data=pcm_data,
                                            samples_per_channel=samples_per_channel,
                                            sample_rate=self._opts.sample_rate,
                                            num_channels=NUM_CHANNELS,
                                        )
                                        
                                        logger.info(f"Created audio frame with {samples_per_channel} samples from WAV data")
                                        emitter.push(frame)
                                    
                                    # Flush after each chunk
                                    emitter.flush()
                                    
                                except Exception as e:
                                    logger.error(f"Error processing audio data: {e}", exc_info=True)
                            
                            # Handle end of audio
                            elif data.get("type") == "audio_end":
                                logger.info("Received audio_end message")
                                # Complete current segment
                                emitter.flush()
                                
                                # Mark request as completed if request_id is present
                                if "request_id" in data:
                                    req_id = data["request_id"]
                                    if req_id in pending_requests:
                                        pending_requests.remove(req_id)
                                        logger.debug(f"Request {req_id} completed, {len(pending_requests)} pending")
                                
                            # Handle errors
                            elif data.get("type") == "error":
                                error_msg = data.get("message", "Unknown error")
                                logger.error(f"Resemble WebSocket API error: {error_msg}")
                                
                                # Don't raise an error for punctuation-only inputs
                                if "would not generate any audio" in error_msg and data.get("request_id") in pending_requests:
                                    req_id = data.get("request_id")
                                    pending_requests.remove(req_id)
                                    logger.debug(f"Ignoring error for request {req_id}, {len(pending_requests)} pending")
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
                        logger.debug("Received end of input signal")
                        break
                    
                    if not text.strip():
                        logger.debug("Skipping empty text")
                        self._text_ch.task_done()
                        continue
                    
                    # Preprocess text before sending
                    text = self._preprocess_text(text)
                    
                    # Prepare request payload
                    logger.info(f"Synthesizing text: {text}")
                    self._mark_started()
                    
                    payload = {
                        "voice_uuid": self._opts.voice_uuid,
                        "data": text,
                        "request_id": self._request_id,
                        "binary_response": self._opts.binary_response,
                        "output_format": self._opts.output_format,
                        "sample_rate": self._opts.sample_rate,
                        "precision": self._opts.precision,
                        "no_audio_header": self._opts.no_audio_header,
                    }
                    
                    # Add request to pending set
                    pending_requests.add(self._request_id)
                    
                    # Send synthesis request
                    async with self._ws_lock:
                        await self._websocket.send(json.dumps(payload))
                    logger.debug(f"Sent request {self._request_id}")
                    self._request_id += 1
                    
                    # Mark the text as processed
                    self._text_ch.task_done()
                
                # Wait for all pending requests to complete
                if pending_requests:
                    logger.info(f"Waiting for {len(pending_requests)} pending audio responses")
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