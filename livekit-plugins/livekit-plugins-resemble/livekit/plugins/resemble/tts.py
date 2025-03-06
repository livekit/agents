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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, AsyncGenerator

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
        )

    def stream(
        self, *, conn_options: Optional[APIConnectOptions] = None
    ) -> "SynthesizeStream":
        """Create a streaming synthesis connection to Resemble AI."""
        return SynthesizeStream(
            tts=self,
            opts=self._opts,
            conn_options=conn_options,
            api_key=self._api_key,
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text into speech in one go using Resemble AI."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _Options,
        conn_options: Optional[APIConnectOptions] = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._segment_id = utils.shortuuid()

    async def _run(self) -> None:
        """Run the synthesis process."""
        request_id = 0
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        websocket = None
        decode_task = None
        
        try:
            # Connect to websocket
            websocket = await websockets.connect(
                RESEMBLE_WEBSOCKET_URL,
                extra_headers={"Authorization": f"Bearer {self._api_key}"},
                ping_interval=5,
                ping_timeout=10,
            )
            
            # Create request payload
            payload = {
                "voice_uuid": self._opts.voice_uuid,
                "project_uuid": self._opts.project_uuid,
                "data": self._input_text,
                "request_id": request_id,
                "binary_response": self._opts.binary_response,
                "output_format": self._opts.output_format,
                "sample_rate": self._opts.sample_rate,
                "precision": self._opts.precision,
                "no_audio_header": self._opts.no_audio_header,
            }
            
            # Send the synthesis request
            await websocket.send(json.dumps(payload))
            
            async def _decode_loop():
                try:
                    while True:
                        message = await websocket.recv()
                        
                        # Handle binary response
                        if self._opts.binary_response and isinstance(message, bytes):
                            decoder.push(message)
                            continue
                        
                        # Handle JSON response
                        try:
                            data = json.loads(message)
                            
                            if data.get("type") == "audio":
                                audio_data = base64.b64decode(data["audio_content"])
                                decoder.push(audio_data)
                            
                            elif data.get("type") == "audio_end":
                                # Stream is complete
                                break
                                
                            elif data.get("type") == "error":
                                error_msg = data.get("message", "Unknown error")
                                logger.error(f"Resemble API error: {error_msg}")
                                raise APIStatusError(
                                    message=error_msg,
                                    status_code=data.get("status_code", 500),
                                    request_id=str(request_id),
                                    body=None,
                                )
                        except json.JSONDecodeError:
                            logger.error("Failed to decode JSON response")
                finally:
                    decoder.end_input()
            
            # Start decode loop
            decode_task = asyncio.create_task(_decode_loop())
            
            # Create audio emitter
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=str(request_id),
                segment_id=self._segment_id,
            )
            
            # Emit audio frames as they're decoded
            async for frame in decoder:
                emitter.push(frame)
            
            # Flush any remaining audio
            emitter.flush()
            
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"Error during synthesis: {e}") from e
        finally:
            # Clean up resources
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            if websocket:
                await websocket.close()
            await decoder.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Resemble AI."""

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
        
        # Channels for communication between components
        self._text_ch = asyncio.Queue()
        self._audio_ch = asyncio.Queue()
        
        # Tasks
        self._websocket_task = None
        self._decode_task = None
        self._emit_task = None

    async def _run(self) -> None:
        """Run the streaming synthesis process."""
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        
        websocket = None
        
        try:
            # Connect to websocket
            websocket = await websockets.connect(
                RESEMBLE_WEBSOCKET_URL,
                extra_headers={"Authorization": f"Bearer {self._api_key}"},
                ping_interval=5,
                ping_timeout=10,
            )
            
            # Create tasks for processing
            self._websocket_task = asyncio.create_task(self._handle_websocket(websocket))
            self._decode_task = asyncio.create_task(self._decode_audio(decoder))
            self._emit_task = asyncio.create_task(self._emit_audio(decoder))
            
            # Wait for all tasks to complete
            done, pending = await asyncio.wait(
                [
                    self._websocket_task,
                    self._decode_task,
                    self._emit_task,
                ],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            
            # Handle any exceptions
            for task in done:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Error in streaming task: {e}")
                    raise
                    
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"Error during synthesis: {e}") from e
        finally:
            # Clean up resources
            for task in [self._websocket_task, self._decode_task, self._emit_task]:
                if task:
                    await utils.aio.gracefully_cancel(task)
            if websocket:
                await websocket.close()
            await decoder.aclose()

    async def _handle_websocket(self, websocket: websockets.WebSocketClientProtocol) -> None:
        """Handle the WebSocket connection for streaming synthesis."""
        try:
            while True:
                # Wait for text to synthesize
                text = await self._text_ch.get()
                
                # Create request payload
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
                
                # Send the synthesis request
                await websocket.send(json.dumps(payload))
                self._request_id += 1
                
                # Process responses until we get audio_end
                while True:
                    message = await websocket.recv()
                    
                    # Put message in audio channel for processing
                    await self._audio_ch.put(message)
                    
                    # Check if this is a JSON message indicating end of stream
                    if not isinstance(message, bytes):
                        try:
                            data = json.loads(message)
                            if data.get("type") == "audio_end":
                                # This request is complete
                                break
                        except json.JSONDecodeError:
                            pass
                
                # Mark this text as processed
                self._text_ch.task_done()
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            return
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            raise

    async def _decode_audio(self, decoder: utils.codecs.AudioStreamDecoder) -> None:
        """Decode audio data from the websocket."""
        try:
            while True:
                # Get next message from audio channel
                message = await self._audio_ch.get()
                
                # Handle binary response
                if self._opts.binary_response and isinstance(message, bytes):
                    decoder.push(message)
                    self._audio_ch.task_done()
                    continue
                
                # Handle JSON response
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "audio":
                        audio_data = base64.b64decode(data["audio_content"])
                        decoder.push(audio_data)
                    
                    elif data.get("type") == "audio_end":
                        # End of this request, but don't end the decoder input
                        # as more requests may come
                        pass
                        
                    elif data.get("type") == "error":
                        error_msg = data.get("message", "Unknown error")
                        logger.error(f"Resemble API error: {error_msg}")
                        raise APIStatusError(
                            message=error_msg,
                            status_code=data.get("status_code", 500),
                            request_id=str(data.get("request_id", 0)),
                            body=None,
                        )
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON response")
                
                self._audio_ch.task_done()
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            return
        except Exception as e:
            logger.error(f"Audio decoder error: {e}")
            raise
        finally:
            decoder.end_input()

    async def _emit_audio(self, decoder: utils.codecs.AudioStreamDecoder) -> None:
        """Emit decoded audio frames."""
        emitter = None
        current_request_id = None
        
        try:
            async for frame in decoder:
                # Create a new emitter for each request_id
                if current_request_id != self._request_id:
                    current_request_id = self._request_id
                    emitter = tts.SynthesizedAudioEmitter(
                        event_ch=self._event_ch,
                        request_id=str(current_request_id),
                        segment_id=utils.shortuuid(),
                    )
                
                if emitter:
                    emitter.push(frame)
                    
            # Flush any remaining audio
            if emitter:
                emitter.flush()
                
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            return
        except Exception as e:
            logger.error(f"Audio emitter error: {e}")
            raise

    async def synthesize_text(self, text: str) -> None:
        """Synthesize text into speech using streaming API."""
        await self._text_ch.put(text) 