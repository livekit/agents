# livekit/plugins/minimax/tts.py (WebSocket version)

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
import asyncio, base64, json, os, weakref, ssl
from dataclasses import dataclass, replace

from typing import List, Dict, Any
import websockets
# <--- Using websockets for WebSocket connections
from livekit.agents import (
    APIConnectionError, APIStatusError, APITimeoutError, tts, utils, tokenize,
    APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
)
# Removed http_context as we're using WebSocket connections now
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import (
    SUPPORTED_VOICES, TTSDefaultBitRates, TTSBitRates, TTSSampleRates, TTSDefaultSampleRates, TTSDefaultVoiceId,
    TTSModels, TTSVoices, TTSSubtitleType, TTSDefaultEmotion, TTSEmotion, TTSLanguages, TTSDefaultLanguage
)

MINIMAX_API_BASE_URL = "https://api-uw.minimax.io"


@dataclass
class _TTSOptions:
    # This internal class holds all the configuration options for a TTS request.
    api_key: str
    group_id: str
    base_url: str
    model: TTSModels
    voice_id: TTSVoices
    sample_rate: TTSSampleRates
    bitrate: TTSBitRates
    emotion: TTSEmotion
    speed: float
    vol: float
    pitch: int
    subtitle_enable: bool
    subtitle_type: TTSSubtitleType
    language: TTSLanguages
    pronunciation_dict: Dict[str, List[str]] | None
    # voice_modify
    intensity: int | None
    timbre: int | None
    sound_effects: str | None

    def get_http_url(self, path: str) -> str:
        """Constructs the full API URL for a given path."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class TTS(tts.TTS):
    def __init__(
            self, *,
            model: TTSModels = "speech-02-turbo",
            voice_id: TTSVoices = TTSDefaultVoiceId,
            sample_rate: TTSSampleRates = TTSDefaultSampleRates,
            bitrate: TTSBitRates = TTSDefaultBitRates,
            emotion: TTSEmotion = TTSDefaultEmotion,
            speed: float = 1.0,
            vol: float = 1.0,
            pitch: int = 0,
            subtitle_enable: bool = True,
            subtitle_type: TTSSubtitleType = "word",
            api_key: str | None = None,
            group_id: str | None = None,
            base_url: str = MINIMAX_API_BASE_URL,
            language: TTSLanguages = TTSDefaultLanguage,
            websocket = None,  # WebSocket connection
            # --- **Advanced para** ---
            pronunciation_dict: Dict[str, List[str]] | None = None,
            intensity: int | None = None,
            timbre: int | None = None,
            sound_effects: str | None = None,
    ):
        enable_aligned_transcript = subtitle_enable and subtitle_type == "word"
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=enable_aligned_transcript),
            sample_rate=sample_rate, num_channels=1)

        # Parameter validation
        minimax_api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        minimax_group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        if not minimax_api_key: raise ValueError("MINIMAX_API_KEY must be set")
        if not minimax_group_id: raise ValueError("MINIMAX_GROUP_ID must be set")
        if not (0.5 <= speed <= 2.0): raise ValueError(f"speed must be between 0.5 and 2.0, but got {speed}")
        if voice_id not in SUPPORTED_VOICES: logger.warning("Voice '%s' is not officially supported.", voice_id)

        if intensity is not None and not (-100 <= intensity <= 100):
            raise ValueError(f"intensity must be between -100 and 100, but got {intensity}")
        if timbre is not None and not (-100 <= timbre <= 100):
            raise ValueError(f"timbre must be between -100 and 100, but got {timbre}")

        supported_effects = ["spacious_echo", "auditorium_echo", "lofi_telephone", "robotic"]
        if sound_effects is not None and sound_effects not in supported_effects:
            raise ValueError(f"sound_effects must be one of {supported_effects}, but got {sound_effects}")

        if subtitle_type == "sentence":
            # We print a warning instead of raising an error to provide more flexibility.
            logger.warning(
                "Minimax streaming TTS does not support 'sentence' level subtitles. "
                "This option will be ignored in streaming mode and may cause errors. "
                "Please use 'word' for streaming timestamps."
            )

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice_id,
            api_key=minimax_api_key,
            group_id=minimax_group_id,
            base_url=base_url,
            sample_rate=sample_rate,
            emotion=emotion,
            bitrate=bitrate,
            speed=speed,
            pitch=pitch,
            vol=vol,
            subtitle_enable=subtitle_enable,
            subtitle_type=subtitle_type,
            language=language,
            timbre=timbre,
            pronunciation_dict=pronunciation_dict,
            sound_effects=sound_effects,
            intensity=intensity
        )

        self._websocket = websocket
        self._streams = weakref.WeakSet["SynthesizeStream"]()
        
        # Add connection pool like Deepgram for better performance
        self._pool = utils.ConnectionPool(
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self, timeout: float):
        """Create a new WebSocket connection for the pool"""
        url = "wss://api.minimax.io/ws/v1/t2a_v2"
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, ssl=ssl_context),
            timeout
        )
        
        # Wait for connection confirmation
        response = json.loads(await ws.recv())
        if response.get("event") != "connected_success":
            await ws.close()
            raise APIConnectionError("Failed to establish WebSocket connection")
        
        # Send task_start message immediately after connection
        start_msg = {
            "event": "task_start",
            "model": self._opts.model,
            "voice_setting": {
                "voice_id": self._opts.voice_id,
                "speed": self._opts.speed,
                "vol": self._opts.vol,
                "pitch": self._opts.pitch
            },
            "audio_setting": {
                "sample_rate": self._opts.sample_rate,
                "bitrate": self._opts.bitrate,
                "format": "pcm",
                "channel": 1
            }
        }
        
        if self._opts.emotion is not None:
            start_msg["voice_setting"]["emotion"] = self._opts.emotion
            
        if self._opts.subtitle_enable:
            start_msg["subtitle_enable"] = True
            start_msg["subtitle_type"] = self._opts.subtitle_type
        
        voice_modify = {}
        if self._opts.intensity is not None:
            voice_modify['intensity'] = self._opts.intensity
        if self._opts.timbre is not None:
            voice_modify['timbre'] = self._opts.timbre
        if self._opts.sound_effects is not None:
            voice_modify['sound_effects'] = self._opts.sound_effects
        
        if voice_modify:
            start_msg['voice_modify'] = voice_modify
        
        # Send task_start and wait for confirmation
        await ws.send(json.dumps(start_msg))
        response = json.loads(await ws.recv())
        if response.get("event") != "task_started":
            await ws.close()
            raise APIConnectionError("Failed to start TTS task")
        
        return ws

    async def _close_ws(self, ws) -> None:
        """Close a WebSocket connection from the pool"""
        try:
            await ws.send(json.dumps({"event": "task_finish"}))
        except:
            pass  # Connection might already be closed
        await ws.close()

    async def _ensure_websocket(self):
        """Get or create a shared WebSocket connection and start the TTS task."""
        if not self._websocket:
            url = "wss://api.minimax.io/ws/v1/t2a_v2"
            headers = {"Authorization": f"Bearer {self._opts.api_key}"}
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            try:
                self._websocket = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
                response = json.loads(await self._websocket.recv())
                if response.get("event") != "connected_success":
                    raise APIConnectionError("Failed to establish WebSocket connection")
                
                # Send task_start message immediately after connection
                start_msg = {
                    "event": "task_start",
                    "model": self._opts.model,
                    "voice_setting": {
                        "voice_id": self._opts.voice_id,
                        "speed": self._opts.speed,
                        "vol": self._opts.vol,
                        "pitch": self._opts.pitch
                    },
                    "audio_setting": {
                        "sample_rate": self._opts.sample_rate,
                        "bitrate": self._opts.bitrate,
                        "format": "pcm",
                        "channel": 1
                    }
                }
                
                if self._opts.emotion is not None:
                    start_msg["voice_setting"]["emotion"] = self._opts.emotion
                    
                if self._opts.subtitle_enable:
                    start_msg["subtitle_enable"] = True
                    start_msg["subtitle_type"] = self._opts.subtitle_type
                
                voice_modify = {}
                if self._opts.intensity is not None:
                    voice_modify['intensity'] = self._opts.intensity
                if self._opts.timbre is not None:
                    voice_modify['timbre'] = self._opts.timbre
                if self._opts.sound_effects is not None:
                    voice_modify['sound_effects'] = self._opts.sound_effects
                
                if voice_modify:
                    start_msg['voice_modify'] = voice_modify
                
                # Send task_start and wait for confirmation
                await self._websocket.send(json.dumps(start_msg))
                response = json.loads(await self._websocket.recv())
                if response.get("event") != "task_started":
                    raise APIConnectionError("Failed to start TTS task")
                    
            except Exception as e:
                raise APIConnectionError(f"WebSocket connection failed: {e}") from e
        return self._websocket

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "SynthesizeStream":
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Pre-warm connection pool to reduce first response latency"""
        self._pool.prewarm()

    async def aclose(self) -> None:
        await asyncio.gather(*(s.aclose() for s in self._streams))
        self._streams.clear()
        
        # Close connection pool
        await self._pool.aclose()
        
        # Clean up old websocket if still around
        if self._websocket:
            try:
                # Send task_finish before closing the connection
                await self._websocket.send(json.dumps({"event": "task_finish"}))
            except:
                pass  # Connection might already be closed
            await self._websocket.close()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

        # Use a channel to decouple sentence production (from the tokenizer)
        # from sentence consumption (by the API consumer task).
        self._sentences_ch = utils.aio.Chan[str]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Start producer-consumer tasks to handle text-to-audio conversion."""
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into sentences and sends them into _sentences_ch
            # Similar to Deepgram's _tokenize_input but for sentences instead of words
            sentence_tokenizer = tokenize.basic.SentenceTokenizer()
            sentence_stream = sentence_tokenizer.stream()
            
            async for input in self._input_ch:
                if isinstance(input, str):
                    sentence_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    sentence_stream.flush()
                    # Process any pending sentences
                    async for sentence in sentence_stream:
                        self._sentences_ch.send_nowait(sentence.token)
                    
            # Process final sentences
            sentence_stream.end_input()
            async for sentence in sentence_stream:
                self._sentences_ch.send_nowait(sentence.token)
                
            self._sentences_ch.close()

        async def _run_sentences() -> None:
            async for sentence in self._sentences_ch:
                await self._run_sentence(sentence, output_emitter)

        # Run the tokenizer and sentence processing tasks concurrently.
        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_sentences()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception:
            logger.exception("synthesize stream failed")
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_sentence(self, sentence: str, output_emitter: tts.AudioEmitter) -> None:
        """Process a single sentence through the Minimax API, similar to Deepgram's _run_ws"""
        if not sentence.strip():
            return

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        self._mark_started()

        # Use connection pool like Deepgram for better performance
        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            try:
                # Send task_continue with sentence text (task_start handled in _connect_ws)
                await ws.send(json.dumps({
                    "event": "task_continue",
                    "text": sentence
                }))
                
                # Collect audio chunks
                while True:
                    response = json.loads(await ws.recv())
                    
                    if "data" in response and "audio" in response["data"]:
                        audio_hex = response["data"]["audio"]
                        output_emitter.push(bytes.fromhex(audio_hex))
                        
                        # Handle subtitles if present
                        if "subtitle" in response["data"]:
                            subtitle_obj = response["data"]["subtitle"]
                            if subtitle_obj and isinstance(subtitle_obj, dict):
                                word_timestamps = subtitle_obj.get("timestamped_words")
                                if word_timestamps and isinstance(word_timestamps, list):
                                    for word_data in word_timestamps:
                                        start_time_sec = word_data['time_begin'] / 1000.0
                                        end_time_sec = word_data['time_end'] / 1000.0
                                        output_emitter.push_timed_transcript(
                                            TimedString(
                                                text=word_data['word'],
                                                start_time=start_time_sec,
                                                end_time=end_time_sec
                                            )
                                        )
                    
                    if response.get("is_final"):
                        break
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed: {e}")
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
            except Exception as e:
                logger.error(f"API call error for sentence '{sentence}': {e}")

        output_emitter.end_segment()

    async def aclose(self) -> None:
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False  # This is a non-streaming synthesis
        )

        if not self._input_text.strip():
            return

        # ChunkedStream will use the same WebSocket connection but with different flow
        ws = await self._tts._ensure_websocket()
        try:
            # Send the entire text at once for chunked stream (task_start handled in _ensure_websocket)
            await ws.send(json.dumps({
                "event": "task_continue",
                "text": self._input_text
            }))
            
            # Collect all audio chunks
            all_audio = ""
            while True:
                response = json.loads(await ws.recv())
                
                if "data" in response and "audio" in response["data"]:
                    audio_hex = response["data"]["audio"]
                    all_audio += audio_hex
                    
                    # Handle subtitles if present
                    if "subtitle" in response["data"]:
                        subtitle_obj = response["data"]["subtitle"]
                        if subtitle_obj and isinstance(subtitle_obj, dict):
                            word_timestamps = subtitle_obj.get("timestamped_words")
                            if word_timestamps and isinstance(word_timestamps, list):
                                for word_data in word_timestamps:
                                    start_time_sec = word_data['time_begin'] / 1000.0
                                    end_time_sec = word_data['time_end'] / 1000.0
                                    output_emitter.push_timed_transcript(
                                        TimedString(
                                            text=word_data['word'],
                                            start_time=start_time_sec,
                                            end_time=end_time_sec
                                        )
                                    )
                
                if response.get("is_final"):
                    break
                    
            # Push all audio at once
            if all_audio:
                output_emitter.push(bytes.fromhex(all_audio))
                
            # Send task_finish message
            await ws.send(json.dumps({"event": "task_finish"}))
            output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"WebSocket connection closed: {e}") from e
        except websockets.exceptions.WebSocketException as e:
            raise APIConnectionError(f"WebSocket error: {e}") from e
        except Exception as e:
            raise APIConnectionError() from e