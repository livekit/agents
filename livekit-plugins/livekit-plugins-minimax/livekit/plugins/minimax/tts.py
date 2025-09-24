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
import asyncio, base64, json, os, weakref, ssl, re
from dataclasses import dataclass, replace
from typing import List, Dict, Any
import websockets
from livekit.agents import (
    APIConnectionError, APIStatusError, APITimeoutError, tts, utils, tokenize,
    APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
)
from livekit.agents.voice.io import TimedString
from livekit.agents import tokenize

from .log import logger
from .models import (
    SUPPORTED_VOICES, TTSDefaultBitRates, TTSBitRates, TTSSampleRates, TTSDefaultSampleRates, TTSDefaultVoiceId,
    TTSModels, TTSVoices, TTSSubtitleType, TTSDefaultEmotion, TTSEmotion, TTSLanguages, TTSDefaultLanguage
)

MINIMAX_API_BASE_URL = "https://api-uw.minimax.io"

# ==============================
#  Smart Text Tokenizer for Multilingual TTS
# ==============================

# Define punctuation and symbol sets for different languages
CHINESE_PUNCTS = "。！？；，："
ENGLISH_PUNCTS = ".!?;,:"
OTHER_SYMBOLS = "＃＄％＆（）＊＋，－／＜＝＞［＼］＾＿｀｛｜｝○~@#$^&*()[]{}"


def char_length(c):
    """
    Estimate character length for TTS/audio timing purposes.

    Different character types have different audio durations:
    - Chinese characters: 3 units (longer pronunciation)
    - Japanese kana: 2 units
    - Korean Hangul: 3 units
    - English letters: 1 unit
    - Numbers: 3 units (spelled out)
    - Punctuation: 4-6 units (pauses)
    """
    if re.match(r'\s', c):
        return 1
    elif re.match(u'[\u4e00-\u9fff]', c):  # Chinese characters
        return 3
    elif re.match(u'[\u3040-\u309F\u30A0-\u30FF]', c):  # Japanese kana
        return 2
    elif re.match(u'[\uAC00-\uD7AF]', c):  # Korean Hangul
        return 3
    elif re.match(r'[0-9]', c):
        return 3
    elif re.match(r'[a-zA-Z]', c):
        return 1
    elif c in CHINESE_PUNCTS:
        return 6  # Longer pause for Chinese punctuation
    elif c in ENGLISH_PUNCTS:
        return 4  # Standard pause for English punctuation
    elif c in OTHER_SYMBOLS:
        return 2
    else:
        return 1


def split_text(text, split_length):
    """
    Intelligently split mixed-language text into TTS-friendly chunks.

    This tokenizer:
    - Protects quoted content ("..." and "...")
    - Preserves English words, numbers, and Korean blocks as units
    - Splits by punctuation first, then by length if needed
    - Merges small chunks for efficiency

    Args:
        text: Input text to split
        split_length: Target chunk size (in Chinese character equivalents)

    Returns:
        List of text chunks ready for TTS processing
    """
    target_len = split_length * 3  # Base unit: Chinese char = 3

    # Step 1: Protect quoted segments from being split
    quote_pattern = re.compile(r'".*?"|".*?"')
    quoted_segments = []
    last_idx = 0

    for m in quote_pattern.finditer(text):
        if m.start() > last_idx:
            quoted_segments.append((text[last_idx:m.start()], False))
        quoted_segments.append((m.group(), True))  # Mark as quoted
        last_idx = m.end()

    if last_idx < len(text):
        quoted_segments.append((text[last_idx:], False))

    # Step 2: Split non-quoted segments by punctuation
    segments = []
    for seg, is_quote in quoted_segments:
        if is_quote:
            segments.append(seg)  # Keep quoted text intact
            continue

        # Split by major punctuation marks
        sub_segs = re.split(r'([。！？；.!?\n;])', seg)
        temp = []
        i = 0
        while i < len(sub_segs):
            if i + 1 < len(sub_segs):
                # Combine text with following punctuation
                temp.append(sub_segs[i] + sub_segs[i + 1])
                i += 2
            else:
                temp.append(sub_segs[i])
                i += 1

        # Further split by minor punctuation
        minor_temp = []
        for s in temp:
            minor_temp.extend(re.split(r'([，,：:；;])', s))
        segments.extend([s for s in minor_temp if s])

    # Step 3: Force split by length while protecting word boundaries
    final_chunks = []
    for seg in segments:
        seg_len = sum(char_length(c) for c in seg)
        if seg_len <= target_len:
            final_chunks.append(seg)
            continue

        current = ""
        current_len = 0
        idx = 0

        while idx < len(seg):
            c = seg[idx]

            # Protect English words and numbers as atomic units
            if re.match(r'[a-zA-Z0-9]', c):
                m = re.match(r'[a-zA-Z0-9]+', seg[idx:])
                token = m.group(0)
                token_len = sum(char_length(ch) for ch in token)

                if current_len + token_len > target_len and current:
                    final_chunks.append(current)
                    current = ""
                    current_len = 0

                current += token
                current_len += token_len
                idx += len(token)
                continue

            # Protect Korean Hangul blocks as atomic units
            if re.match(r'[\uAC00-\uD7AF]', c):
                m = re.match(r'[\uAC00-\uD7AF]+', seg[idx:])
                token = m.group(0)
                token_len = sum(char_length(ch) for ch in token)

                if current_len + token_len > target_len and current:
                    final_chunks.append(current)
                    current = ""
                    current_len = 0

                current += token
                current_len += token_len
                idx += len(token)
                continue

            # Handle spaces intelligently for word-based languages
            if c.isspace():
                # Break at space if close to target length
                if current_len >= target_len * 0.9:
                    final_chunks.append(current)
                    current = ""
                    current_len = 0
                else:
                    current += c
                    current_len += char_length(c)
                idx += 1
                continue

            # Handle other single characters
            cl = char_length(c)
            if current_len + cl > target_len and current:
                final_chunks.append(current)
                current = ""
                current_len = 0

            current += c
            current_len += cl
            idx += 1

        if current:
            final_chunks.append(current)

    # Step 4: Merge small chunks for efficiency
    merged_chunks = []
    buf = ""
    buf_len = 0

    for c in final_chunks:
        l = sum(char_length(ch) for ch in c)
        if buf_len + l <= target_len:
            buf += c
            buf_len += l
        else:
            if buf:
                merged_chunks.append(buf)
            buf = c
            buf_len = l

    if buf:
        merged_chunks.append(buf)

    return merged_chunks


@dataclass
class _TTSOptions:
    """Internal configuration class for TTS request parameters."""
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
    text_normalization: bool

    def get_http_url(self, path: str) -> str:
        """Construct the full API URL for a given path."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class TTS(tts.TTS):
    """
    Minimax TTS implementation using WebSocket connections.

    Features:
    - WebSocket-based streaming synthesis
    - Connection pooling for better performance
    - Smart text tokenization for multilingual content
    - Support for voice emotions and settings
    - Word-level timing information
    """

    def __init__(self, *, model: TTSModels = "speech-02-turbo", voice_id: TTSVoices = TTSDefaultVoiceId,
                 sample_rate: TTSSampleRates = TTSDefaultSampleRates, bitrate: TTSBitRates = TTSDefaultBitRates,
                 emotion: TTSEmotion | None = None, speed: float = 1.0, vol: float = 1.0, pitch: int = 0,
                 subtitle_enable: bool = True, subtitle_type: TTSSubtitleType = "word",
                 api_key: str | None = None, group_id: str | None = None, base_url: str = MINIMAX_API_BASE_URL,
                 language: TTSLanguages = TTSDefaultLanguage, text_normalization: bool = True,
                 pronunciation_dict: Dict[str, List[str]] | None = None):

        enable_aligned_transcript = subtitle_enable and (subtitle_type == "word" or subtitle_type == "sentence")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=enable_aligned_transcript),
            sample_rate=sample_rate, num_channels=1)

        # Parameter validation
        minimax_api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        minimax_group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        if not minimax_api_key:
            raise ValueError("MINIMAX_API_KEY must be set")
        if not minimax_group_id:
            raise ValueError("MINIMAX_GROUP_ID must be set")
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"speed must be between 0.5 and 2.0, but got {speed}")

        self._opts = _TTSOptions(
            model=model, voice_id=voice_id, api_key=minimax_api_key, group_id=minimax_group_id,
            base_url=base_url, sample_rate=sample_rate, emotion=emotion, bitrate=bitrate,
            speed=speed, pitch=pitch, vol=vol, subtitle_enable=subtitle_enable,
            subtitle_type=subtitle_type, language=language, pronunciation_dict=pronunciation_dict,
            text_normalization=text_normalization,
        )

        # Track active streams for cleanup
        self._streams = weakref.WeakSet[SynthesizeStream]()

        # Connection pool for efficient WebSocket reuse
        self._pool = utils.ConnectionPool(
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour max session
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self, timeout: float):
        """
        Create and configure a new WebSocket connection.

        Returns a ready-to-use WebSocket connection with TTS task initialized.
        """
        url = "wss://api-uw.minimax.io/ws/v1/t2a_v2"
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}

        # Create SSL context (disable verification for development)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Establish WebSocket connection
        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, ssl=ssl_context),
            timeout
        )

        # Wait for connection confirmation
        response = json.loads(await ws.recv())
        logger.warning(f"WebSocket connected with trace_id: {response['trace_id']}")

        if response.get("event") != "connected_success":
            await ws.close()
            raise APIConnectionError("Failed to establish WebSocket connection")

        # Initialize TTS task with voice and audio settings
        start_msg = {
            "event": "task_start",
            "model": self._opts.model,
            "voice_setting": {
                "voice_id": self._opts.voice_id,
                "speed": self._opts.speed,
                "vol": self._opts.vol,
                "pitch": self._opts.pitch,
                "emotion": self._opts.emotion,
                "text_normalization": self._opts.text_normalization,
                "pronunciation_dict": self._opts.pronunciation_dict,
            },
            "audio_setting": {
                "sample_rate": self._opts.sample_rate,
                "bitrate": self._opts.bitrate,
                "format": "pcm",
                "channel": 1
            }
        }

        # Add optional settings
        if self._opts.emotion is not None:
            start_msg["voice_setting"]["emotion"] = self._opts.emotion
        if self._opts.subtitle_enable:
            start_msg["subtitle_enable"] = True
            start_msg["subtitle_type"] = self._opts.subtitle_type

        # Send task initialization and wait for confirmation
        await ws.send(json.dumps(start_msg))
        response = json.loads(await ws.recv())

        if response.get("event") != "task_started":
            await ws.close()
            raise APIConnectionError(f"Failed to start TTS task, trace_id: {response.get('trace_id')}")

        return ws

    async def _close_ws(self, ws) -> None:
        """Properly close a WebSocket connection."""
        try:
            # Send task finish signal
            await ws.send(json.dumps({"event": "task_finish"}))
        except:
            pass  # Connection might already be closed
        await ws.close()

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        """Create a non-streaming synthesis request."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> SynthesizeStream:
        """Create a streaming synthesis session."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Pre-warm connection pool to reduce first request latency."""
        self._pool.prewarm()

    async def aclose(self) -> None:
        """Clean up all resources."""
        # Close all active streams
        await asyncio.gather(*(s.aclose() for s in self._streams))
        self._streams.clear()
        # Close connection pool
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """
    Streaming TTS synthesis with intelligent text tokenization.

    This class handles:
    - Asynchronous text input via push_text()
    - Smart multilingual text segmentation
    - Real-time audio streaming
    - Word-level timing information
    """

    class _FlushSentinel:
        """Marker to indicate end of text input."""
        pass

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

        # Custom input handling (bypasses base class input channel)
        self._input_ch: asyncio.Queue[Any] = asyncio.Queue()
        self._accumulated_text = ""
        self._input_ended = False
        self._closed = False

    async def push_text(self, text: str):
        """
        Asynchronously push text for synthesis.

        Text will be accumulated and processed when end_input() is called.
        """
        print(f">>> Pushing text: {text}")
        await self._input_ch.put(text)

    async def end_input(self):
        """Signal that no more text will be pushed."""
        print(">>> Ending text input")
        await self._input_ch.put(self._FlushSentinel())

    async def _run(self, output_emitter: tts.AudioEmitter):
        """
        Core synthesis logic with smart tokenization.

        Process:
        1. Collect all input text
        2. Apply intelligent tokenization
        3. Synthesize each segment
        4. Stream audio output
        """
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        print(">>> Starting synthesis stream")

        try:
            # Step 1: Collect all input text
            while True:
                item = await self._input_ch.get()
                print(f">>> Received input: {item}")

                if isinstance(item, self._FlushSentinel):
                    print(">>> Input collection complete")
                    self._input_ended = True
                    break
                elif isinstance(item, str):
                    self._accumulated_text += item
                    print(f">>> Accumulated text: '{self._accumulated_text}'")

            # Step 2: Apply smart tokenization
            if self._accumulated_text.strip():
                print(f">>> Processing text: '{self._accumulated_text}'")

                # Use intelligent tokenizer for multilingual content
                sentences = split_text(self._accumulated_text, 15)  # 15 Chinese chars per chunk
                sentences = [s.strip() for s in sentences if s.strip()]
                print(f">>> Smart tokenizer produced {len(sentences)} segments: {sentences}")

                # Step 3: Synthesize each segment
                for i, sentence in enumerate(sentences):
                    print(f">>> Processing segment {i + 1}/{len(sentences)}: '{sentence}'")
                    if sentence.strip():
                        await self._synthesize_sentence(sentence, output_emitter)
                        print(f">>> Completed segment {i + 1}")
            else:
                print(">>> No text to process")

        except Exception as e:
            print(f">>> Synthesis error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(">>> Finalizing synthesis stream")
            output_emitter.flush()
            print(">>> Audio buffer flushed")
            output_emitter.end_input()
            print(">>> Stream processing completed")

    async def _synthesize_sentence(self, sentence: str, output_emitter: tts.AudioEmitter):
        """
        Synthesize a single text segment using WebSocket connection.

        Handles:
        - WebSocket communication
        - Audio data streaming
        - Word timing information
        - Error recovery
        """
        if not sentence.strip():
            print(">>> Skipping empty sentence")
            return

        print(f">>> Synthesizing: '{sentence}'")
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        self._mark_started()  # Notify base class that synthesis has started

        try:
            # Get WebSocket connection from pool
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                print(f">>> Sending to WebSocket: '{sentence}'")

                # Send synthesis request
                await ws.send(json.dumps({
                    "event": "task_continue",
                    "text": sentence
                }))

                audio_chunks_received = 0

                # Process WebSocket responses
                while True:
                    try:
                        response = json.loads(await ws.recv())

                        # Handle audio data
                        if "data" in response and "audio" in response["data"]:
                            audio_hex = response["data"]["audio"]
                            if audio_hex:
                                audio_data = bytes.fromhex(audio_hex)
                                output_emitter.push(audio_data)
                                audio_chunks_received += 1

                            # Handle word timing information
                            subtitle_obj = response["data"].get("subtitle")
                            if subtitle_obj and isinstance(subtitle_obj, dict):
                                for word_data in subtitle_obj.get("timestamped_words", []):
                                    output_emitter.push_timed_transcript(
                                        TimedString(
                                            text=word_data["word"],
                                            start_time=word_data["time_begin"] / 1000.0,
                                            end_time=word_data["time_end"] / 1000.0,
                                        )
                                    )

                        # Check for completion
                        if response.get("is_final") or response.get("task_finished"):
                            print(f">>> Sentence complete: {audio_chunks_received} audio chunks received")
                            break

                    except Exception as e:
                        print(f">>> WebSocket response error: {e}")
                        break

        except Exception as e:
            print(f">>> Synthesis error for '{sentence}': {e}")
            import traceback
            traceback.print_exc()
        finally:
            output_emitter.end_segment()
            print(f">>> Segment completed: '{sentence}'")

    async def aclose(self):
        """Clean up stream resources."""
        print(">>> Closing synthesis stream")
        if not self._closed:
            self._closed = True
            # Clear any remaining items in input queue
            try:
                while not self._input_ch.empty():
                    self._input_ch.get_nowait()
            except asyncio.QueueEmpty:
                pass


class ChunkedStream(tts.ChunkedStream):
    """
    Non-streaming TTS synthesis for complete text input.

    Synthesizes entire text at once and returns combined audio.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Synthesize complete text and emit as single audio stream."""
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False  # Non-streaming mode
        )

        if not self._input_text.strip():
            return

        # Use connection pool for WebSocket
        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            try:
                # Send complete text for synthesis
                await ws.send(json.dumps({
                    "event": "task_continue",
                    "text": self._input_text
                }))

                # Collect all audio data
                all_audio_hex = ""
                while True:
                    response = json.loads(await ws.recv())

                    if "data" in response and "audio" in response["data"]:
                        all_audio_hex += response["data"]["audio"]

                    if response.get("task_finished") or response.get("is_final"):
                        logger.warning("Synthesis completed")
                        break

                # Emit complete audio as single chunk
                if all_audio_hex:
                    output_emitter.push(bytes.fromhex(all_audio_hex))

                output_emitter.flush()

            except websockets.exceptions.ConnectionClosed as e:
                raise APIConnectionError(f"WebSocket connection closed unexpectedly: {e}") from e
            except Exception as e:
                logger.error(f"ChunkedStream synthesis error: {e}")
                raise APIConnectionError from e