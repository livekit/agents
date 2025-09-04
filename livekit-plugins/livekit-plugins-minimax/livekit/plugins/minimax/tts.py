# livekit/plugins/minimax/tts.py (Final aiohttp version)

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
import asyncio, base64, json, os, weakref
from dataclasses import dataclass, replace

from typing import List, Dict, Any 
import aiohttp  # <--- Using aiohttp for asynchronous HTTP requests
from livekit.agents import (
    APIConnectionError, APIStatusError, APITimeoutError, tts, utils, tokenize,
    APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
)
from livekit.agents.utils import http_context  # <--- Using the correct http_context for session management
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
            http_session: aiohttp.ClientSession | None = None,  # <--- Type hint for aiohttp.ClientSession
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
        )

        self._session = http_session
        self._streams = weakref.WeakSet["SynthesizeStream"]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session instance."""
        if not self._session:
            # Get the session from the global http_context to reuse connections.
            self._session = http_context.http_session()
        return self._session

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "SynthesizeStream":
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(*(s.aclose() for s in self._streams))
        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

        # Create a sentence tokenizer stream to split incoming text into sentences.
        tokenizer = tokenize.basic.SentenceTokenizer()
        self._sentence_stream = tokenizer.stream()

        # Use a queue to decouple sentence production (from the tokenizer)
        # from sentence consumption (by the API consumer task).
        self._sentence_queue = asyncio.Queue[str | None]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Start producer-consumer tasks to handle text-to-audio conversion."""
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        # Run the producer and consumer tasks concurrently.
        producer_task = asyncio.create_task(self._producer_task())
        consumer_task = asyncio.create_task(self._consumer_task(output_emitter))

        try:
            await asyncio.gather(producer_task, consumer_task)
        except Exception:
            logger.exception("synthesize stream failed")
        finally:
            for task in [producer_task, consumer_task]:
                if not task.done():
                    task.cancel()
            # Ensure the sentence stream is closed.
            await self._sentence_stream.aclose()

    async def _producer_task(self):
        """
        Producer: Reads sentences from the tokenizer stream and puts them into the queue.
        """
        async for sentence in self._sentence_stream:
            await self._sentence_queue.put(sentence.token)
        # When the tokenizer stream ends, put a termination signal (None) into the queue.
        await self._sentence_queue.put(None)

    async def _consumer_task(self, output_emitter: tts.AudioEmitter):
        """
        Consumer: Takes sentences from the queue and calls the API for each sentence.
        """
        client = self._tts._ensure_session()
        self._mark_started()

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        while True:
            # Get a sentence from the queue.
            sentence = await self._sentence_queue.get()
            if sentence is None:
                # Received termination signal, break the loop.
                break

            if not sentence.strip():
                self._sentence_queue.task_done()
                continue

            try:
                # This logic handles the API call and SSE (Server-Sent Events) parsing.
                url = self._opts.get_http_url("v1/t2a_v2")
                print("URL:", url)  # For debugging purposes
                params = {"GroupId": self._opts.group_id}
                headers = {"Authorization": f"Bearer {self._opts.api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self._opts.model,
                    "text": sentence,
                    "stream": True,
                    "language_boost": self._opts.language,
                    "voice_setting": {
                        "voice_id": self._opts.voice_id,
                        "speed": self._opts.speed,
                        "vol": self._opts.vol,
                        "pitch": self._opts.pitch
                    },
                    "audio_setting": {
                        "format": "pcm",
                        "sample_rate": self._opts.sample_rate,
                        "bitrate": self._opts.bitrate
                    }
                }

                if self._opts.pronunciation_dict:
                    # API form "pronunciation_dict": {"tone": [...]}
                    # be make sure "tone" is eixst
                    if "tone" in self._opts.pronunciation_dict:
                        payload['pronunciation_dict'] = self._opts.pronunciation_dict

                if self._opts.emotion is not None:
                    print("emotion", self._opts.emotion)
                    payload["voice_setting"]["emotion"] = self._opts.emotion

                if self._opts.subtitle_enable:
                    payload['subtitle_enable'] = True
                    payload['subtitle_type'] = self._opts.subtitle_type

                voice_modify = {}
                if self._opts.intensity is not None:
                    voice_modify['intensity'] = self._opts.intensity
                if self._opts.timbre is not None:
                    voice_modify['timbre'] = self._opts.timbre
                if self._opts.sound_effects is not None:
                    voice_modify['sound_effects'] = self._opts.sound_effects


                if voice_modify:
                    payload['voice_modify'] = voice_modify

                async with client.post(url, headers=headers, params=params, json=payload, timeout=20) as resp:
                    resp.raise_for_status()
                    buffer = b""
                    async for chunk in resp.content.iter_any():
                        buffer += chunk
                        while b"\n" in buffer:
                            line_bytes, buffer = buffer.split(b"\n", 1)
                            line = line_bytes.decode().strip()
                            last_line_info = line

                            if line.startswith("data:"):
                                content = line[5:].strip()
                                if not content: continue
                                try:
                                    sse_chunk = json.loads(content)
                                    if sse_chunk.get("base_resp", {}).get("status_code", 0) != 0:
                                        logger.error("Minimax API error: %s", sse_chunk.get("base_resp"))
                                        continue

                                    data_obj = sse_chunk.get("data")
                                    if not data_obj: continue

                                    # --- Subtitle Parsing Logic ---
                                    # The SSE payload for subtitles has a nested structure:
                                    # {
                                    #   "data": {
                                    #     "subtitle": {
                                    #       "timestamped_words": [
                                    #         {"word": "Hello", "time_begin": 100, "time_end": 300}, ...
                                    #       ]
                                    #     }, ...
                                    #   }
                                    # }
                                    # We parse this to extract word-level timestamps.
                                    subtitle_obj = data_obj.get("subtitle")
                                    if subtitle_obj and isinstance(subtitle_obj, dict):
                                        word_timestamps = subtitle_obj.get("timestamped_words")
                                        if word_timestamps and isinstance(word_timestamps, list):
                                            # Iterate over the list of word data dictionaries
                                            for word_data in word_timestamps:
                                                # word_data is a dictionary like {'word': 'Hello', 'time_begin': ...}
                                                start_time_sec = word_data['time_begin'] / 1000.0
                                                end_time_sec = word_data['time_end'] / 1000.0
                                                print(start_time_sec, end_time_sec)
                                                output_emitter.push_timed_transcript(
                                                    TimedString(
                                                        text=word_data['word'],
                                                        start_time=start_time_sec,
                                                        end_time=end_time_sec
                                                    )
                                                )
                                    # --- End of Subtitle Parsing ---

                                    audio_hex = data_obj.get("audio")
                                    if audio_hex:
                                        output_emitter.push(bytes.fromhex(audio_hex))

                                    if data_obj.get("status") == 2:
                                        break
                                except json.JSONDecodeError:
                                    logger.warning("Could not decode JSON from stream: %s", content)

                            if last_line_info and (
                                    '"status":2' in last_line_info or '"finish":true' in last_line_info): break
                    if last_line_info and (
                            '"status":2' in last_line_info or '"finish":true' in last_line_info): break
            except Exception as e:
                print(f"\n\n❌❌❌ API Call Error (Sentence: '{sentence}') ❌❌❌")
                import traceback
                traceback.print_exc()
            finally:
                self._sentence_queue.task_done()

        output_emitter.end_segment()
        output_emitter.end_input()

    # The Agent's input methods are piped to our sentence stream.
    def push_text(self, token: str) -> None:
        self._sentence_stream.push_text(token)

    def flush(self) -> None:
        self._sentence_stream.flush()

    def end_input(self) -> None:
        self._sentence_stream.end_input()

    async def aclose(self) -> None:
        await self._sentence_stream.aclose()
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

        session = self._tts._ensure_session()
        try:
            url = self._opts.get_http_url("v1/t2a_v2")
            print("url", url)
            params = {"GroupId": self._opts.group_id}
            headers = {"Authorization": f"Bearer {self._opts.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self._opts.model,
                "text": self._input_text,
                "stream": False,
                "voice_setting": {
                    "voice_id": self._opts.voice_id,
                    "speed": self._opts.speed,
                    "vol": self._opts.vol,
                    "pitch": self._opts.pitch
                },
                "audio_setting": {
                    "format": "pcm",
                    "sample_rate": self._opts.sample_rate,
                    "bitrate": self._opts.bitrate
                }
            }

            if self._opts.emotion is not None:
                print("emotion", self._opts.emotion)
                payload["voice_setting"]["emotion"] = self._opts.emotion

            if self._opts.subtitle_enable:
                print("subtitle", self._opts.subtitle_enable)
                payload['subtitle_enable'] = True
                payload['subtitle_type'] = self._opts.subtitle_type

            voice_modify = {}

            # 1. pronunciation_dict
            if self._opts.pronunciation_dict:
                # API form is "pronunciation_dict": {"tone": [...]}
                # be sure the "tone" para is exist
                if "tone" in self._opts.pronunciation_dict:
                    payload['pronunciation_dict'] = self._opts.pronunciation_dict


            if self._opts.intensity is not None:
                voice_modify['intensity'] = self._opts.intensity
            if self._opts.timbre is not None:
                voice_modify['timbre'] = self._opts.timbre
            if self._opts.sound_effects is not None:
                voice_modify['sound_effects'] = self._opts.sound_effects

            # only when voice_modify is not None，
            if voice_modify:
                payload['voice_modify'] = voice_modify

            # Make the main API call for audio and subtitle info
            async with session.post(url, headers=headers, params=params, json=payload, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()

            if data.get("base_resp", {}).get("status_code", 0) != 0:
                raise APIStatusError(f"Minimax API returned an error: {data.get('base_resp')}")

            subtitle_url = data.get("data", {}).get("subtitle_file")
            print(subtitle_url)

            # If a subtitle file URL is provided, download and process it.
            if subtitle_url:
                logger.info("Downloading subtitles from: %s", subtitle_url)

                # Make a second GET request to download the subtitle file.
                async with session.get(subtitle_url, timeout=10) as sub_response:
                    # Check if the download request was successful.
                    sub_response.raise_for_status()

                    # The server might return an incorrect Content-Type, so we force JSON parsing.
                    subtitle_list = await sub_response.json(content_type=None)

                    # Parse the specific structure of the subtitle file.
                    if isinstance(subtitle_list, list):
                        # Iterate through the top-level list (usually has one element).
                        for title_obj in subtitle_list:
                            # Get the list of word timestamps from the object.
                            word_timestamps = title_obj.get("timestamped_words")
                            if word_timestamps and isinstance(word_timestamps, list):
                                # Iterate through word-level timestamps.
                                for word_data in word_timestamps:
                                    # Minimax timestamps are in milliseconds, so convert to seconds.
                                    start_time_sec = word_data['time_begin'] / 1000.0
                                    end_time_sec = word_data['time_end'] / 1000.0

                                    # Create and push the TimedString object.
                                    output_emitter.push_timed_transcript(
                                        TimedString(
                                            text=word_data['word'],
                                            start_time=start_time_sec,
                                            end_time=end_time_sec
                                        )
                                    )

            # --- Process audio data ---
            # The audio data is in the main API response.
            audio_hex = data.get("data", {}).get("audio")
            if audio_hex:
                output_emitter.push(bytes.fromhex(audio_hex))

            output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            # Catch specific aiohttp errors and re-raise them as LiveKit Agents API errors
            # for consistent error handling.
            error_message = f"Minimax API request failed with status {e.status}: {e.message}"
            raise APIStatusError(error_message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e