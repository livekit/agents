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
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Literal, Union

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer

from ._utils import PeriodicCollector
from .log import logger

BASE_URL = "https://api.gladia.io/v2/live"

MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state


@dataclass
class LanguageConfiguration:
    languages: Optional[List[str]] = None
    code_switching: bool = True


@dataclass
class TranslationConfiguration:
    enabled: bool = False
    target_languages: List[str] = dataclasses.field(default_factory=list)
    model: str = "base"
    match_original_utterances: bool = True


@dataclass
class STTOptions:
    language_config: LanguageConfiguration
    interim_results: bool
    sample_rate: int
    bit_depth: Literal[8, 16, 24, 32]
    channels: int
    encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"]
    translation_config: TranslationConfiguration = dataclasses.field(
        default_factory=TranslationConfiguration
    )
    energy_filter: Union[AudioEnergyFilter, bool] = False


class STT(stt.STT):
    def __init__(
        self,
        *,
        interim_results: bool = True,
        languages: Optional[List[str]] = None,
        code_switching: bool = True,
        sample_rate: int = 16000,
        bit_depth: Literal[8, 16, 24, 32] = 16,
        channels: int = 1,
        encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"] = "wav/pcm",
        api_key: Optional[str] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
        base_url: str = BASE_URL,
        energy_filter: Union[AudioEnergyFilter, bool] = False,
        translation_enabled: bool = False,
        translation_target_languages: Optional[List[str]] = None,
        translation_model: str = "base",
        translation_match_original_utterances: bool = True,
    ) -> None:
        """Create a new instance of Gladia STT.

        Args:
            interim_results: Whether to return interim (non-final) transcription results. Defaults to True.
            languages: List of language codes to use for recognition. Defaults to None (auto-detect).
            code_switching: Whether to allow switching between languages during recognition. Defaults to True.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            bit_depth: The bit depth of the audio. Defaults to 16.
            channels: The number of audio channels. Defaults to 1.
            encoding: The encoding of the audio. Defaults to "wav/pcm".
            api_key: Your Gladia API key. If not provided, will look for GLADIA_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for Gladia API. Defaults to "https://api.gladia.io/v2/live".
            energy_filter: Audio energy filter configuration for voice activity detection.
                         Can be a boolean or AudioEnergyFilter instance. Defaults to False.
            translation_enabled: Whether to enable translation. Defaults to False.
            translation_target_languages: List of target languages for translation. Required if translation_enabled is True.
            translation_model: Translation model to use. Defaults to "base".
            translation_match_original_utterances: Whether to match original utterances with translations. Defaults to True.

        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=interim_results)
        )
        self._base_url = base_url

        api_key = api_key or os.environ.get("GLADIA_API_KEY")
        if api_key is None:
            raise ValueError("Gladia API key is required")

        self._api_key = api_key

        language_config = LanguageConfiguration(languages=languages, code_switching=code_switching)

        translation_config = TranslationConfiguration(
            enabled=translation_enabled,
            target_languages=translation_target_languages or [],
            model=translation_model,
            match_original_utterances=translation_match_original_utterances,
        )

        if translation_enabled and not translation_target_languages:
            raise ValueError(
                "translation_target_languages is required when translation_enabled is True"
            )

        self._opts = STTOptions(
            language_config=language_config,
            interim_results=interim_results,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            encoding=encoding,
            translation_config=translation_config,
            energy_filter=energy_filter,
        )
        self._session = http_session
        self._streams = weakref.WeakSet()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[List[str]] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Implement synchronous speech recognition for Gladia using the live endpoint."""
        config = self._sanitize_options(languages=language)

        streaming_config = {
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "bit_depth": config.bit_depth,
            "channels": config.channels,
            "language_config": {
                "languages": config.language_config.languages or [],
                "code_switching": config.language_config.code_switching,
            },
            "realtime_processing": {
                "words_accurate_timestamps": False,
                "custom_vocabulary": False,
                "custom_vocabulary_config": {
                    "vocabulary": [
                        "Gladia",
                        {"value": "Gladia", "intensity": 0.5},
                    ],
                    "default_intensity": 0.5,
                },
                "custom_spelling": False,
                "custom_spelling_config": {
                    "spelling_dictionary": {
                        "SQL": ["Sequel"],
                    }
                },
            },
        }

        # Add translation configuration if enabled
        if config.translation_config.enabled:
            streaming_config["realtime_processing"]["translation"] = True
            streaming_config["realtime_processing"]["translation_config"] = {
                "target_languages": config.translation_config.target_languages,
                "model": config.translation_config.model,
                "match_original_utterances": config.translation_config.match_original_utterances,
            }

        try:
            # Initialize a session with Gladia
            session_response = await self._init_live_session(streaming_config, conn_options)
            session_id = session_response["id"]
            session_url = session_response["url"]

            # Connect to the WebSocket
            async with self._ensure_session().ws_connect(
                session_url,
                timeout=aiohttp.ClientTimeout(
                    total=30,  # Keep a reasonable total timeout
                    sock_connect=conn_options.timeout,
                ),
            ) as ws:
                # Combine audio frames to get a single frame with all raw PCM data
                combined_frame = rtc.combine_audio_frames(buffer)
                # Get the raw bytes from the combined frame
                pcm_data = combined_frame.data.tobytes()

                bytes_per_second = config.sample_rate * config.channels * (config.bit_depth // 8)
                chunk_size = (bytes_per_second * 150) // 1000
                chunk_size = max(chunk_size, 1024)

                logger.debug(
                    f"Sending {len(pcm_data)} bytes of PCM data in chunks of {chunk_size} bytes"
                )

                # Send raw PCM audio data in chunks
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i : i + chunk_size]
                    chunk_b64 = base64.b64encode(chunk).decode("utf-8")
                    await ws.send_str(
                        json.dumps({"type": "audio_chunk", "data": {"chunk": chunk_b64}})
                    )

                # Tell Gladia we're done sending audio
                await ws.send_str(json.dumps({"type": "stop_recording"}))
                logger.debug("Sent stop_recording message")

                # Wait for final transcript
                utterances = []

                # Receive messages until we get the post_final_transcript message
                try:
                    # Set a timeout for waiting for the final results after sending stop_recording
                    receive_timeout = conn_options.timeout * 5
                    async for msg in ws.iter(timeout=receive_timeout):
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            logger.debug(
                                f"Received Gladia message: {data.get('type')}"
                            )  # Log message type
                            # Collect final utterances
                            if data["type"] == "transcript" and data["data"]["is_final"]:
                                utterance = data["data"]["utterance"]
                                logger.debug(f"Received final utterance: {utterance.get('text')}")
                                utterances.append(utterance)
                            # Check for translation as the final result if enabled
                            elif (
                                data["type"] == "translation" and config.translation_config.enabled
                            ):
                                logger.debug("Received translation message")
                                pass
                            elif data["type"] == "post_final_transcript":
                                logger.debug("Received post_final_transcript message")
                                break
                            elif data["type"] == "error":
                                logger.error(f"Gladia WebSocket error: {data.get('data')}")
                                raise APIConnectionError(
                                    f"Gladia WebSocket error: {data.get('data')}"
                                )

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"Gladia WebSocket connection error: {ws.exception()}")
                            raise ws.exception() or APIConnectionError(
                                "Gladia WebSocket connection error"
                            )
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            logger.warning(
                                f"Gladia WebSocket closed unexpectedly during result receiving: type={msg.type}"
                            )
                            break

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for Gladia final transcript ({receive_timeout}s)"
                    )
                    if not utterances:
                        raise APITimeoutError(
                            f"Timeout waiting for Gladia final transcript ({receive_timeout}s)"
                        )

                # Create a speech event from the collected final utterances
                return self._create_speech_event(
                    utterances, session_id, config.language_config.languages
                )

        except asyncio.TimeoutError as e:
            # Catch timeout during connection or initial phase
            logger.error(f"Timeout during Gladia connection/initialization: {e}")
            raise APITimeoutError("Timeout connecting to or initializing Gladia session") from e
        except aiohttp.ClientResponseError as e:
            # Error during session initialization POST request
            logger.error(f"Gladia API status error during session init: {e.status} {e.message}")
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=e.headers.get(
                    "X-Request-ID"
                ),  # Check if Gladia provides a request ID header
                body=await e.response.text() if hasattr(e, "response") else None,
            ) from e
        except aiohttp.ClientError as e:
            # General client errors (connection refused, DNS resolution etc.)
            logger.error(f"Gladia connection error: {e}")
            raise APIConnectionError(f"Gladia connection error: {e}") from e
        except Exception as e:
            # Catch-all for other unexpected errors
            logger.exception(
                f"Unexpected error during Gladia synchronous recognition: {e}"
            )  # Use logger.exception to include stack trace
            raise APIConnectionError(f"An unexpected error occurred: {e}") from e

    async def _init_live_session(self, config: dict, conn_options: APIConnectOptions) -> dict:
        """Initialize a live transcription session with Gladia."""
        try:
            async with self._ensure_session().post(
                url=self._base_url,
                json=config,
                headers={"X-Gladia-Key": self._api_key},
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                # Gladia returns 201 Created when successfully creating a session
                if res.status not in (200, 201):
                    raise APIStatusError(
                        message=f"Failed to initialize Gladia session: {res.status}",
                        status_code=res.status,
                        request_id=None,
                        body=await res.text(),
                    )
                return await res.json()
        except Exception as e:
            logger.exception(f"Failed to initialize Gladia session: {e}")
            raise APIConnectionError(f"Failed to initialize Gladia session: {str(e)}") from e

    def _create_speech_event(
        self, utterances: List[dict], session_id: str, languages: Optional[List[str]]
    ) -> stt.SpeechEvent:
        """Create a SpeechEvent from Gladia's transcript data."""
        alternatives = []

        # Process each utterance into a SpeechData object
        for utterance in utterances:
            text = utterance.get("text", "").strip()
            if text:
                alternatives.append(
                    stt.SpeechData(
                        language=utterance.get("language", languages[0] if languages else "en"),
                        start_time=utterance.get("start", 0),
                        end_time=utterance.get("end", 0),
                        confidence=utterance.get("confidence", 1.0),
                        text=text,
                    )
                )

        if not alternatives:
            alternatives.append(
                stt.SpeechData(
                    language=languages[0] if languages and len(languages) > 0 else "en",
                    start_time=0,
                    end_time=0,
                    confidence=1.0,
                    text="",
                )
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=session_id,
            alternatives=alternatives,
        )

    def stream(
        self,
        *,
        language: Optional[List[str]] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(languages=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        languages: Optional[List[str]] = None,
        code_switching: Optional[bool] = None,
        interim_results: Optional[bool] = None,
        sample_rate: Optional[int] = None,
        bit_depth: Optional[Literal[8, 16, 24, 32]] = None,
        channels: Optional[int] = None,
        encoding: Optional[Literal["wav/pcm", "wav/alaw", "wav/ulaw"]] = None,
        translation_enabled: Optional[bool] = None,
        translation_target_languages: Optional[List[str]] = None,
        translation_model: Optional[str] = None,
        translation_match_original_utterances: Optional[bool] = None,
    ):
        if languages is not None or code_switching is not None:
            language_config = dataclasses.replace(
                self._opts.language_config,
                languages=languages
                if languages is not None
                else self._opts.language_config.languages,
                code_switching=code_switching
                if code_switching is not None
                else self._opts.language_config.code_switching,
            )
            self._opts.language_config = language_config

        if (
            translation_enabled is not None
            or translation_target_languages is not None
            or translation_model is not None
            or translation_match_original_utterances is not None
        ):
            translation_config = dataclasses.replace(
                self._opts.translation_config,
                enabled=translation_enabled
                if translation_enabled is not None
                else self._opts.translation_config.enabled,
                target_languages=translation_target_languages
                if translation_target_languages is not None
                else self._opts.translation_config.target_languages,
                model=translation_model
                if translation_model is not None
                else self._opts.translation_config.model,
                match_original_utterances=translation_match_original_utterances
                if translation_match_original_utterances is not None
                else self._opts.translation_config.match_original_utterances,
            )
            self._opts.translation_config = translation_config

        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if bit_depth is not None:
            self._opts.bit_depth = bit_depth
        if channels is not None:
            self._opts.channels = channels
        if encoding is not None:
            self._opts.encoding = encoding

        for stream in self._streams:
            stream.update_options(
                languages=languages,
                code_switching=code_switching,
                interim_results=interim_results,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                encoding=encoding,
                translation_enabled=translation_enabled,
                translation_target_languages=translation_target_languages,
                translation_model=translation_model,
                translation_match_original_utterances=translation_match_original_utterances,
            )

    def _sanitize_options(self, *, languages: Optional[List[str]] = None) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if languages is not None:
            language_config = dataclasses.replace(
                config.language_config,
                languages=languages,
            )
            config.language_config = language_config
        return config


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._base_url = base_url
        self._speaking = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

        self._audio_energy_filter: Optional[AudioEnergyFilter] = None
        if opts.energy_filter:
            if isinstance(opts.energy_filter, AudioEnergyFilter):
                self._audio_energy_filter = opts.energy_filter
            else:
                self._audio_energy_filter = AudioEnergyFilter()

        self._pushed_audio_duration = 0.0
        self._request_id = ""
        self._reconnect_event = asyncio.Event()
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    def update_options(
        self,
        *,
        languages: Optional[List[str]] = None,
        code_switching: Optional[bool] = None,
        interim_results: Optional[bool] = None,
        sample_rate: Optional[int] = None,
        bit_depth: Optional[Literal[8, 16, 24, 32]] = None,
        channels: Optional[int] = None,
        encoding: Optional[Literal["wav/pcm", "wav/alaw", "wav/ulaw"]] = None,
        translation_enabled: Optional[bool] = None,
        translation_target_languages: Optional[List[str]] = None,
        translation_model: Optional[str] = None,
        translation_match_original_utterances: Optional[bool] = None,
    ):
        if languages is not None or code_switching is not None:
            language_config = dataclasses.replace(
                self._opts.language_config,
                languages=languages
                if languages is not None
                else self._opts.language_config.languages,
                code_switching=code_switching
                if code_switching is not None
                else self._opts.language_config.code_switching,
            )
            self._opts.language_config = language_config

        if (
            translation_enabled is not None
            or translation_target_languages is not None
            or translation_model is not None
            or translation_match_original_utterances is not None
        ):
            translation_config = dataclasses.replace(
                self._opts.translation_config,
                enabled=translation_enabled
                if translation_enabled is not None
                else self._opts.translation_config.enabled,
                target_languages=translation_target_languages
                if translation_target_languages is not None
                else self._opts.translation_config.target_languages,
                model=translation_model
                if translation_model is not None
                else self._opts.translation_config.model,
                match_original_utterances=translation_match_original_utterances
                if translation_match_original_utterances is not None
                else self._opts.translation_config.match_original_utterances,
            )
            self._opts.translation_config = translation_config

        if interim_results is not None:
            self._opts.interim_results = interim_results
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
        if bit_depth is not None:
            self._opts.bit_depth = bit_depth
        if channels is not None:
            self._opts.channels = channels
        if encoding is not None:
            self._opts.encoding = encoding

        self._reconnect_event.set()

    async def _run(self) -> None:
        backoff_time = 1.0
        max_backoff = 30.0

        while True:
            try:
                # Initialize the Gladia session
                session_info = await self._init_live_session()
                session_url = session_info["url"]
                self._request_id = session_info["id"]

                # Reset backoff on success
                backoff_time = 1.0

                # Connect to the WebSocket
                async with self._session.ws_connect(session_url) as ws:
                    self._ws = ws
                    logger.info(f"Connected to Gladia session {self._request_id}")

                    send_task = asyncio.create_task(self._send_audio_task())
                    recv_task = asyncio.create_task(self._recv_messages_task())

                    wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                    try:
                        done, _ = await asyncio.wait(
                            [send_task, recv_task, wait_reconnect_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in done:
                            if task != wait_reconnect_task:
                                task.result()

                        if wait_reconnect_task not in done:
                            break

                        self._reconnect_event.clear()
                        logger.info("Reconnecting Gladia session due to options change")
                    finally:
                        await utils.aio.gracefully_cancel(send_task, recv_task, wait_reconnect_task)
                        self._ws = None
            except APIStatusError as e:
                if e.status_code == 429:
                    logger.warning(
                        f"Rate limited by Gladia API. Backing off for {backoff_time} seconds."
                    )
                    await asyncio.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, max_backoff)
                else:
                    logger.exception(f"Error in speech stream: {e}")
                    await asyncio.sleep(backoff_time)
            except Exception as e:
                logger.exception(f"Error in speech stream: {e}")
                # Wait a bit before reconnecting to avoid rapid reconnection attempts
                await asyncio.sleep(backoff_time)

    async def _init_live_session(self) -> dict:
        """Initialize a live session with Gladia."""
        streaming_config = {
            "encoding": self._opts.encoding,
            "sample_rate": self._opts.sample_rate,
            "bit_depth": self._opts.bit_depth,
            "channels": self._opts.channels,
            "language_config": {
                "languages": self._opts.language_config.languages or [],
                "code_switching": self._opts.language_config.code_switching,
            },
            "realtime_processing": {},
        }

        # Add translation configuration if enabled
        if self._opts.translation_config.enabled:
            streaming_config["realtime_processing"]["translation"] = True
            streaming_config["realtime_processing"]["translation_config"] = {
                "target_languages": self._opts.translation_config.target_languages,
                "model": self._opts.translation_config.model,
                "match_original_utterances": self._opts.translation_config.match_original_utterances,
            }

        try:
            async with self._session.post(
                url=self._base_url,
                json=streaming_config,
                headers={"X-Gladia-Key": self._api_key},
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                # Gladia returns 201 Created when successfully creating a session
                if res.status not in (200, 201):
                    raise APIStatusError(
                        message=f"Failed to initialize Gladia session: {res.status}",
                        status_code=res.status,
                        request_id=None,
                        body=await res.text(),
                    )
                return await res.json()
        except Exception as e:
            logger.exception(f"Failed to initialize Gladia session: {e}")
            raise

    async def _send_audio_task(self):
        """Send audio data to Gladia WebSocket."""
        if not self._ws:
            return

        # We'll aim to send audio chunks every ~100ms
        samples_100ms = self._opts.sample_rate // 10
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.channels,
            samples_per_channel=samples_100ms,
        )

        has_ended = False
        last_frame: Optional[rtc.AudioFrame] = None

        async for data in self._input_ch:
            if not self._ws:
                break

            frames: list[rtc.AudioFrame] = []
            if isinstance(data, rtc.AudioFrame):
                state = self._check_energy_state(data)
                if state in (
                    AudioEnergyFilter.State.START,
                    AudioEnergyFilter.State.SPEAKING,
                ):
                    if last_frame:
                        frames.extend(audio_bstream.write(last_frame.data.tobytes()))
                        last_frame = None
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif state == AudioEnergyFilter.State.END:
                    frames = audio_bstream.flush()
                    has_ended = True
                elif state == AudioEnergyFilter.State.SILENCE:
                    last_frame = data
            elif isinstance(data, self._FlushSentinel):
                frames = audio_bstream.flush()
                has_ended = True

            for frame in frames:
                self._audio_duration_collector.push(frame.duration)
                # Encode the audio data as base64
                chunk_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                message = json.dumps({"type": "audio_chunk", "data": {"chunk": chunk_b64}})
                await self._ws.send_str(message)

                if has_ended:
                    self._audio_duration_collector.flush()
                    await self._ws.send_str(json.dumps({"type": "stop_recording"}))
                    has_ended = False

        # Tell Gladia we're done sending audio when the stream ends
        if self._ws:
            await self._ws.send_str(json.dumps({"type": "stop_recording"}))

    async def _recv_messages_task(self):
        """Receive and process messages from Gladia WebSocket."""
        if not self._ws:
            return

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    self._process_gladia_message(data)
                except Exception as e:
                    logger.exception(f"Error processing Gladia message: {e}")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                break
            else:
                logger.warning(f"Unexpected message type from Gladia: {msg.type}")

    def _process_gladia_message(self, data: dict):
        """Process messages from Gladia WebSocket."""
        if data["type"] == "transcript":
            is_final = data["data"]["is_final"]
            utterance = data["data"]["utterance"]
            text = utterance.get("text", "").strip()

            if not self._speaking and text:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                    )
                )

            if text:
                language = utterance.get(
                    "language",
                    self._opts.language_config.languages[0]
                    if self._opts.language_config.languages
                    else "en",
                )

                speech_data = stt.SpeechData(
                    language=language,
                    start_time=utterance.get("start", 0),
                    end_time=utterance.get("end", 0),
                    confidence=utterance.get("confidence", 1.0),
                    text=text,
                )

                if is_final:
                    # Only emit FINAL_TRANSCRIPT for the *original* language
                    # if translation is NOT enabled.
                    if not self._opts.translation_config.enabled:
                        event = stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[speech_data],
                        )
                        self._event_ch.send_nowait(event)

                        # End of speech after final original transcript only if not translating
                        if self._speaking:
                            self._speaking = False
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.END_OF_SPEECH,
                                    request_id=self._request_id,
                                )
                            )
                    # If translation *is* enabled, we suppress this final event
                    # and wait for the 'translation' message to emit the final event.
                elif self._opts.interim_results:
                    # Always send INTERIM_TRANSCRIPT for the original language if enabled
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    )
                    self._event_ch.send_nowait(event)

        elif data["type"] == "translation":
            # Process translation messages according to Gladia's documentation:
            # https://docs.gladia.io/reference/realtime-messages/translation
            if self._opts.translation_config.enabled and "data" in data:
                translation_data = data["data"]

                # Extract translated utterance
                translated_utterance = translation_data.get("translated_utterance", {})
                if not translated_utterance:
                    logger.warning(
                        f"No translated_utterance in translation message: {translation_data}"
                    )
                    return

                # Get language information
                target_language = translation_data.get("target_language", "")
                language = translated_utterance.get("language", target_language)

                # Get the translated text
                translated_text = translated_utterance.get("text", "").strip()

                if translated_text and language:
                    # Log the translation
                    logger.info(
                        f"Translation from {translation_data.get('original_language')} to {language}: {translated_text}"
                    )

                    # Create speech data for the translation
                    speech_data = stt.SpeechData(
                        language=language,  # Use the target language
                        start_time=translated_utterance.get("start", 0),
                        end_time=translated_utterance.get("end", 0),
                        confidence=translated_utterance.get("confidence", 1.0),
                        text=translated_text,  # Use the translated text
                    )

                    # Emit FINAL_TRANSCRIPT containing the TRANSLATION
                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],  # Now contains translated data
                    )
                    self._event_ch.send_nowait(event)

                    # Emit END_OF_SPEECH after the final *translated* transcript
                    if self._speaking:
                        self._speaking = False
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                            )
                        )

        elif data["type"] == "post_final_transcript":
            # This is sent at the end of a session
            # We now tie END_OF_SPEECH to the emission of the relevant FINAL_TRANSCRIPT
            # (either original if no translation, or translated if translation is enabled).
            # So, we might not strictly need to act on this message anymore for END_OF_SPEECH,
            # but ensure speaking state is reset if somehow missed.
            if self._speaking:
                logger.warning(
                    "Resetting speaking state on post_final_transcript, END_OF_SPEECH might have been missed."
                )
                self._speaking = False
                # Optionally emit END_OF_SPEECH here as a fallback, though it might cause duplicates
                # self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id))

    def _check_energy_state(self, frame: rtc.AudioFrame) -> AudioEnergyFilter.State:
        """Check the energy state of an audio frame."""
        if self._audio_energy_filter:
            return self._audio_energy_filter.update(frame)
        return AudioEnergyFilter.State.SPEAKING

    def _on_audio_duration_report(self, duration: float) -> None:
        """Report the audio duration for usage tracking."""
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)
