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

import base64
import os
from dataclasses import dataclass
from typing import Optional

from hume import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import AsyncChatClientWithWebsocket, ChatConnectOptions
from hume.empathic_voice.types.assistant_input import AssistantInput
from hume.empathic_voice.types.session_settings import SessionSettings
from hume.empathic_voice.types.audio_output import AudioOutput
from hume.empathic_voice.types.assistant_end import AssistantEnd

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)

from .log import logger

NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 24000

@dataclass
class _TTSOptions:
    api_key: str
    sample_rate: int
    config_id: Optional[str] = None
    config_version: Optional[str] = None
    verbose_transcription: Optional[bool] = None
    secret_key: Optional[str] = None
    resumed_chat_group_id: Optional[str] = None

class HumeClientWrapper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = AsyncHumeClient(api_key=api_key)

    def get_headers(self, include_auth: bool = True) -> dict[str, str]:
        return {}

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        http_session: Optional[AsyncChatClientWithWebsocket] = None,
        config_id: Optional[str] = None,
        config_version: Optional[str] = None,
        verbose_transcription: Optional[bool] = None,
        secret_key: Optional[str] = None,
        resumed_chat_group_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("HUME_API_KEY")
        if not api_key:
            raise ValueError("HUME_API_KEY must be set")

        self._opts = _TTSOptions(
            api_key=api_key,
            sample_rate=sample_rate,
            config_id=config_id,
            config_version=config_version,
            verbose_transcription=verbose_transcription,
            secret_key=secret_key,
            resumed_chat_group_id=resumed_chat_group_id,
        )
        self._session = http_session
        self._client_wrapper = HumeClientWrapper(api_key=api_key)
        self._last_chat_group_id = resumed_chat_group_id

    def _ensure_session(self) -> AsyncChatClientWithWebsocket:
        if not self._session:
            self._session = AsyncChatClientWithWebsocket(client_wrapper=self._client_wrapper)
        return self._session

    def update_options(
        self,
        *,
        sample_rate: int | None = None,
    ) -> None:
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate

    @property
    def last_chat_group_id(self) -> Optional[str]:
        return self._last_chat_group_id

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        def extract_wav_data(wav_bytes: bytes) -> bytes:
            """Extract raw PCM data from WAV bytes, skipping RIFF header."""
            # WAV format:
            # Offset  Size  Description
            # 0-3     4    "RIFF"
            # 4-7     4    File size (ignore)
            # 8-11    4    "WAVE"
            # 12-15   4    "fmt "
            # 16-19   4    Length of format data (16)
            # 20-21   2    Type of format (1 is PCM)
            # 22-23   2    Number of channels
            # 24-27   4    Sample rate
            # 28-31   4    Bytes per second
            # 32-33   2    Bytes per sample
            # 34-35   2    Bits per sample
            # 36-39   4    "data"
            # 40-43   4    Length of data
            # 44+     n    The actual audio data
            
            # Basic validation
            if not wav_bytes.startswith(b'RIFF') or b'WAVE' not in wav_bytes[:12]:
                raise ValueError("Invalid WAV format")
                
            # Find data chunk
            data_pos = wav_bytes.find(b'data')
            if data_pos == -1:
                raise ValueError("No data chunk found in WAV")
                
            # Skip 'data' and length fields (8 bytes) to get to actual audio data
            audio_data_start = data_pos + 8
            return wav_bytes[audio_data_start:]

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        last_frame = None

        try:
            session = self._tts._ensure_session()
            connect_options = ChatConnectOptions(
                api_key=self._tts._opts.api_key,
                resumed_chat_group_id=self._tts.last_chat_group_id,
                config_id=self._tts._opts.config_id,
                config_version=self._tts._opts.config_version,
                verbose_transcription=self._tts._opts.verbose_transcription,
                secret_key=self._tts._opts.secret_key,
            )

            async with session.connect(connect_options) as connection:
                # Configure session settings
                settings = SessionSettings(
                    audio={
                        "encoding": "linear16",
                        "sample_rate": self._tts.sample_rate,
                        "channels": NUM_CHANNELS,
                    }
                )
                await connection.send_session_settings(settings)

                assistant_input = AssistantInput(text=self._input_text)
                await connection.send_assistant_input(assistant_input)

                async for event in connection:
                    if isinstance(event, AudioOutput):
                        audio_data = base64.b64decode(event.data)
                        pcm_data = extract_wav_data(audio_data)
                        for frame in audio_bstream.write(pcm_data):
                            if last_frame is not None:
                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id=request_id,
                                        frame=last_frame,
                                    )
                                )
                            last_frame = frame
                    elif isinstance(event, AssistantEnd):
                        # Send the final frame with is_final=True
                        if last_frame is not None:
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=last_frame,
                                    is_final=True,
                                )
                            )
                        break
        except Exception as e:
            logger.error("Error in Hume TTS synthesis", exc_info=e)
            raise APIConnectionError() from e
