# Copyright 2025 LiveKit, Inc.
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

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels, Voice, VoiceSettings
from .version import __version__

API_VERSION = __version__
API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "LiveKit-Plugin-Respeecher-Version"
API_BASE_URL = "https://api.respeecher.com/v1"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "/public/tts/en-rt",
        encoding: TTSEncoding = "pcm_s16le",
        voice_id: str = "samantha",
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        sample_rate: int = 24000,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of Respeecher TTS.

        Args:
            api_key: Respeecher API key. If not provided, uses RESPEECHER_API_KEY env variable.
            model: The Respeecher TTS model to use.
            encoding: Audio encoding format.
            voice_id: ID of the voice to use. Different set of voices is available for different models. Thus, update the value after getting list_voices() API.
            voice_settings: Optional voice settings including sampling parameters.
            sample_rate: Audio sample rate in Hz.
            http_session: Optional aiohttp session to use for requests.
            base_url: The base URL for the Respeecher API.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        respeecher_api_key = api_key if is_given(api_key) else os.environ.get("RESPEECHER_API_KEY")
        if not respeecher_api_key:
            raise ValueError("RESPEECHER_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            voice_id=voice_id,
            voice_settings=voice_settings,
            api_key=respeecher_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        """List available voices from Respeecher API"""
        async with self._ensure_session().get(
            f"{self._opts.base_url}{self._opts.model}/voices",
            headers={
                API_AUTH_HEADER: self._opts.api_key,
                API_VERSION_HEADER: API_VERSION,
            },
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            voices = []
            for voice_data in data:
                voices.append(Voice(voice_data))

            if len(voices) == 0:
                raise APIError("No voices are available")

            return voices

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
    ) -> None:
        """Update TTS options"""
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(voice_settings):
            self._opts.voice_settings = voice_settings

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        if self._session:
            await self._session.close()
            self._session = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text using Respeecher HTTPS endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis"""
        json_data = {
            "transcript": self._input_text,
            "voice": {
                "id": self._tts._opts.voice_id,
            },
            "output_format": {
                "sample_rate": self._tts._opts.sample_rate,
                "encoding": self._tts._opts.encoding,
            },
        }

        if (
            is_given(self._tts._opts.voice_settings)
            and self._tts._opts.voice_settings.sampling_params
        ):
            json_data["voice"]["sampling_params"] = self._tts._opts.voice_settings.sampling_params

        http_url = f"{self._tts._opts.base_url}{self._tts._opts.model}/tts/bytes"

        try:
            async with self._tts._ensure_session().post(
                http_url,
                headers={
                    API_AUTH_HEADER: self._tts._opts.api_key,
                    API_VERSION_HEADER: API_VERSION,
                    "Content-Type": "application/json",
                },
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/wav",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using WebSocket for real-time synthesis"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        ws_url = self._tts._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        full_ws_url = f"{ws_url}{self._tts._opts.model}/tts/websocket?api_key={self._tts._opts.api_key}&source={API_VERSION_HEADER}&version={API_VERSION}"

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()

        async def _ws_operation():
            try:
                ws = await asyncio.wait_for(
                    self._tts._ensure_session().ws_connect(full_ws_url),
                    timeout=self._conn_options.timeout,
                )
            except asyncio.TimeoutError:
                raise APITimeoutError() from None

            async with ws:

                @utils.log_exceptions(logger=logger)
                async def input_task() -> None:
                    async for data in self._input_ch:
                        if isinstance(data, self._FlushSentinel):
                            sent_tokenizer_stream.flush()
                            continue

                        sent_tokenizer_stream.push_text(data)

                    sent_tokenizer_stream.end_input()

                @utils.log_exceptions(logger=logger)
                async def send_task() -> None:
                    async for sent in sent_tokenizer_stream:
                        generate_request = {
                            "context_id": request_id,
                            "transcript": sent.token,
                            "voice": {
                                "id": self._tts._opts.voice_id,
                            },
                            "continue": True,  # Always True for streamed sentences
                            "output_format": {
                                "encoding": self._tts._opts.encoding,
                                "sample_rate": self._tts._opts.sample_rate,
                            },
                        }
                        if (
                            is_given(self._tts._opts.voice_settings)
                            and self._tts._opts.voice_settings.sampling_params
                        ):
                            generate_request["voice"]["sampling_params"] = (
                                self._tts._opts.voice_settings.sampling_params
                            )

                        self._mark_started()
                        await ws.send_str(json.dumps(generate_request))

                    # Send final message with continue=False to signal end of stream
                    end_request = {
                        "context_id": request_id,
                        "transcript": "",
                        "voice": {
                            "id": self._tts._opts.voice_id,
                        },
                        "continue": False,
                        "output_format": {
                            "encoding": self._tts._opts.encoding,
                            "sample_rate": self._tts._opts.sample_rate,
                        },
                    }
                    if (
                        is_given(self._tts._opts.voice_settings)
                        and self._tts._opts.voice_settings.sampling_params
                    ):
                        end_request["voice"]["sampling_params"] = (
                            self._tts._opts.voice_settings.sampling_params
                        )

                    await ws.send_str(json.dumps(end_request))

                @utils.log_exceptions(logger=logger)
                async def recv_task() -> None:
                    current_segment_id: str | None = None
                    while True:
                        msg = await ws.receive()
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            raise APIStatusError(
                                message="Respeecher websocket closed unexpectedly",
                                status_code=500,
                                request_id=request_id,
                                body=None,
                            )

                        if msg.type != aiohttp.WSMsgType.TEXT:
                            logger.warning("Unexpected Respeecher message type %s", msg.type)
                            continue

                        data = json.loads(msg.data)

                        if data.get("type") == "error":
                            logger.error(f"Respeecher API error: {data.get('error')}")
                            raise APIError(message=data.get("error"))

                        if data.get("type") == "chunk":
                            if current_segment_id is None:
                                current_segment_id = request_id
                                output_emitter.start_segment(segment_id=current_segment_id)

                            audio_data = base64.b64decode(data["data"])
                            output_emitter.push(audio_data)
                        elif data.get("type") == "done":
                            logger.debug(f"Received done message: {data}")
                            # End the current segment if one was started
                            if current_segment_id is not None:
                                output_emitter.end_segment()
                                current_segment_id = None

                            # Only end input when the sentence tokenizer stream is closed
                            # and we've received the final done message
                            if sent_tokenizer_stream.closed:
                                output_emitter.end_input()
                                return
                        else:
                            raise APIError("Unexpected websocket message type")

                tasks = [
                    asyncio.create_task(input_task()),
                    asyncio.create_task(send_task()),
                    asyncio.create_task(recv_task()),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)

        try:
            await _ws_operation()
        except APITimeoutError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
