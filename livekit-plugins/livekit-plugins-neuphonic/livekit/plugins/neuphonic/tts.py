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

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncodings, TTSLangCodes, TTSModels

API_BASE_URL = "api.neuphonic.com"
AUTHORIZATION_HEADER = "X-API-KEY"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    base_url: str
    api_key: str
    model: TTSModels | str
    lang_code: TTSLangCodes | str
    encoding: TTSEncodings | str
    sampling_rate: int
    speed: float
    voice_id: NotGivenOr[str] = NOT_GIVEN

    @property
    def model_params(self) -> dict:
        """Returns a dictionary of model parameters for API requests."""
        params = {
            "voice_id": self.voice_id,
            "model": self.model,
            "lang_code": self.lang_code,
            "encoding": self.encoding,
            "sampling_rate": self.sampling_rate,
            "speed": self.speed,
        }
        return {k: v for k, v in params.items() if is_given(v) and v is not None}

    def get_query_param_string(self):
        """Forms the query parameter string from all model parameters."""
        queries = []
        for key, value in self.model_params.items():
            queries.append(f"{key}={value}")

        return "?" + "&".join(queries)


def _parse_sse_message(message: str) -> dict:
    """
    Parse each response from the SSE endpoint.

    The message will either be a string reading:
    - `event: error`
    - `event: message`
    - `data: { "status_code": 200, "data": {"audio": ... } }`
    """
    message = message.strip()

    if not message or "data" not in message:
        return None

    _, value = message.split(": ", 1)
    message = json.loads(value)

    if message.get("errors") is not None:
        raise Exception(f"Status {message.status_code} error received: {message.errors}.")

    return message


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "neu_hq",
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        lang_code: TTSLangCodes | str = "en",
        encoding: TTSEncodings | str = "pcm_linear",
        speed: float = 1.0,
        sample_rate: int = 22050,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of the Neuphonic TTS.

        See https://docs.neuphonic.com for more documentation on all of these options, or go to https://app.neuphonic.com/ to test out different options.

        Args:
            model (TTSModels | str, optional): The Neuphonic model to use. See Defaults to "neu_hq".
            voice_id (str, optional): The voice ID for the desired voice. Defaults to None.
            lang_code (TTSLanguages | str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncodings | str, optional): The audio encoding format. Defaults to "pcm_mulaw".
            speed (float, optional): The audio playback speed. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 22050.
            api_key (str | None, optional): The Neuphonic API key. If not provided, it will be read from the NEUPHONIC_API_TOKEN environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Neuphonic API. Defaults to "api.neuphonic.com".
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        neuphonic_api_key = api_key if is_given(api_key) else os.environ.get("NEUPHONIC_API_TOKEN")
        if not neuphonic_api_key:
            raise ValueError("API key must be provided or set in NEUPHONIC_API_TOKEN")

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice_id,
            lang_code=lang_code,
            encoding=encoding,
            speed=speed,
            sampling_rate=sample_rate,
            api_key=neuphonic_api_key,
            base_url=base_url,
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=90,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = f"wss://{self._opts.base_url}/speak/{self._opts.lang_code}{self._opts.get_query_param_string()}"

        return await asyncio.wait_for(
            session.ws_connect(url, headers={AUTHORIZATION_HEADER: self._opts.api_key}),
            self._conn_options.timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        lang_code: NotGivenOr[TTSLangCodes] = NOT_GIVEN,
        encoding: NotGivenOr[TTSEncodings] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, voice_id, lang_code,
        encoding, speed and sample_rate. If any parameter is not provided, the existing value will be
        retained.

        Args:
            model (TTSModels | str, optional): The Neuphonic model to use.
            voice_id (str, optional): The voice ID for the desired voice.
            lang_code (TTSLanguages | str, optional): The language code for synthesis..
            encoding (TTSEncodings | str, optional): The audio encoding format.
            speed (float, optional): The audio playback speed.
            sample_rate (int, optional): The audio sample rate in Hz.
        """  # noqa: E501
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(lang_code):
            self._opts.lang_code = lang_code
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sampling_rate = sample_rate
        self._pool.invalidate()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            pool=self._pool,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the SSE endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sampling_rate, num_channels=NUM_CHANNELS
        )

        json_data = {
            "text": self._input_text,
            **self._opts.model_params,
        }

        headers = {
            AUTHORIZATION_HEADER: self._opts.api_key,
        }

        try:
            async with self._session.post(
                f"https://{self._opts.base_url}/sse/speak/{self._opts.lang_code}",
                headers=headers,
                json=json_data,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
                read_bufsize=10
                * 1024
                * 1024,  # large read_bufsize to avoid `ValueError: Chunk too big`
            ) as resp:
                resp.raise_for_status()
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )

                async for line in resp.content:
                    message = line.decode("utf-8").strip()
                    if message:
                        parsed_message = _parse_sse_message(message)

                        if (
                            parsed_message is not None
                            and parsed_message.get("data", {}).get("audio") is not None
                        ):
                            audio_bytes = base64.b64decode(parsed_message["data"]["audio"])

                            for frame in bstream.write(audio_bytes):
                                emitter.push(frame)

                for frame in bstream.flush():
                    emitter.push(frame)

                emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ):
        super().__init__(tts=tts)
        self._opts, self._pool = opts, pool

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse):
            """Stream text to the websocket."""
            async for data in self._input_ch:
                self._mark_started()

                if isinstance(data, self._FlushSentinel):
                    await ws.send_str(json.dumps({"text": "<STOP>"}))
                    continue

                await ws.send_str(json.dumps({"text": data}))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sampling_rate,
                num_channels=NUM_CHANNELS,
            )
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Neuphonic connection closed unexpectedly",
                        request_id=request_id,
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected Neuphonic message type %s", msg.type)
                    continue

                data = json.loads(msg.data)

                if data.get("data"):
                    b64data = base64.b64decode(data["data"]["audio"])
                    for frame in audio_bstream.write(b64data):
                        emitter.push(frame)

                    if data["data"].get("stop"):  # A bool flag, is True when audio reaches "<STOP>"
                        for frame in audio_bstream.flush():
                            emitter.push(frame)
                        emitter.flush()
                        break  # we are not going to receive any more audio
                else:
                    logger.error("Unexpected Neuphonic message %s", data)

        async with self._pool.connection() as ws:
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
