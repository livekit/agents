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
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, cast

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    Language,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .constants import (
    API_AUTH_HEADER,
    API_VERSION,
    API_VERSION_HEADER,
    API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    REQUEST_ID_HEADER,
    USER_AGENT,
)
from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModels,
    TTSVoiceEmotion,
    TTSVoiceSpeed,
    _is_sonic_3,
)


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    speed: TTSVoiceSpeed | float | None
    emotion: list[TTSVoiceEmotion | str] | None
    volume: float | None
    word_timestamps: bool
    api_key: str
    language: Language | None
    base_url: str
    api_version: str
    pronunciation_dict_id: str | None

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "sonic-3",
        language: str | None = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str | list[float] = TTSDefaultVoiceId,
        speed: TTSVoiceSpeed | float | None = None,
        emotion: TTSVoiceEmotion | str | list[TTSVoiceEmotion | str] | None = None,
        volume: float | None = None,
        sample_rate: int = 24000,
        word_timestamps: bool = True,
        pronunciation_dict_id: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
        base_url: str = "https://api.cartesia.ai",
        api_version: str = API_VERSION,
    ) -> None:
        """
        Create a new instance of Cartesia TTS.

        See https://docs.cartesia.ai/reference/web-socket/stream-speech/stream-speech for more details on the Cartesia API.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-3".
            language (str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Speed of speech, with sonic-3, the value is valid between 0.6 and 2.0 (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-speed)
            emotion (list[TTSVoiceEmotion], optional): Emotion of the speech (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-emotion)
            volume (float, optional): Volume of the speech, with sonic-3, the value is valid between 0.5 and 2.0
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            word_timestamps (bool, optional): Whether to add word timestamps to the output. Defaults to True.
            pronunciation_dict_id (str, optional): The pronunciation dictionary ID to use for custom pronunciations. Defaults to None.
            api_key (str, optional): The Cartesia API key. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use. Defaults to `livekit.agents.tokenize.blingfire.SentenceTokenizer`.
            text_pacing (tts.SentenceStreamPacer | bool, optional): Stream pacer for the TTS. Set to True to use the default pacer, False to disable.
            base_url (str, optional): The base URL for the Cartesia API. Defaults to "https://api.cartesia.ai".
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        cartesia_api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError(
                "Cartesia API key is required, either as argument or set"
                " CARTESIA_API_KEY environment variable"
            )

        if isinstance(emotion, str):
            emotion = [emotion]

        self._opts = _TTSOptions(
            model=model,
            language=Language(language) if language else None,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            speed=speed,
            emotion=emotion,
            volume=volume,
            api_key=cartesia_api_key,
            base_url=base_url,
            word_timestamps=word_timestamps,
            api_version=api_version,
            pronunciation_dict_id=pronunciation_dict_id,
        )

        if speed or emotion or volume or pronunciation_dict_id:
            self._check_generation_config()

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

        if word_timestamps:
            if "preview" not in self._opts.model and (
                self._opts.language is not None
                and self._opts.language.language
                not in {
                    "en",
                    "de",
                    "es",
                    "fr",
                }
            ):
                # https://docs.cartesia.ai/api-reference/tts/compare-tts-endpoints
                logger.warning(
                    "word_timestamps is only supported for languages en, de, es, and fr with `sonic` models"
                    " or all languages with `preview` models"
                )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Cartesia"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/tts/websocket?api_key={self._opts.api_key}&cartesia_version={self._opts.api_version}"
        )
        ws = await asyncio.wait_for(
            session.ws_connect(url, headers={"User-Agent": USER_AGENT}), timeout
        )
        c_request_id = ws._response.headers.get(REQUEST_ID_HEADER)
        logger.debug(
            "Established new Cartesia TTS WebSocket connection",
            extra={"cartesia_request_id": c_request_id},
        )
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
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
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        voice: NotGivenOr[str | list[float]] = NOT_GIVEN,
        speed: NotGivenOr[TTSVoiceSpeed | float] = NOT_GIVEN,
        emotion: NotGivenOr[TTSVoiceEmotion | str | list[TTSVoiceEmotion | str]] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        pronunciation_dict_id: NotGivenOr[str] = NOT_GIVEN,
        api_version: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language, voice, speed,
        and emotion. If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-3".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Voice Control - Speed (https://docs.cartesia.ai/user-guides/voice-control)
            emotion (list[TTSVoiceEmotion], optional): Voice Control - Emotion (https://docs.cartesia.ai/user-guides/voice-control)
            pronunciation_dict_id (str, optional): The pronunciation dictionary ID to use for custom pronunciations.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = Language(language) if language else None
        if is_given(voice):
            self._opts.voice = cast(str | list[float], voice)
        if is_given(speed):
            self._opts.speed = cast(TTSVoiceSpeed | float, speed)
        if is_given(emotion):
            emotion = [emotion] if isinstance(emotion, str) else emotion
            self._opts.emotion = cast(list[TTSVoiceEmotion | str], emotion)
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pronunciation_dict_id):
            self._opts.pronunciation_dict_id = pronunciation_dict_id
        if is_given(api_version):
            self._opts.api_version = api_version

        if speed or emotion or volume or pronunciation_dict_id:
            self._check_generation_config()

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
        await self._pool.aclose()

    def _check_generation_config(self) -> None:
        if _is_sonic_3(self._opts.model):
            if self._opts.speed:
                if not isinstance(self._opts.speed, float):
                    raise ValueError("speed must be a float for sonic-3")
                if not 0.6 <= self._opts.speed <= 2.0:
                    logger.warning("speed must be between 0.6 and 2.0 for sonic-3")
            if self._opts.volume is not None and not 0.5 <= self._opts.volume <= 2.0:
                logger.warning("volume must be between 0.5 and 2.0 for sonic-3")
        elif (
            self._opts.api_version != API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS
            or self._opts.model != MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS
        ):
            logger.warning(
                f"speed and emotion controls are only supported for model '{MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS}', and API version '{API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS}', "
                "see https://docs.cartesia.ai/developer-tools/changelog for details",
                extra={
                    "model": self._opts.model,
                    "speed": self._opts.speed,
                    "emotion": self._opts.emotion,
                },
            )

        if self._opts.pronunciation_dict_id and not _is_sonic_3(self._opts.model):
            logger.warning(
                "pronunciation_dict_id is only supported for sonic-3 models",
                extra={
                    "model": self._opts.model,
                    "pronunciation_dict_id": self._opts.pronunciation_dict_id,
                },
            )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        json = _to_cartesia_options(self._opts, streaming=False)
        json["transcript"] = self._input_text

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_http_url("/tts/bytes"),
                headers={
                    API_AUTH_HEADER: self._opts.api_key,
                    API_VERSION_HEADER: API_VERSION,
                    "User-Agent": USER_AGENT,
                },
                json=json,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
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
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        input_sent_event = asyncio.Event()
        sent_tokens = deque[str]()

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_tokenizer_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_tokenizer_stream,
                audio_emitter=output_emitter,
            )

        async def _sentence_stream_task(
            ws: aiohttp.ClientWebSocketResponse, cartesia_context_id: str
        ) -> None:
            base_pkt = _to_cartesia_options(self._opts, streaming=True)
            async for ev in sent_tokenizer_stream:
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = cartesia_context_id
                token_pkt["transcript"] = ev.token + " "
                sent_tokens.append(ev.token + " ")
                token_pkt["continue"] = True
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))
                input_sent_event.set()

            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = cartesia_context_id
            end_pkt["transcript"] = " "
            sent_tokens.append(" ")
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))
            input_sent_event.set()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue

                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse, cartesia_context_id: str) -> None:
            current_segment_id: str | None = None
            await input_sent_event.wait()
            skip_aligning = False
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.error(
                        "Cartesia connection closed unexpectedly. Include the cartesia_context_id to support@cartesia.ai for help debugging.",
                        extra={"cartesia_context_id": cartesia_context_id},
                    )
                    raise APIStatusError(
                        "Cartesia connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                if current_segment_id is None:
                    current_segment_id = segment_id
                    output_emitter.start_segment(segment_id=segment_id)
                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    output_emitter.push(b64data)
                elif data.get("done"):
                    if sent_tokenizer_stream.closed:
                        # close only if the input stream is closed
                        output_emitter.end_input()
                        break
                elif word_timestamps := data.get("word_timestamps"):
                    # assuming Cartesia echos the sent text in the original format and order.
                    for word, start, end in zip(
                        word_timestamps["words"],
                        word_timestamps["start"],
                        word_timestamps["end"],
                        strict=False,
                    ):
                        if not sent_tokens or skip_aligning:
                            word = f"{word} "
                            skip_aligning = True
                        else:
                            sent = sent_tokens.popleft()
                            if (idx := sent.find(word)) != -1:
                                word, sent = sent[: idx + len(word)], sent[idx + len(word) :]
                                if sent.strip():
                                    sent_tokens.appendleft(sent)
                                elif sent and sent_tokens:
                                    # merge the remaining whitespace to the next sentence
                                    sent_tokens[0] = sent + sent_tokens[0]
                            else:
                                word = f"{word} "
                                skip_aligning = True

                        output_emitter.push_timed_transcript(
                            TimedString(text=word, start_time=start, end_time=end)
                        )
                elif data.get("type") == "error":
                    logger.error(
                        "Cartesia returned error. Include the cartesia_context_id to support@cartesia.ai for help debugging.",
                        extra={"cartesia_context_id": cartesia_context_id, "error": data},
                    )
                    raise APIError(f"Cartesia returned error: {data}")
                else:
                    logger.warning("unexpected message %s", data)

        cartesia_context_id = utils.shortuuid()
        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws, cartesia_context_id)),
                    asyncio.create_task(_recv_task(ws, cartesia_context_id)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            logger.exception(
                "Cartesia connection error. Include the cartesia_context_id to support@cartesia.ai for help debugging.",
                extra={"cartesia_context_id": cartesia_context_id},
            )
            raise APIConnectionError() from e


def _to_cartesia_options(opts: _TTSOptions, *, streaming: bool) -> dict[str, Any]:
    voice: dict[str, Any] = {}
    if isinstance(opts.voice, str):
        voice["mode"] = "id"
        voice["id"] = opts.voice
    else:
        voice["mode"] = "embedding"
        voice["embedding"] = opts.voice

    if opts.api_version == API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS:
        voice_controls: dict = {}
        if opts.speed:
            voice_controls["speed"] = opts.speed

        if opts.emotion:
            voice_controls["emotion"] = opts.emotion

        if voice_controls:
            voice["__experimental_controls"] = voice_controls

    options: dict[str, Any] = {
        "model_id": opts.model,
        "voice": voice,
        "output_format": {
            "container": "raw",
            "encoding": opts.encoding,
            "sample_rate": opts.sample_rate,
        },
        "language": opts.language.language if opts.language else None,
    }

    if opts.pronunciation_dict_id:
        options["pronunciation_dict_id"] = opts.pronunciation_dict_id

    if opts.api_version > API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS and _is_sonic_3(
        opts.model
    ):
        generation_config: dict[str, Any] = {}
        if opts.speed:
            generation_config["speed"] = opts.speed
        if opts.emotion:
            generation_config["emotion"] = opts.emotion[0]
        if opts.volume:
            generation_config["volume"] = opts.volume
        if generation_config:
            options["generation_config"] = generation_config

    if streaming:
        options["add_timestamps"] = opts.word_timestamps

    return options
