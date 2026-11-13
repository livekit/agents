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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from .constants import (
    API_AUTH_HEADER,
    API_VERSION,
    API_VERSION_HEADER,
    API_VERSION_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    AUDIO_ENCODING,
    MODEL_ID_WITH_EMBEDDINGS_AND_EXPERIMENTAL_CONTROLS,
    REQUEST_ID_HEADER,
    USER_AGENT,
)
from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSModels,
    TTSVoiceEmotion,
    TTSVoiceSpeed,
)


def _is_sonic_3(model: str) -> bool:
    return model.startswith("sonic-3")


@dataclass
class _TTSOptions:
    model: TTSModels | str
    sample_rate: int
    voice: str | list[float]
    speed: TTSVoiceSpeed | float | None
    emotion: list[TTSVoiceEmotion | str] | None
    volume: float | None
    word_timestamps: bool
    api_key: str
    language: LanguageCode | None
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
        voice: str | list[float] = TTSDefaultVoiceId,
        speed: TTSVoiceSpeed | float | None = None,
        emotion: TTSVoiceEmotion | str | list[TTSVoiceEmotion | str] | None = None,
        volume: float | None = None,
        sample_rate: int = 24000,
        word_timestamps: bool = True,
        pronunciation_dict_id: str | None = None,
        http_session: http.HttpSession | None = None,
        base_url: str = "https://api.cartesia.ai",
        api_version: str = API_VERSION,
    ) -> None:
        """
        Create a new instance of Cartesia TTS.

        See https://docs.cartesia.ai/reference/web-socket/stream-speech/stream-speech for more details on the Cartesia API.

        Args:
            model (TTSModels, optional): The Cartesia TTS model to use. Defaults to "sonic-3".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (str | list[float], optional): The voice ID or embedding array.
            speed (TTSVoiceSpeed | float, optional): Speed of speech, with sonic-3, the value is valid between 0.6 and 2.0 (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-speed)
            emotion (list[TTSVoiceEmotion], optional): Emotion of the speech (https://docs.cartesia.ai/api-reference/tts/bytes#body-generation-config-emotion)
            volume (float, optional): Volume of the speech, with sonic-3, the value is valid between 0.5 and 2.0
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            word_timestamps (bool, optional): Whether to add word timestamps to the output. Defaults to True.
            pronunciation_dict_id (str, optional): The pronunciation dictionary ID to use for custom pronunciations. Defaults to None.
            api_key (str, optional): The Cartesia API key. If not provided, it will be read from the CARTESIA_API_KEY environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
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
            language=LanguageCode(language) if language else None,
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
        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

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

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            return "cartesia"

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Cartesia"

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        url = self._opts.get_ws_url(f"/tts/websocket?cartesia_version={self._opts.api_version}")
        ws = await self._ensure_session().ws_connect(
            url,
            headers={
                "User-Agent": USER_AGENT,
                API_AUTH_HEADER: self._opts.api_key,
            },
            conn_options=conn_options,
        )
        c_request_id = ws.headers.get(REQUEST_ID_HEADER)
        logger.debug(
            "Established new Cartesia TTS WebSocket connection",
            extra={"cartesia_request_id": c_request_id},
        )
        return ws

    async def _close_ws(self, ws: http.WebSocket) -> None:
        await ws.close()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

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
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
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
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = LanguageCode(language) if language else None
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = cast(TTSVoiceSpeed | float, speed)
        if is_given(emotion):
            emotion = [emotion] if isinstance(emotion, str) else emotion
            self._opts.emotion = emotion
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pronunciation_dict_id):
            self._opts.pronunciation_dict_id = pronunciation_dict_id
        if is_given(api_version) and api_version != self._opts.api_version:
            self._opts.api_version = api_version
            # cartesia_version is a query parameter on the websocket URL, so a pooled
            # socket keeps speaking the version it was opened with while
            # _to_cartesia_options starts shaping the body for the new one.
            self._pool.invalidate()

        if speed or emotion or volume or pronunciation_dict_id:
            self._check_generation_config()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            get_session=self._ensure_session,
            opts=replace(self._opts),
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            pool=self._pool,
            opts=replace(self._opts),
            tts=self,
            conn_options=conn_options,
        )
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

    def __init__(
        self,
        *,
        get_session: Callable[[], http.HttpSession],
        opts: _TTSOptions,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        json = _to_cartesia_options(self._opts, streaming=False)
        json["transcript"] = self._input_text

        try:
            async with self._get_session().post(
                self._opts.get_http_url("/tts/bytes"),
                headers={
                    API_AUTH_HEADER: self._opts.api_key,
                    API_VERSION_HEADER: self._opts.api_version,
                    "User-Agent": USER_AGENT,
                },
                json=json,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        pool: utils.ConnectionPool[http.WebSocket],
        opts: _TTSOptions,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._pool = pool

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
        input_done = False

        async def _send_task(ws: http.WebSocket, cartesia_context_id: str) -> None:
            nonlocal input_done
            base_pkt = _to_cartesia_options(self._opts, streaming=True)
            base_pkt["max_buffer_delay_ms"] = 0
            async for data in self.input:
                if isinstance(data, self.Flush):
                    continue

                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = cartesia_context_id
                token_pkt["transcript"] = data.text + " "
                sent_tokens.append(data.said + " ")
                token_pkt["continue"] = True
                self.mark_started()
                await ws.send_str(json.dumps(token_pkt))
                input_sent_event.set()

            input_done = True
            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = cartesia_context_id
            end_pkt["transcript"] = " "
            sent_tokens.append(" ")
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))
            input_sent_event.set()

        async def _recv_task(ws: http.WebSocket, cartesia_context_id: str) -> None:
            current_segment_id: str | None = None
            await input_sent_event.wait()
            skip_aligning = False
            while True:
                msg = await ws.receive(timeout=self._conn_options.read_timeout)
                if msg.type is http.WebSocketMessageType.CLOSE:
                    logger.error(
                        "Cartesia connection closed unexpectedly. Include the cartesia_context_id to support@cartesia.ai for help debugging.",
                        extra={"cartesia_context_id": cartesia_context_id},
                    )
                    raise APIStatusError(
                        "Cartesia connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=}",
                    )

                if msg.type is not http.WebSocketMessageType.TEXT:
                    logger.warning("unexpected Cartesia message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                # A pooled websocket may still hold audio/done from an interrupted
                # previous context; ignore messages tagged with another context id.
                if segment_id is not None and segment_id != cartesia_context_id:
                    continue
                if current_segment_id is None:
                    current_segment_id = cartesia_context_id
                    output_emitter.start_segment(segment_id=cartesia_context_id)
                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    output_emitter.push(b64data)
                elif data.get("done"):
                    if input_done:
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
                            TranscriptChunk(text=word, start_time=start, end_time=end)
                        )
                elif data.get("type") == "error":
                    logger.error(
                        "Cartesia returned error. Include the cartesia_context_id to support@cartesia.ai for help debugging.",
                        extra={"cartesia_context_id": cartesia_context_id, "error": data},
                    )
                    raise APIError(f"Cartesia returned error: {data}")
                elif data.get("type") == "flush_done":
                    pass
                else:
                    logger.warning("unexpected message", extra={"lk.pii.data": data})

        cartesia_context_id = utils.shortuuid()
        try:
            async with self._pool.connection(conn_options=self._conn_options) as ws:
                async with utils.aio.TaskSet() as tasks:
                    try:
                        await asyncio.gather(
                            tasks.spawn(_send_task(ws, cartesia_context_id)),
                            tasks.spawn(_recv_task(ws, cartesia_context_id)),
                        )
                    finally:
                        input_sent_event.set()
        except APIError:
            raise
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
        voice_controls: dict[str, float | str | list[str]] = {}
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
            "encoding": AUDIO_ENCODING,
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
