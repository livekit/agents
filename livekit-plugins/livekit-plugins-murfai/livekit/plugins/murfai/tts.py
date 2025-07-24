from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSDefaultVoiceStyle,
    TTSEncoding,
    TTSLocales,
    TTSModels,
    TTSStyles,
)

API_AUTH_HEADER = "api-key"
BUFFERED_WORDS_COUNT = 10


@dataclass
class _TTSOptions:
    api_key: str
    locale: TTSLocales | str | None = None
    model: TTSModels | str = "GEN2"
    voice: str = TTSDefaultVoiceId
    style: str | None = TTSDefaultVoiceStyle
    speed: int | None = None
    pitch: int | None = None
    sample_rate: int = 44100
    encoding: TTSEncoding | str = "pcm"
    base_url: str = "https://api.murf.ai"

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "GEN2",
        locale: TTSLocales | str | None = None,
        voice: str = TTSDefaultVoiceId,
        style: TTSStyles | str | None = None,
        speed: int | None = None,
        pitch: int | None = None,
        sample_rate: int = 44100,
        encoding: TTSEncoding | str = "pcm",
        base_url: str = "https://api.murf.ai",
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Murf AI TTS.

        See https://murf.ai/api/docs/api-reference/text-to-speech/stream-input for more details on the the Murf AI API.

        Args:
            api_key (str | None, optional): The Murf AI API key. If not provided, it will be read from the MURFAI_API_KEY environment variable.
            model (TTSModels | str, optional): The Murf AI TTS model to use. Defaults to "GEN2".
            locale (str | None, optional): The locale for synthesis (e.g., "en-US", "en-UK"). If not provided, will be inferred from voice.
            voice (str, optional): The voice ID from Murf AI's voice library (e.g., "en-US-amara"). Defaults to TTSDefaultVoiceId.
            style (TTSStyles | str | None, optional): The voice style to apply (e.g., "Conversational"). Can be None for default style.
            speed (int | None, optional): The speech speed control. Higher values = faster speech. None for default speed.
            pitch (int | None, optional): The speech pitch control. Higher values = higher pitch. None for default pitch.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 44100.
            encoding (str, optional): The audio encoding format. Defaults to "pcm".
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Murf AI API. Defaults to "https://api.murf.ai".
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use. Defaults to tokenize.basic.SentenceTokenizer(min_sentence_len=BUFFERED_WORDS_COUNT).
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        murf_api_key = api_key or os.environ.get("MURFAI_API_KEY")
        if not murf_api_key:
            raise ValueError("MURFAI_API_KEY must be set")

        self._opts = _TTSOptions(
            api_key=murf_api_key,
            model=model,
            locale=locale,
            voice=voice,
            style=style or TTSDefaultVoiceStyle,
            speed=speed,
            pitch=pitch,
            sample_rate=sample_rate,
            encoding=encoding,
            base_url=base_url,
        )
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

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(
            f"/v1/speech/stream-input?api-key={self._opts.api_key}&sample_rate={self._opts.sample_rate}&format={self._opts.encoding}"
        )
        return await asyncio.wait_for(session.ws_connect(url), timeout)

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
        locale: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        style: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[int | None] = NOT_GIVEN,
        pitch: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model, locale, voice, style,
        speed and pitch. If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModels | str, optional): The Murf AI TTS model to use.
            locale (str, optional): The locale for synthesis (e.g., "en-US", "en-UK").
            voice (str, optional): The voice ID from Murf AI's voice library. (e.g. "en-US-amara")
            style (str | None, optional): The voice style to apply (e.g., "Conversational", "Calm").
            speed (int | None, optional): Controls the speech speed. Positive values increase speed, negative values decrease it. Valid range: -50 to 50.
            pitch (int | None, optional): Controls the speech pitch. Positive values raise pitch, negative values lower it. Valid range: -50 to 50.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(locale):
            self._opts.locale = locale
        if is_given(voice):
            self._opts.voice = voice
        if is_given(style):
            self._opts.style = style
        if is_given(speed):
            self._opts.speed = speed
        if is_given(pitch):
            self._opts.pitch = pitch

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the http streaming output endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                self._opts.get_http_url("/v1/speech/stream"),
                headers={API_AUTH_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model_version": self._opts.model,
                    "multiNativeLocale": self._opts.locale,
                    "voice_id": self._opts.voice,
                    "style": self._opts.style,
                    "rate": self._opts.speed,
                    "pitch": self._opts.pitch,
                    "format": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                },
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
        self._sent_tokenizer_stream = tts._sentence_tokenizer.stream()
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

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            context_id = utils.shortuuid()
            base_pkt = _to_murf_websocket_pkt(self._opts)
            async for ev in self._sent_tokenizer_stream:
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = context_id
                token_pkt["text"] = ev.token + " "
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = context_id
            end_pkt["end"] = True
            await ws.send_str(json.dumps(end_pkt))

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue

                self._sent_tokenizer_stream.push_text(data)

            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            current_segment_id: str | None = None
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Murf AI connection closed unexpectedly", request_id=request_id
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Murf AI message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                if current_segment_id is None:
                    current_segment_id = segment_id
                    output_emitter.start_segment(segment_id=current_segment_id)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    output_emitter.push(b64data)
                elif data.get("final"):
                    output_emitter.end_input()
                    break
                else:
                    logger.warning("unexpected message %s", data)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _to_murf_websocket_pkt(opts: _TTSOptions) -> dict[str, Any]:
    voice_config: dict[str, Any] = {}

    if opts.voice:
        voice_config["voice_id"] = opts.voice

    if opts.style:
        voice_config["style"] = opts.style

    if opts.speed:
        voice_config["rate"] = opts.speed

    if opts.pitch:
        voice_config["pitch"] = opts.pitch

    if opts.locale:
        voice_config["multi_native_locale"] = opts.locale

    return {
        "voice_config": voice_config,
    }
