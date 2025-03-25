from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass
from urllib.parse import urlencode

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
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

BASE_URL = "https://api.deepgram.com/v1/speak"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: str
    encoding: str
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = "aura-asteria-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Deepgram TTS.

        Args:
            model (str): TTS model to use. Defaults to "aura-asteria-en".
            encoding (str): Audio encoding to use. Defaults to "linear16".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            api_key (str): Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY in environment.
            base_url (str): Base URL for Deepgram TTS API. Defaults to "https://api.deepgram.com/v1/speak"
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.

        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key if is_given(api_key) else os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY or provide api_key.")

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(  # noqa: B008
                ignore_punctuation=False
            )

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
        self._base_url = base_url
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        config = {
            "encoding": self._opts.encoding,
            "model": self._opts.model,
            "sample_rate": self._opts.sample_rate,
        }
        return await asyncio.wait_for(
            session.ws_connect(
                _to_deepgram_url(config, self._base_url, websocket=True),
                headers={"Authorization": f"Token {self._api_key}"},
            ),
            self._conn_options.timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        args:
            model (str): TTS model to use.
            sample_rate (int): Sample rate of audio.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        for stream in self._streams:
            stream.update_options(
                model=model,
                sample_rate=sample_rate,
            )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            base_url=self._base_url,
            api_key=self._api_key,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            base_url=self._base_url,
            api_key=self._api_key,
            opts=self._opts,
            session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        base_url: str,
        api_key: str,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._base_url = base_url
        self._api_key = api_key

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        try:
            config = {
                "encoding": self._opts.encoding,
                "model": self._opts.model,
                "sample_rate": self._opts.sample_rate,
            }
            async with self._session.post(
                _to_deepgram_url(config, self._base_url, websocket=False),
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": self._input_text},
                timeout=self._conn_options.timeout,
            ) as res:
                if res.status != 200:
                    raise APIStatusError(
                        message=res.reason or "Unknown error occurred.",
                        status_code=res.status,
                        request_id=request_id,
                        body=await res.json(),
                    )

                async for bytes_data, _ in res.content.iter_chunks():
                    for frame in audio_bstream.write(bytes_data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )

                for frame in audio_bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        base_url: str,
        api_key: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._base_url = base_url
        self._api_key = api_key
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None
            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run_segments(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            async for word_stream in self._segments_ch:
                async for word in word_stream:
                    speak_msg = {"type": "Speak", "text": f"{word.token} "}
                    self._mark_started()
                    await ws.send_str(json.dumps(speak_msg))

                # Always flush after a segment
                flush_msg = {"type": "Flush"}
                await ws.send_str(json.dumps(flush_msg))

            # after all segments, close
            close_msg = {"type": "Close"}
            closing_ws = True
            await ws.send_str(json.dumps(close_msg))

        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not closing_ws:
                        raise APIStatusError(
                            "Deepgram websocket connection closed unexpectedly",
                            request_id=request_id,
                        )
                    return

                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    for frame in audio_bstream.write(data):
                        emitter.push(frame)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    resp = json.loads(msg.data)
                    mtype = resp.get("type")
                    if mtype == "Flushed":
                        for frame in audio_bstream.flush():
                            emitter.push(frame)
                        emitter.flush()
                        break
                    elif mtype == "Warning":
                        logger.warning("Deepgram warning: %s", resp.get("warn_msg"))
                    elif mtype == "Metadata":
                        pass
                    else:
                        logger.debug("Unknown message type: %s", resp)

        async def _connection_timeout():
            # Deepgram has a 60-minute timeout period for websocket connections
            await asyncio.sleep(3300)
            logger.warning("Deepgram TTS maximum connection time reached. Reconnecting...")
            self._reconnect_event.set()

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                config = {
                    "encoding": self._opts.encoding,
                    "model": self._opts.model,
                    "sample_rate": self._opts.sample_rate,
                }
                ws = await asyncio.wait_for(
                    self._session.ws_connect(
                        _to_deepgram_url(config, self._base_url, websocket=True),
                        headers={"Authorization": f"Token {self._api_key}"},
                    ),
                    self._conn_options.timeout,
                )
                closing_ws = False

                tasks = [
                    asyncio.create_task(_tokenize_input()),
                    asyncio.create_task(_run_segments(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                connection_timeout_task = asyncio.create_task(_connection_timeout())

                try:
                    done, _ = await asyncio.wait(
                        [
                            asyncio.gather(*tasks),
                            wait_reconnect_task,
                            connection_timeout_task,
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore
                    if wait_reconnect_task not in done:
                        break
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(
                        *tasks, wait_reconnect_task, connection_timeout_task
                    )

            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=request_id,
                    body=None,
                ) from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                if ws is not None and not ws.closed:
                    await ws.close()


def _to_deepgram_url(
    opts: dict,
    base_url: str,
    *,
    websocket: bool,
) -> str:
    if websocket and base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    elif not websocket and base_url.startswith("ws"):
        base_url = base_url.replace("ws", "http", 1)

    return f"{base_url}?{urlencode(opts, doseq=True)}"
