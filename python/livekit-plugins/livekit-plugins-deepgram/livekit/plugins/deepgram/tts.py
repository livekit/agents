from __future__ import annotations

import asyncio
import json
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ._utils import to_deepgram_url
from .log import logger
from .models import TTSModels

BASE_URL = "https://api.deepgram.com/v1/speak"
NUM_CHANNELS = 1
# Speaking rate bounds accepted by the Deepgram TTS API; values outside are rejected
# with a 400, which on the WebSocket path surfaces only as a failed handshake.
MIN_SPEED = 0.7
MAX_SPEED = 1.5


_FLUSH_MSG = json.dumps({"type": "Flush"})
_CLOSE_MSG = json.dumps({"type": "Close"})


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: str
    sample_rate: int
    base_url: str
    api_key: str
    mip_opt_out: bool = False
    bit_rate: int | None = None
    speed: float | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "aura-2-andromeda-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        bit_rate: int | None = None,
        speed: float | None = None,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: http.HttpSession | None = None,
        mip_opt_out: bool = False,
    ) -> None:
        """
        Create a new instance of Deepgram TTS.

        Args:
            model (TTSModels | str): TTS model to use. Defaults to "aura-2-andromeda-en".
                See https://developers.deepgram.com/docs/tts-models for available models.
            encoding (str): Audio encoding to use. Defaults to "linear16".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3). Defaults to None.
                See https://developers.deepgram.com/reference/text-to-speech-api#query-bit_rate
            speed (float | None): Speaking rate, from 0.7 to 1.5; 1.0 is the normal rate. Defaults to None,
                which leaves it out of the request. Deepgram supports it on Aura-2 English and Spanish
                voices and rejects the request for other models.
                See https://developers.deepgram.com/docs/tts-voice-controls
            api_key (str): Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY in environment.
            base_url (str): Base URL for Deepgram TTS API. Defaults to "https://api.deepgram.com/v1/speak"
            http_session (http.HttpSession): Optional HTTP session to use for requests.

        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY or provide api_key.")

        if speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}")

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            speed=speed,
            base_url=base_url,
            api_key=api_key,
            mip_opt_out=mip_opt_out,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepgram"

    def _core_text_unit(self) -> str:
        return "words"

    def _core_max_text_len(self) -> int | None:
        return 2000

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        session = self._ensure_session()
        config: dict[str, str | float | int | bool] = {
            "encoding": self._opts.encoding,
            "model": self._opts.model,
            "sample_rate": self._opts.sample_rate,
            "mip_opt_out": self._opts.mip_opt_out,
        }
        if self._opts.bit_rate is not None:
            config["bit_rate"] = self._opts.bit_rate
        if self._opts.speed is not None:
            config["speed"] = self._opts.speed
        ws = await session.ws_connect(
            to_deepgram_url(config, self._opts.base_url, websocket=True),
            headers={"Authorization": f"Token {self._opts.api_key}"},
            conn_options=conn_options,
        )
        ws_headers = {k: v for k, v in ws.headers.items() if k.startswith("dg-") or k == "Date"}
        logger.debug(
            "Established new Deepgram TTS WebSocket connection:",
            extra={"headers": ws_headers},
        )

        return ws

    async def _close_ws(self, ws: http.WebSocket) -> None:
        try:
            # Send Flush and Close messages to ensure Deepgram processes all remaining audio
            # and properly terminates the session, preventing lingering TTS sessions
            await ws.send_str(_FLUSH_MSG)
            await ws.send_str(_CLOSE_MSG)

            # Wait for server acknowledgment to prevent race conditions and ensure
            # proper cleanup, avoiding 429 Too Many Requests errors from lingering sessions
            try:
                await ws.receive(timeout=1.0)
            except APITimeoutError:
                pass
        except Exception as e:
            logger.warning(f"Error during WebSocket close sequence: {e}")
        finally:
            await ws.close()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model (TTSModels | str): TTS model to use.
            encoding (str): Audio encoding to use.
            sample_rate (int): Sample rate of audio in Hz.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3).
                See https://developers.deepgram.com/reference/text-to-speech-api#query-bit_rate
            speed (float | None): Speaking rate, from 0.7 to 1.5; None leaves it out of the request.
                See https://developers.deepgram.com/docs/tts-voice-controls
            sentence_tokenizer (tokenize.SentenceTokenizer): What splits the text into the chunks
                the TTS is sent.
            pacing (tts.SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send
                each chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        if is_given(speed) and speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, but got {speed}")

        connection_params_changed = False
        if is_given(model):
            self._opts.model = model
            connection_params_changed = True
        if is_given(encoding):
            self._opts.encoding = encoding
            connection_params_changed = True
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._sample_rate = sample_rate  # keep base class property in sync
            connection_params_changed = True
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
            connection_params_changed = True
        if is_given(speed):
            self._opts.speed = speed
            connection_params_changed = True

        if connection_params_changed:
            # These params are baked into the WebSocket URL at connection time, so any
            # existing pooled connection must be invalidated to avoid serving audio at
            # the wrong rate/encoding/speed.
            self._pool.invalidate()

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
            pool=self._pool, opts=replace(self._opts), tts=self, conn_options=conn_options
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


class ChunkedStream(tts.ChunkedStream):
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
        try:
            http_params: dict[str, str | float | int | bool] = {
                "encoding": self._opts.encoding,
                "container": "none",
                "model": self._opts.model,
                "sample_rate": self._opts.sample_rate,
                "mip_opt_out": self._opts.mip_opt_out,
            }
            if self._opts.bit_rate is not None:
                http_params["bit_rate"] = self._opts.bit_rate
            if self._opts.speed is not None:
                http_params["speed"] = self._opts.speed
            async with self._get_session().post(
                to_deepgram_url(http_params, self._opts.base_url, websocket=False),
                headers={
                    "Authorization": f"Token {self._opts.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": self._input_text},
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                if (billed := resp.headers.get("dg-char-count")) is not None:
                    output_emitter.report_usage(characters_count=int(billed))
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
        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _segment_input() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for input in self.input:
                if isinstance(input, tts.SpeechChunk):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments_ch.send_nowait(segment)
                    segment.send_nowait(input.text)
                elif segment is not None:
                    segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            segments_ch.close()

        async def _run_segments() -> None:
            async for segment in segments_ch:
                await self._run_ws(segment, output_emitter)

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(_segment_input()), tasks.spawn(_run_segments()))
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()

        async def send_task(ws: http.WebSocket) -> None:
            async for chunk in segment:
                speak_msg = {"type": "Speak", "text": f"{chunk} "}
                self.mark_started()
                await ws.send_str(json.dumps(speak_msg))
                input_sent_event.set()

            # always flush after a segment
            flush_msg = {"type": "Flush"}
            await ws.send_str(json.dumps(flush_msg))
            input_sent_event.set()

        async def recv_task(ws: http.WebSocket) -> None:
            await input_sent_event.wait()
            while True:
                msg = await ws.receive(timeout=self._conn_options.read_timeout)
                if msg.type == http.WebSocketMessageType.CLOSE:
                    raise APIStatusError(
                        "Deepgram websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=}",
                    )

                if msg.type == http.WebSocketMessageType.BINARY:
                    output_emitter.push(msg.binary)
                elif msg.type == http.WebSocketMessageType.TEXT:
                    resp = json.loads(msg.data)
                    mtype = resp.get("type")
                    if mtype == "Flushed":
                        output_emitter.end_segment()
                        break
                    elif mtype == "Warning":
                        logger.warning("Deepgram warning: %s", resp.get("warn_msg"))
                    elif mtype in ("Error", "error"):
                        raise APIError(message="Deepgram TTS returned error", body=resp)
                    elif mtype == "Metadata":
                        pass
                    else:
                        logger.warning("Unknown Deepgram message type", extra={"lk.pii.data": resp})

        async with self._pool.connection(conn_options=self._conn_options) as ws:
            async with utils.aio.TaskSet() as tasks:
                try:
                    await asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
                finally:
                    input_sent_event.set()
