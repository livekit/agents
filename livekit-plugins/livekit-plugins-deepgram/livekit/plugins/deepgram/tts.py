from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace

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
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ._utils import _to_deepgram_url
from .log import logger
from .models import TTSModels

BASE_URL = "https://api.deepgram.com/v1/speak"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: str
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer
    base_url: str
    mip_opt_out: bool = False
    bit_rate: int | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "aura-2-andromeda-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        bit_rate: int | None = None,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        mip_opt_out: bool = False,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
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
            api_key (str): Deepgram API key. If not provided, will look for DEEPGRAM_API_KEY in environment.
            base_url (str): Base URL for Deepgram TTS API. Defaults to "https://api.deepgram.com/v1/speak"
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
            extra_headers: Additional HTTP headers sent on every connection, merged over the
                default ``Authorization: Token`` header. When no API key is set, these become
                the sole auth (e.g. the Cloudflare AI Gateway).

        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        # only fall back to the env var when no api_key was passed at all; an explicit "" means
        # "no Deepgram key" (e.g. with_cloudflare), so it must not pick up DEEPGRAM_API_KEY
        if api_key is None:
            api_key = os.environ.get("DEEPGRAM_API_KEY")
        extra = dict(extra_headers) if is_given(extra_headers) else {}
        if not api_key and not extra:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY or provide api_key.")
        # default Token auth only when a key is present; extra_headers merge on top (and are
        # the sole auth when no key is set, e.g. the Cloudflare AI Gateway)
        self._connect_headers = ({"Authorization": f"Token {api_key}"} if api_key else {}) | extra

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            word_tokenizer=word_tokenizer,
            base_url=base_url,
            mip_opt_out=mip_opt_out,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
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

    @staticmethod
    def with_cloudflare(
        *,
        model: str = "aura-1",
        account_id: str | None = None,
        gateway_id: str = "default",
        cf_aig_token: str | None = None,
        base_url: str | None = None,
        encoding: str = "linear16",
        sample_rate: int = 24000,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> TTS:
        """Create a Deepgram TTS routed through the Cloudflare AI Gateway.

        Connects to the gateway's ``workers-ai`` WebSocket, which proxies Deepgram's
        streaming protocol. Auth uses the ``cf-aig-authorization`` header; no Deepgram
        API key is required.

        Args:
            model: Deepgram model name (e.g. ``"aura-1"``); the ``@cf/deepgram/`` prefix is
                added automatically. A value already prefixed with ``@cf/`` is used as-is.
            account_id: Cloudflare account ID. Falls back to ``CLOUDFLARE_ACCOUNT_ID``.
                Required unless ``base_url`` is given.
            gateway_id: Gateway name. Defaults to ``"default"``.
            cf_aig_token: Gateway token for ``cf-aig-authorization``. Falls back to
                ``CLOUDFLARE_AI_GATEWAY_TOKEN``.
            base_url: Full gateway endpoint; overrides ``account_id`` / ``gateway_id``.
            encoding: Audio encoding, forwarded to ``TTS``. Defaults to ``"linear16"``.
            sample_rate: Audio sample rate in Hz, forwarded to ``TTS``.
            word_tokenizer: Optional tokenizer, forwarded to ``TTS``.
            http_session: Optional aiohttp session, forwarded to ``TTS``.
        """
        cf_aig_token = cf_aig_token or os.environ.get("CLOUDFLARE_AI_GATEWAY_TOKEN")
        if not cf_aig_token:
            raise ValueError(
                "Cloudflare AI Gateway token is required, either as argument or set"
                " CLOUDFLARE_AI_GATEWAY_TOKEN environment variable"
            )
        if base_url is None:
            account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                raise ValueError(
                    "Cloudflare account_id is required, either as argument or set"
                    " CLOUDFLARE_ACCOUNT_ID environment variable (or pass base_url directly)"
                )
            base_url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai"

        if not model.startswith("@cf/"):
            model = f"@cf/deepgram/{model}"

        return TTS(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            base_url=base_url,
            word_tokenizer=word_tokenizer,
            http_session=http_session,
            # explicit empty key opts out of the DEEPGRAM_API_KEY env fallback, so the gateway
            # only ever receives cf-aig-authorization (no stray Authorization: Token header)
            api_key="",
            extra_headers={"cf-aig-authorization": cf_aig_token},
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        config: dict = {
            "encoding": self._opts.encoding,
            "model": self._opts.model,
            "sample_rate": self._opts.sample_rate,
            "mip_opt_out": self._opts.mip_opt_out,
        }
        if self._opts.bit_rate is not None:
            config["bit_rate"] = self._opts.bit_rate
        ws = await asyncio.wait_for(
            session.ws_connect(
                _to_deepgram_url(config, self._opts.base_url, websocket=True),
                headers=self._connect_headers,
            ),
            timeout,
        )
        ws_headers = {
            k: v for k, v in ws._response.headers.items() if k.startswith("dg-") or k == "Date"
        }
        logger.debug(
            "Established new Deepgram TTS WebSocket connection:",
            extra={"headers": ws_headers},
        )

        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            # Send Flush and Close messages to ensure Deepgram processes all remaining audio
            # and properly terminates the session, preventing lingering TTS sessions
            await ws.send_str(SynthesizeStream._FLUSH_MSG)
            await ws.send_str(SynthesizeStream._CLOSE_MSG)

            # Wait for server acknowledgment to prevent race conditions and ensure
            # proper cleanup, avoiding 429 Too Many Requests errors from lingering sessions
            try:
                await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            logger.warning(f"Error during WebSocket close sequence: {e}")
        finally:
            await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model (TTSModels | str): TTS model to use.
            encoding (str): Audio encoding to use.
            sample_rate (int): Sample rate of audio in Hz.
            bit_rate (int | None): Bit rate for compressed encodings (e.g. mp3).
                See https://developers.deepgram.com/reference/text-to-speech-api#query-bit_rate
        """
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

        if connection_params_changed:
            # These params are baked into the WebSocket URL at connection time, so any
            # existing pooled connection must be invalidated to avoid serving audio at
            # the wrong rate/encoding.
            self._pool.invalidate()

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

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            http_params: dict = {
                "encoding": self._opts.encoding,
                "container": "none",
                "model": self._opts.model,
                "sample_rate": self._opts.sample_rate,
                "mip_opt_out": self._opts.mip_opt_out,
            }
            if self._opts.bit_rate is not None:
                http_params["bit_rate"] = self._opts.bit_rate
            async with self._tts._ensure_session().post(
                _to_deepgram_url(http_params, self._opts.base_url, websocket=False),
                headers={
                    **self._tts._connect_headers,
                    "Content-Type": "application/json",
                },
                json={"text": self._input_text},
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
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
    _FLUSH_MSG: str = json.dumps({"type": "Flush"})
    _CLOSE_MSG: str = json.dumps({"type": "Close"})

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[tokenize.WordStream]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into segments_ch
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for word in word_stream:
                speak_msg = {"type": "Speak", "text": f"{word.token} "}
                self._mark_started()
                await ws.send_str(json.dumps(speak_msg))
                input_sent_event.set()

            # always flush after a segment
            flush_msg = {"type": "Flush"}
            await ws.send_str(json.dumps(flush_msg))
            input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Deepgram websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
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
                        logger.warning("Unknown Deepgram message type: %s", resp)

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                input_sent_event.set()
                await utils.aio.gracefully_cancel(*tasks)
