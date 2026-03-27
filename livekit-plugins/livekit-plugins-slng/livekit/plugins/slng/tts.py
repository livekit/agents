from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Literal

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

from .gateway_adapter import build_tts_init_payload, is_rime_arcana_model, normalize_region_override
from .log import logger

NUM_CHANNELS = 1


def _default_tts_endpoint(*, slng_base_url: str, model: str) -> str:
    protocol = "ws" if "localhost" in slng_base_url or "127.0.0.1" in slng_base_url else "wss"
    return f"{protocol}://{slng_base_url}/v1/tts/{model}"


@dataclass
class _TTSOptions:
    model_endpoint: str
    model: str
    voice: str
    language: str
    sample_rate: int
    encoding: Literal["linear16"]
    speed: float
    word_tokenizer: tokenize.WordTokenizer
    api_key: str
    model_options: dict[str, object]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepgram/aura:2",
        model_endpoint: str | None = None,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        voice: str = "default",
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        **model_options: object,
    ) -> None:
        """
        Create a new instance of SLNG TTS (based on Deepgram's architecture).

        Args:
            model (str): SLNG model identifier (e.g., "deepgram/aura:2").
            model_endpoint (str): Optional full SLNG WebSocket endpoint.
            slng_base_url (str): SLNG gateway host. Defaults to "api.slng.ai".
            region_override (str | list[str] | None): Optional gateway region override.
            voice (str): Voice to use. Defaults to "default".
            language (str): Language code. Defaults to "en".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            api_key (str): SLNG API key. Falls back to SLNG_API_KEY environment variable.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
        """
        # Resolve api_key from parameter or SLNG_API_KEY env var
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set SLNG_API_KEY environment variable")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        resolved_model_endpoint = model_endpoint or _default_tts_endpoint(
            slng_base_url=slng_base_url,
            model=model,
        )

        self._opts = _TTSOptions(
            model_endpoint=resolved_model_endpoint,
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            # LiveKit expects raw PCM. Some SLNG models default to MP3 unless explicitly requested.
            encoding="linear16",
            speed=speed,
            word_tokenizer=word_tokenizer,
            api_key=resolved_key,
            model_options=dict(model_options),
        )
        self._region_override_header = normalize_region_override(region_override)
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
        return "slng"

    @property
    def provider(self) -> str:
        return "SLNG"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()

        # Connect to WebSocket
        model_endpoint = self._opts.model_endpoint
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "X-API-Key": self._opts.api_key,
        }
        if self._region_override_header:
            headers["X-Region-Override"] = self._region_override_header
        ws = await asyncio.wait_for(
            session.ws_connect(
                model_endpoint,
                headers=headers,
            ),
            timeout,
        )

        # SLNG-specific: Send init and wait for ready
        init_payload = build_tts_init_payload(
            model=self._opts.model,
            voice=self._opts.voice,
            language=self._opts.language,
            sample_rate=self._opts.sample_rate,
            encoding=self._opts.encoding,
            speed=self._opts.speed,
            model_options=self._opts.model_options,
        )

        await ws.send_str(json.dumps(init_payload))

        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            # Send final flush (similar to Deepgram's Flush+Close pattern).
            # Arcana-specific cancel/EOS is handled in the streaming send_task when bypassing
            # the connection pool.
            await ws.send_str(SynthesizeStream._FLUSH_MSG)

            # Wait for server acknowledgment
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ws.receive(), timeout=5.0)
        except Exception as e:
            logger.warning(f"[SLNG TTS] error during WebSocket close sequence: {e}")
        finally:
            await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice to use.
            language (str): Language code.
        """
        invalidate_pool = False
        if is_given(voice):
            invalidate_pool = invalidate_pool or self._opts.voice != voice
            self._opts.voice = voice
        if is_given(language):
            invalidate_pool = invalidate_pool or self._opts.language != language
            self._opts.language = language

        if invalidate_pool:
            self._pool.invalidate()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
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
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        is_rime_arcana = is_rime_arcana_model(self._opts.model)
        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await self._tts._connect_ws(timeout=self._conn_options.timeout)
            if self._input_text:
                await ws.send_str(json.dumps({"type": "text", "text": self._input_text}))

            if is_rime_arcana:
                await ws.send_str(SynthesizeStream._CANCEL_MSG)
            else:
                await ws.send_str(SynthesizeStream._FLUSH_MSG)

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if is_rime_arcana:
                        output_emitter.flush()
                        break
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    resp = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug("[SLNG TTS] ignoring non-JSON text frame: %s", msg.data)
                    continue

                if not isinstance(resp, dict):
                    continue

                if "type" not in resp:
                    is_final_value = resp.get("isFinal")
                    is_final = (
                        is_final_value is True
                        or is_final_value == 1
                        or (
                            isinstance(is_final_value, str)
                            and is_final_value.strip().lower() in ("true", "1")
                        )
                    )
                    audio_b64 = resp.get("audio")
                    if isinstance(audio_b64, str) and audio_b64:
                        try:
                            output_emitter.push(base64.b64decode(audio_b64))
                        except Exception:
                            logger.warning(
                                "[SLNG TTS] invalid base64 audio in chunked synthesis",
                                exc_info=True,
                            )

                    if is_final:
                        output_emitter.flush()
                        break

                    if resp.get("error") is not None:
                        raise APIStatusError(f"SLNG TTS error: {resp.get('error')}")
                    continue

                mtype = resp.get("type")
                if mtype in ("Metadata", "Open", "control_ack", "ready"):
                    continue
                if mtype == "Flushed":
                    mtype = "audio_end"
                elif mtype in ("Audio", "chunk"):
                    mtype = "audio_chunk"
                elif mtype == "Error":
                    mtype = "error"

                if mtype == "audio_chunk":
                    audio_b64 = resp.get("data")
                    if not audio_b64 and isinstance(resp.get("audio"), str):
                        audio_b64 = resp.get("audio")
                    if isinstance(audio_b64, str) and audio_b64:
                        try:
                            output_emitter.push(base64.b64decode(audio_b64))
                        except Exception:
                            logger.warning(
                                "[SLNG TTS] invalid base64 audio (audio_chunk)",
                                exc_info=True,
                            )
                elif mtype in ("audio_end", "end", "flushed"):
                    output_emitter.flush()
                    break
                elif mtype == "error":
                    error_msg = (
                        resp.get("message")
                        or resp.get("description")
                        or resp.get("error")
                        or "Unknown error"
                    )
                    raise APIStatusError(f"SLNG TTS error: {error_msg}")
                else:
                    logger.debug("[SLNG TTS] ignoring unknown message: %s", resp)
        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                await ws.close()


class SynthesizeStream(tts.SynthesizeStream):
    # SLNG protocol messages (different from Deepgram)
    _FLUSH_MSG: str = json.dumps({"type": "flush"})
    _CANCEL_MSG: str = json.dumps({"type": "cancel"})

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for the_input in self._input_ch:
                if isinstance(the_input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(the_input)
                elif isinstance(the_input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except APIStatusError:
            raise
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
        is_rime_arcana = is_rime_arcana_model(self._opts.model)

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for word in word_stream:
                # SLNG: Use "text" type instead of "Speak"
                text_msg = {"type": "text", "text": f"{word.token} "}
                self._mark_started()
                await ws.send_str(json.dumps(text_msg))
                input_sent_event.set()

            if is_rime_arcana:
                # Rime Arcana supports EOS: synthesize remaining buffer and then close the socket.
                # The gateway maps SLNG "cancel" -> "<EOS>" for Arcana.
                await ws.send_str(self._CANCEL_MSG)
            else:
                # Always flush after a segment
                # SLNG: Use "flush" type instead of "Flush"
                await ws.send_str(self._FLUSH_MSG)
            input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # Rime Arcana EOS closes the socket after sending the final audio.
                    # Treat that close as a normal end-of-segment, otherwise LiveKit will error.
                    if is_rime_arcana:
                        output_emitter.end_segment()
                        break
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                # SLNG: Handle both binary (legacy) and JSON audio_chunk messages
                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        resp = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.debug("[SLNG TTS] ignoring non-JSON text frame: %s", msg.data)
                        continue

                    if not isinstance(resp, dict):
                        continue

                    if "type" not in resp:
                        is_final_value = resp.get("isFinal")
                        is_final = (
                            is_final_value is True
                            or is_final_value == 1
                            or (
                                isinstance(is_final_value, str)
                                and is_final_value.strip().lower() in ("true", "1")
                            )
                        )
                        audio_b64 = resp.get("audio")
                        if isinstance(audio_b64, str) and audio_b64:
                            try:
                                output_emitter.push(base64.b64decode(audio_b64))
                            except Exception:
                                if is_final:
                                    logger.warning(
                                        "[SLNG TTS] invalid base64 audio (isFinal frame)",
                                        exc_info=True,
                                    )
                                else:
                                    logger.warning(
                                        "[SLNG TTS] invalid base64 audio (audio frame)",
                                        exc_info=True,
                                    )

                        if is_final:
                            output_emitter.end_segment()
                            break

                        if resp.get("error") is not None:
                            raise APIStatusError(f"SLNG TTS error: {resp.get('error')}")
                        continue

                    mtype = resp.get("type")
                    if mtype in ("Metadata", "Open"):
                        continue
                    if mtype == "Flushed":
                        mtype = "audio_end"
                    elif mtype in ("Audio", "chunk"):
                        mtype = "audio_chunk"
                    elif mtype == "Error":
                        mtype = "error"

                    # SLNG: Handle audio_chunk with base64 data
                    if mtype == "audio_chunk":
                        audio_b64 = resp.get("data")
                        if not audio_b64 and isinstance(resp.get("audio"), str):
                            audio_b64 = resp.get("audio")
                        if isinstance(audio_b64, str) and audio_b64:
                            try:
                                audio_data = base64.b64decode(audio_b64)
                            except Exception:
                                logger.warning(
                                    "[SLNG TTS] invalid base64 audio (audio_chunk)",
                                    exc_info=True,
                                )
                            else:
                                output_emitter.push(audio_data)

                    # SLNG: "audio_end" or "end" instead of "Flushed"
                    elif mtype in ("audio_end", "end", "flushed"):
                        output_emitter.end_segment()
                        break

                    elif mtype == "error":
                        error_msg = (
                            resp.get("message")
                            or resp.get("description")
                            or resp.get("error")
                            or "Unknown error"
                        )
                        raise APIStatusError(f"SLNG TTS error: {error_msg}")

                    elif mtype in ("control_ack", "ready"):
                        # Informational messages, ignore
                        pass

                    else:
                        logger.debug("[SLNG TTS] ignoring unknown message: %s", resp)

        if is_rime_arcana:
            # EOS closes the websocket; avoid pooling to prevent any cross-segment leakage or reuse
            # of a server-closed socket.
            ws = await self._tts._connect_ws(timeout=self._conn_options.timeout)
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                input_sent_event.set()
                await utils.aio.gracefully_cancel(*tasks)
                await ws.close()
        else:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await utils.aio.gracefully_cancel(*tasks)
