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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSLanguages, TTSModels

NUM_CHANNELS = 1
SAMPLE_RATE = 24000
"""Maya synthesizes at 24 kHz. The ``metadata`` frame is authoritative and is
checked against this on connect."""

DEFAULT_BASE_URL = "https://tts.mayaresearch.ai"
DEFAULT_VOICE = "Ananya"


@dataclass
class _TTSOptions:
    model: NotGivenOr[TTSModels | str]
    voice: str
    language: NotGivenOr[TTSLanguages | str]
    api_key: str
    base_url: str
    sample_rate: int

    def get_ws_url(self) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}/v1/tts/stream"

    def start_frame(self) -> dict:
        """The handshake that selects the v2 protocol and sets sticky settings.

        Voice and language hold for the whole connection; Maya rejects codes it
        doesn't know, so both are sent as given rather than filtered here.
        """
        frame: dict = {"type": "start", "v2": True, "voice": self.voice}
        if is_given(self.language):
            frame["language"] = self.language
        if is_given(self.model):
            frame["model"] = self.model
        return frame


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
    ) -> None:
        """Create a new instance of Maya TTS.

        A conversation runs over one persistent websocket, so the handshake is
        paid once rather than per utterance. Each agent turn is a Maya context:
        sentences stream out as they arrive and a single cancel drops the whole
        turn when the user interrupts.

        See https://www.mayaresearch.ai/llm.txt for the API.

        Args:
            voice (str, optional): Voice name, case-sensitive. Every voice speaks
                every language. See Maya's docs for the current catalogue.
            language (TTSLanguages | str, optional): Language code. Omit it for text
                that switches languages mid-sentence, so each part is pronounced with
                its own script's rules.
            model (TTSModels | str, optional): Synthesis model. Defaults to
                ``Maya 2 Native``. ``Maya 2 Global`` is HTTP-only and unavailable here.
            api_key (str, optional): Maya API key. Falls back to the ``MAYA_API_KEY``
                environment variable.
            base_url (str, optional): API base URL, for self-hosted deployments.
                Falls back to the ``MAYA_BASE_URL`` environment variable, then to
                Maya's hosted endpoint.
            http_session (aiohttp.ClientSession, optional): An existing session to use.
            tokenizer (tokenize.SentenceTokenizer, optional): Sentence tokenizer for the
                streaming path. Defaults to ``tokenize.blingfire.SentenceTokenizer``,
                which breaks on western punctuation only; pass one that also breaks on
                the danda to stream Indic sentences as they are written.
            text_pacing (tts.SentenceStreamPacer | bool, optional): Stream pacer. True
                uses the default pacer.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_base_url = (
            base_url if is_given(base_url) else os.environ.get("MAYA_BASE_URL", DEFAULT_BASE_URL)
        )

        maya_api_key = api_key or os.environ.get("MAYA_API_KEY")
        if not maya_api_key:
            raise ValueError(
                "Maya API key is required, either as argument or set"
                " MAYA_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language,
            api_key=maya_api_key,
            base_url=resolved_base_url,
            sample_rate=SAMPLE_RATE,
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
        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

    @property
    def model(self) -> str:
        return self._opts.model if is_given(self._opts.model) else "Maya 2 Native"

    @property
    def provider(self) -> str:
        return "Maya"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._opts.get_ws_url(),
                    headers={"Authorization": f"Bearer {self._opts.api_key}"},
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            # authentication headers can appear in RequestInfo.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # transport errors can contain credentials in URLs.
            raise APIConnectionError(type(e).__name__) from None

        # Nothing else may be sent until `metadata` arrives: until the handshake
        # is accepted the connection is not on v2, and turns sent meanwhile are
        # rejected rather than served in the old shape.
        try:
            await ws.send_str(json.dumps(self._opts.start_frame()))
            msg = await asyncio.wait_for(ws.receive(), timeout)
        except asyncio.TimeoutError:
            await ws.close()
            raise APITimeoutError() from None

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise APIConnectionError(f"unexpected Maya handshake message type {msg.type}")

        data = json.loads(msg.data)
        if data.get("type") != "metadata":
            await ws.close()
            raise APIError(f"Maya rejected the connection settings: {data}")

        if (rate := data.get("sample_rate")) and rate != self._opts.sample_rate:
            logger.warning(
                "Maya reported a sample rate this plugin does not emit",
                extra={"reported": rate, "expected": self._opts.sample_rate},
            )

        logger.debug(
            "established new Maya TTS websocket connection",
            extra={"maya_session_id": data.get("session_id")},
        )
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
    ) -> None:
        """Update the voice, language or model.

        These are sticky per connection, so the change takes effect on
        connections opened after this call rather than on one already in use.

        Args:
            voice (str, optional): Voice name.
            language (TTSLanguages | str, optional): Language code.
            model (TTSModels | str, optional): Synthesis model.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(model):
            self._opts.model = model

        # Settings ride on the handshake, so existing sockets still carry the
        # old ones; drop them rather than let the two disagree.
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

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


def _text_frame(context_id: str, text: str, *, cont: bool) -> str:
    return json.dumps({"type": "text", "context_id": context_id, "text": text, "continue": cont})


async def _cancel_turn(ws: aiohttp.ClientWebSocketResponse, context_id: str) -> None:
    """Drop a turn that is still generating, so a pooled connection is not left
    streaming a turn nobody is listening to.

    Best-effort: the caller is often already unwinding from a cancellation.
    """
    if ws.closed:
        return
    try:
        await asyncio.shield(ws.send_str(json.dumps({"type": "cancel", "context_id": context_id})))
    except Exception:
        logger.debug("failed to cancel Maya turn", extra={"maya_context_id": context_id})


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a single block of text as one turn."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                await ws.send_str(_text_frame(context_id, self._input_text, cont=False))

                while True:
                    msg = await ws.receive(timeout=self._conn_options.timeout)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "Maya connection closed unexpectedly",
                            request_id=context_id,
                            status_code=ws.close_code or -1,
                            body=f"{msg.data=} {msg.extra=}",
                        )

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning("unexpected Maya message type %s", msg.type)
                        continue

                    data = json.loads(msg.data)
                    kind = data.get("type")
                    if kind == "audio":
                        output_emitter.push(base64.b64decode(data["audio"]))
                    elif kind in ("end", "cancelled"):
                        break
                    elif kind == "error":
                        raise APIError(f"Maya returned error: {data.get('error')}")
                    else:
                        logger.warning("unexpected Maya message %s", data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # One turn, one context: every sentence carries it, so a single cancel
        # drops the whole turn and audio routes correctly while several
        # sentences are in flight. Maya spends an id once, so it is minted per
        # turn rather than per connection.
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)
        turn_closed = False

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_tokenizer_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_tokenizer_stream,
                audio_emitter=output_emitter,
            )

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Sentences go out as the LLM produces them, without waiting for the
            # previous one's audio; the empty `continue: false` frame is what
            # closes the turn and makes Maya emit its single `end`.
            async for ev in sent_tokenizer_stream:
                self._mark_started()
                await ws.send_str(_text_frame(context_id, ev.token, cont=True))

            await ws.send_str(_text_frame(context_id, "", cont=False))

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue

                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal turn_closed
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Maya connection closed unexpectedly",
                        request_id=context_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Maya message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                kind = data.get("type")
                if kind == "audio":
                    output_emitter.push(base64.b64decode(data["audio"]))
                elif kind in ("end", "cancelled"):
                    # A turn ends with exactly one of these, never both.
                    turn_closed = True
                    output_emitter.end_input()
                    break
                elif kind == "error":
                    raise APIError(f"Maya returned error: {data.get('error')}")
                elif kind == "pong":
                    pass
                else:
                    logger.warning("unexpected Maya message %s", data)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
                    if not turn_closed:
                        await _cancel_turn(ws, context_id)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
