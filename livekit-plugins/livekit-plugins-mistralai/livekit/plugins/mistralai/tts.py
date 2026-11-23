from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import struct
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args
from urllib.parse import urlencode

import httpx2  # mistralai raises httpx2.TimeoutException from its http client
import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.protocol import State

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from mistralai.client import Mistral, models
from mistralai.client.errors import HTTPValidationError, SDKError
from mistralai.client.utils import generate_url, get_security, get_security_from_env

from .models import TTSModels, TTSVoices, TTSVoxtral3Voices

DEFAULT_MODEL: TTSModels = "voxtral-mini-tts-3"
DEFAULT_VOICE: TTSVoices = "en_paul_neutral"

SAMPLE_RATE: int = 24000
NUM_CHANNELS: int = 1

RESPONSE_FORMAT = Literal["mp3", "wav", "pcm", "opus", "flac"]
DEFAULT_RESPONSE_FORMAT: RESPONSE_FORMAT = "mp3"


def _is_voxtral_3(model: str) -> bool:
    return re.search(r"-tts-3(-|$)", model) is not None


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str | None
    response_format: RESPONSE_FORMAT
    ref_audio: str | None
    expressiveness: float | None = None
    phoneme_hints: dict[str, list[models.IPAPhoneme]] | None = None
    use_realtime: bool = False


class TTS(tts.TTS):
    def __init__(
        self,
        client: Mistral | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMAT] = NOT_GIVEN,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        phoneme_hints: NotGivenOr[dict[str, list[models.IPAPhoneme]]] = NOT_GIVEN,
        use_realtime: bool = False,
    ) -> None:
        """
        Create a new instance of MistralAI TTS.

        Args:
            client: Optional pre-configured MistralAI client instance.
            api_key: Your Mistral AI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            model: The Mistral AI model to use for text-to-speech, default is "voxtral-mini-tts-3".
            voice: The voice ID to use for synthesis. Mutually exclusive with ``ref_audio``. Defaults to ``en_paul_neutral`` when neither is given.
                The ``voice_<n>`` voices are only supported by latest model, which also accept the older voices.
            ref_audio: Base64-encoded audio sample (3–25 s) for zero-shot voice cloning. Mutually exclusive with ``voice``.
            response_format: The audio format of synthesized speech, between ``mp3``, ``wav``, ``pcm``, ``opus`` or ``flac``. Defaults to ``mp3``.
                Ignored with ``use_realtime``: the speech streamed over the websocket is PCM.
            expressiveness: How expressive the speech is, between 1 and 5. Only supported by latest model.
            phoneme_hints: Pronunciation hints mapping a (case-sensitive) word to its ordered list of IPA phones, e.g.
                ``{"tomato": ["t", "ə", "m", "ɑ", "t", "oʊ"]}``. Only supported by latest model, and not with ``use_realtime``.
            use_realtime: Whether to stream the speech over a realtime websocket while the text is generated,
                instead of synthesizing it sentence by sentence over http. It doesn't support
                ``phoneme_hints``. Only supported by latest model. Defaults to False.
        """
        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")

        resolved_model = model if is_given(model) else DEFAULT_MODEL
        resolved_hints = (phoneme_hints if is_given(phoneme_hints) else None) or None
        if is_given(expressiveness) and not _is_voxtral_3(resolved_model):
            raise ValueError("expressiveness is only supported by latest speech model")
        _check_voice(resolved_model, voice if is_given(voice) else None)
        _check_realtime(resolved_model, use_realtime, resolved_hints)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_realtime),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=resolved_model,
            voice=voice if is_given(voice) else (None if is_given(ref_audio) else DEFAULT_VOICE),
            ref_audio=ref_audio if is_given(ref_audio) else None,
            response_format=response_format
            if is_given(response_format)
            else DEFAULT_RESPONSE_FORMAT,
            expressiveness=expressiveness if is_given(expressiveness) else None,
            phoneme_hints=resolved_hints,
            use_realtime=use_realtime,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)
        self._pool = utils.ConnectionPool[ClientConnection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMAT] = NOT_GIVEN,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        phoneme_hints: NotGivenOr[dict[str, list[models.IPAPhoneme]]] = NOT_GIVEN,
        use_realtime: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model: The model to use for text-to-speech. Clears ``ref_audio``.
            voice: The voice ID to use for synthesis. The ``voice_<n>`` voices are only supported by the Voxtral 3 models.
            ref_audio: Base64-encoded audio sample for zero-shot voice cloning. Clears ``voice``.
            response_format: The audio format of synthesized speech, between ``mp3``, ``wav``, ``pcm``, ``opus`` or ``flac``. Defaults to ``mp3``.
            expressiveness: How expressive the speech is, between 1 and 5. Only supported by the Voxtral 3 models.
            phoneme_hints: Pronunciation hints mapping a word to its IPA phones. Only supported by the Voxtral 3 models,
                and not with ``use_realtime``.
            use_realtime: Whether to stream the speech over the realtime websocket api. Only supported by the Voxtral 3 models.
        """
        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")

        resolved_model = model if is_given(model) else self._opts.model
        resolved_expressiveness = (
            expressiveness if is_given(expressiveness) else self._opts.expressiveness
        )
        if resolved_expressiveness is not None and not _is_voxtral_3(resolved_model):
            raise ValueError("expressiveness is only supported by latest speech model")
        resolved_voice: str | None
        if is_given(voice):
            resolved_voice = voice
        else:
            resolved_voice = None if is_given(ref_audio) else self._opts.voice
        _check_voice(resolved_model, resolved_voice)
        resolved_hints = (
            phoneme_hints if is_given(phoneme_hints) else self._opts.phoneme_hints
        ) or None
        resolved_realtime = use_realtime if is_given(use_realtime) else self._opts.use_realtime
        _check_realtime(resolved_model, resolved_realtime, resolved_hints)

        session = (self._opts.model, self._session())

        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
            self._opts.ref_audio = None
        if is_given(ref_audio):
            self._opts.ref_audio = ref_audio
            self._opts.voice = None
        if is_given(response_format):
            self._opts.response_format = response_format
        self._opts.expressiveness = resolved_expressiveness
        self._opts.phoneme_hints = resolved_hints
        self._opts.use_realtime = resolved_realtime

        self._capabilities.streaming = resolved_realtime
        if session != (self._opts.model, self._session()):
            self._pool.invalidate()

    def prewarm(self) -> None:
        if self._capabilities.streaming:
            self._pool.prewarm()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        if not self._capabilities.streaming:
            return super().stream(conn_options=conn_options)
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._pool.aclose()
        await super().aclose()

    async def _acquire(self, timeout: float) -> ClientConnection:
        while True:
            ws = await self._pool.get(timeout=timeout)
            if ws.state is State.OPEN:
                return ws
            self._pool.remove(ws)

    async def _connect_ws(self, timeout: float) -> ClientConnection:
        config = self._client.sdk_configuration
        base_url, _ = config.get_server_details()
        url = generate_url(base_url, "/v1/audio/speech/realtime", None)
        url = url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)

        query = {"model": self._opts.model}
        headers: dict[str, str] = {}
        security = config.security
        if callable(security):
            security = security()
        resolved_security = get_security_from_env(security, models.Security)
        if resolved_security is not None:
            security_headers, security_query = get_security(resolved_security)
            headers |= security_headers
            query |= {key: values[-1] for key, values in security_query.items() if values}

        async def _connect() -> ClientConnection:
            ws = await connect(
                f"{url}?{urlencode(query)}",
                additional_headers=headers,
                user_agent_header=config.user_agent,
                open_timeout=timeout,
            )
            try:
                await _expect(ws, "session.created")
                # configure the whole session with a single update before any text is sent
                await ws.send(json.dumps({"type": "session.update", "session": self._session()}))
                await _expect(ws, "session.updated")
            except BaseException:
                await ws.close()
                raise
            return ws

        return await asyncio.wait_for(_connect(), timeout=timeout)

    def _session(self) -> dict[str, Any]:
        """The settings of the realtime session, sent with a ``session.update`` message."""
        session: dict[str, Any] = {"streaming_mode": "responsive", "response_format": "pcm"}
        if self._opts.ref_audio is not None:
            session["ref_audio"] = self._opts.ref_audio
        elif self._opts.voice is not None:
            session["voice"] = self._opts.voice
        if self._opts.expressiveness is not None:
            session["expressiveness"] = self._opts.expressiveness
        return session

    async def _close_ws(self, ws: ClientConnection) -> None:
        await ws.close()


class ChunkedStream(tts.ChunkedStream):
    """ChunkedStream for MistralAI TTS. Sends the full text to the API and
    returns the synthesized audio as a single chunk."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=f"audio/{self._opts.response_format}",
            )
            params: dict[str, Any] = (
                {"ref_audio": self._opts.ref_audio}
                if self._opts.ref_audio is not None
                else {"voice_id": self._opts.voice or DEFAULT_VOICE}
            )
            if self._opts.expressiveness is not None:
                params["expressiveness"] = self._opts.expressiveness
            if self._opts.phoneme_hints is not None:
                params["phoneme_hints"] = self._opts.phoneme_hints

            stream = await self._tts._client.audio.speech.complete_async(
                model=self._opts.model,
                input=self.input_text,
                response_format=self._opts.response_format,
                timeout_ms=int(self._conn_options.timeout * 1000),
                stream=True,
                **params,
            )
            async for ev in stream:
                if ev.event == "speech.audio.delta":
                    data = base64.b64decode(ev.data.audio_data)
                    if self._opts.response_format == "pcm":
                        data = _f32le_to_s16le(data)
                    output_emitter.push(data)
                elif ev.event == "speech.audio.done":
                    self._set_token_usage(
                        input_tokens=ev.data.usage.prompt_tokens,
                        output_tokens=ev.data.usage.completion_tokens,
                    )

            output_emitter.flush()

        except (asyncio.TimeoutError, httpx2.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
                retryable=e.status_code != 429,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """SynthesizeStream for the Voxtral 3 models: streams the text to the realtime websocket
    API as it is generated, and the speech back as it is synthesized."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        try:
            ws = await self._tts._acquire(self._conn_options.timeout)
            try:
                await self._synthesize(ws, output_emitter)
            except BaseException:
                self._tts._pool.remove(ws)
                raise
            self._tts._pool.put(ws)

        except APIError:
            raise
        except InvalidStatus as e:
            body = e.response.body
            raise APIStatusError(
                str(e),
                status_code=e.response.status_code,
                request_id=e.response.headers.get("x-request-id"),
                body=body.decode(errors="replace"),
                retryable=e.response.status_code != 429,
            ) from e
        except ConnectionClosed as e:
            if e.rcvd is not None and e.rcvd.code == 1008:
                raise APIStatusError(
                    e.rcvd.reason or "the realtime speech connection was rejected",
                    status_code=e.rcvd.code,
                    body=e.rcvd.reason,
                    retryable=False,
                ) from e
            raise APIConnectionError() from e
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError() from e

    async def _synthesize(self, ws: ClientConnection, output_emitter: tts.AudioEmitter) -> None:
        text = await self._next_text(until_flush=False)
        if text is None:
            return

        output_emitter.start_segment(segment_id=utils.shortuuid())
        input_tokens = output_tokens = 0

        while text is not None:
            flushed = asyncio.Event()
            send = asyncio.create_task(self._send_text(ws, text, flushed))
            recv = asyncio.create_task(self._recv_audio(ws, flushed, output_emitter))
            try:
                (split, rest), usage = await asyncio.gather(send, recv)
            finally:
                await utils.aio.gracefully_cancel(send, recv)

            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
            if rest:
                text = rest
            else:
                text = await self._next_text(until_flush=True) if split else None

        output_emitter.end_segment()
        self._set_token_usage(input_tokens=input_tokens, output_tokens=output_tokens)

    async def _next_text(self, *, until_flush: bool) -> str | None:
        """The next text to synthesize, or None when the segment (with `until_flush`) or the
        input ended."""
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if until_flush:
                    return None
            elif item.strip():
                return item
        return None

    async def _send_text(
        self, ws: ClientConnection, text: str, flushed: asyncio.Event
    ) -> tuple[bool, str]:
        """Send the text of a request up to the end of the segment. Returns whether the request
        was cut short, so that the segment goes on, and the part of the text which didn't fit in
        it, to send first in the next request."""
        self._mark_started()
        head, rest = _cut(text, _MAX_REQUEST_CHARS, force=True)
        await ws.send(json.dumps({"type": "input_text.append", "text": head}))
        sent = len(head)
        split = bool(rest)

        if not split:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    break

                head, rest = _cut(item, _MAX_REQUEST_CHARS - sent, force=False)
                if head:
                    await ws.send(json.dumps({"type": "input_text.append", "text": head}))
                    sent += len(head)
                if (
                    rest
                    or sent >= _MAX_REQUEST_CHARS
                    or (sent >= _SPLIT_AFTER_CHARS and head.rstrip(" ").endswith(_SENTENCE_ENDS))
                ):
                    split = True
                    break

        await ws.send(json.dumps({"type": "input_text.flush"}))
        flushed.set()
        return split, rest

    async def _recv_audio(
        self, ws: ClientConnection, flushed: asyncio.Event, output_emitter: tts.AudioEmitter
    ) -> dict[str, int]:
        pcm = _Float32ToInt16()
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), self._conn_options.timeout)
            except (asyncio.TimeoutError, TimeoutError):
                # nothing is expected until the text is flushed, it can still be generated
                if flushed.is_set():
                    raise
                continue

            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "speech.audio.delta":
                output_emitter.push(pcm.convert(base64.b64decode(event["audio_data"])))
            elif event_type == "speech.audio.done":
                usage: dict[str, int] = event.get("usage") or {}
                return usage
            elif event_type == "error":
                raise _realtime_error(event)


class _Float32ToInt16:
    """Converts the float32 PCM of the realtime API to 16-bit PCM, whatever how it is chunked."""

    def __init__(self) -> None:
        self._remainder = b""

    def convert(self, data: bytes) -> bytes:
        data = self._remainder + data
        usable = len(data) - len(data) % 4
        self._remainder = data[usable:]
        samples = np.frombuffer(data[:usable], dtype="<f4")
        return bytes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


async def _expect(ws: ClientConnection, event_type: str) -> None:
    event = json.loads(await ws.recv())
    if event.get("type") == "error":
        raise _realtime_error(event)
    if event.get("type") != event_type:
        raise APIConnectionError(
            f"expected {event_type} from the realtime speech API, got {event.get('type')}"
        )


def _realtime_error(event: dict[str, Any]) -> APIStatusError:
    error = event.get("error") or {}
    code = error.get("code")
    return APIStatusError(
        str(error.get("message", "realtime speech error")),
        status_code=code if isinstance(code, int) else -1,
        body=error,
        retryable=False,
    )


def _f32le_to_s16le(data: bytes) -> bytes:
    n = len(data) // 4
    floats = struct.unpack(f"<{n}f", data)
    return struct.pack(f"<{n}h", *(max(-32768, min(32767, int(s * 32767))) for s in floats))


def _check_voice(model: str, voice: str | None) -> None:
    if voice in frozenset(get_args(TTSVoxtral3Voices)) and not _is_voxtral_3(model):
        raise ValueError(f"voice {voice!r} is only supported by latest model")


def _check_realtime(
    model: str, use_realtime: bool, phoneme_hints: dict[str, list[models.IPAPhoneme]] | None
) -> None:
    if (use_realtime or phoneme_hints is not None) and not _is_voxtral_3(model):
        raise ValueError("use_realtime and phoneme_hints are only supported by latest speech model")
    if use_realtime and phoneme_hints is not None:
        raise ValueError("phoneme_hints is not supported with use_realtime")


_SPLIT_AFTER_CHARS = 1500
_MAX_REQUEST_CHARS = 2500
_SENTENCE_ENDS = (".", "!", "?", "…", "\n")


def _cut(text: str, room: int, *, force: bool) -> tuple[str, str]:
    """Cut `text` to at most `room` characters, at the last sentence end, else at the last
    space. Without such a boundary the text is cut anywhere if `force` is set, and kept whole
    for the next request otherwise."""
    if len(text) <= room:
        return text, ""
    window = text[: max(room, 0)]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _SENTENCE_ENDS:
            return text[: i + 1], text[i + 1 :]
    i = window.rfind(" ")
    if i > 0:
        return text[:i], text[i:]
    return (window, text[len(window) :]) if force else ("", text)
