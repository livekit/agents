# Copyright 2026 LiveKit, Inc.
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

"""Audrey, p99lab's speech-to-text API, as a LiveKit Agents STT.

Usage::

    from livekit.plugins import p99lab

    session = AgentSession(stt=p99lab.STT(), vad=..., llm=..., tts=..., ...)

Create an API key at https://console.p99lab.com and set ``P99LAB_API_KEY``.

The stream sends the session's audio to Audrey over a WebSocket as 16-bit mono
PCM. Audrey answers with partial transcripts while the user speaks and a final
transcript at each pause. ``recognize()`` posts a whole buffer as WAV.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

# The public Audrey API. Override with base_url= or the P99LAB_API_URL environment variable.
DEFAULT_BASE_URL = "https://api.p99lab.com"
API_KEY_ENV = "P99LAB_API_KEY"
API_URL_ENV = "P99LAB_API_URL"
BILLING_URL = "https://console.p99lab.com/dashboard/billing"
SAMPLE_RATES = (8000, 16000, 24000, 48000)
LANGUAGES = (
    "en",
    "es",
    "hi",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "pl",
    "sv",
    "da",
    "fi",
    "cs",
    "ro",
    "hu",
    "el",
    "ru",
    "tr",
    "ar",
    "fa",
    "zh",
    "yue",
    "ja",
    "ko",
    "th",
    "vi",
    "id",
    "ms",
    "fil",
    "tl",
    "mk",
    "auto",
)  # the public API's languages (2026-10-25); "auto" detects English vs Spanish
CHUNK_MS = 50
USAGE_REPORT_SECS = 5.0
# How long to wait for the server's "done" after the end of the audio.
DONE_TIMEOUT_SECS = 60.0
MAX_PROMPT_CHARS = 500

# The fields of a server error payload that are kept in ``APIStatusError.body``.
_ERROR_FIELDS = ("code", "reason", "top_up_url", "plans_url")

# Error messages after which reconnecting cannot help. Account refusals also carry the HTTP
# status as ``code`` (401, 402, 429, 503); these strings cover servers that send only a message.
_FATAL_ERRORS = (
    "api key",
    "allowance",
    "must be",
    "first message",
    "larger than 64",
    "credits are used up",
    "payment failed",
    "daily limit",
    "daily audio limit",
)


@dataclass
class STTOptions:
    language: str
    sample_rate: int
    prompt: str | None
    endpoint_ms: int | None
    partial_ms: int | None
    base_url: str
    stream_url: str


class STT(stt.STT):
    """Streaming speech-to-text with p99lab's Audrey API.

    Partial transcripts arrive while the user speaks; a final transcript
    arrives when the speaker pauses for ``endpoint_ms`` or when the stream is
    flushed.

    Account refusals raise ``APIStatusError`` with the HTTP status and are not
    retried: 401 (unknown or revoked key), 402 (no credit; ``body["reason"]``
    is ``"no_plan"``, ``"no_credit"`` or ``"payment_failed"`` and
    ``body["console_url"]`` the billing page) and 429 for the key's daily
    limit. 503 and "server busy" are retried.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        endpoint_ms: NotGivenOr[int] = NOT_GIVEN,
        partial_ms: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: int = 16000,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        stream_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an Audrey STT.

        Args:
            language: "en" (default), or any of LANGUAGES (30 languages; ISO codes, region
                suffixes such as "pt-BR" accepted). "auto" detects English vs Spanish.
            api_key: Audrey API key from https://console.p99lab.com. Defaults to
                the P99LAB_API_KEY environment variable.
            prompt: Context text, up to 500 characters: names, product terms.
            endpoint_ms: Silence, in ms, after which the server finalises an
                utterance (200 to 3000). Omitted, the server default applies.
            partial_ms: How often, in ms of new speech, the server sends a
                partial transcript (250 to 5000). Omitted, the server default applies.
            sample_rate: Rate the audio is sent at: 8000, 16000 (default),
                24000 or 48000. Audio at any other rate is resampled to it.
            base_url: HTTP base of the API. Defaults to the P99LAB_API_URL
                environment variable, else the public Audrey API.
            stream_url: WebSocket URL for streaming. Defaults to ``base_url``
                with a ws(s) scheme and the path ``/v1/stream``.
            http_session: aiohttp session to use. Defaults to the job's shared
                session.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=True,
                manual_flush=True,
            )
        )

        key = (
            api_key
            if is_given(api_key)
            else os.environ.get(API_KEY_ENV) or os.environ.get("AUDREY_API_KEY")
        )
        if not key:
            raise ValueError(
                f"an Audrey API key is required, either as api_key or in the {API_KEY_ENV}"
                " environment variable (create one at https://console.p99lab.com)"
            )
        self._api_key = key
        url = base_url if is_given(base_url) else os.environ.get(API_URL_ENV) or DEFAULT_BASE_URL

        self._opts = STTOptions(
            language=_validate_language(language),
            sample_rate=_validate_sample_rate(sample_rate),
            prompt=_validate_prompt(prompt) if is_given(prompt) else None,
            endpoint_ms=_validate_range("endpoint_ms", endpoint_ms, 200, 3000)
            if is_given(endpoint_ms)
            else None,
            partial_ms=_validate_range("partial_ms", partial_ms, 250, 5000)
            if is_given(partial_ms)
            else None,
            base_url=url.rstrip("/"),
            stream_url=stream_url if is_given(stream_url) else _to_stream_url(url),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "audrey"

    @property
    def provider(self) -> str:
        return "p99lab"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = _validate_language(language)
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        frame = rtc.combine_audio_frames(buffer)
        params = {"language": config.language}
        if config.prompt:
            params["prompt"] = config.prompt

        try:
            async with self._ensure_session().post(
                url=f"{config.base_url}/v1/transcribe",
                params=params,
                data=frame.to_wav_bytes(),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                timeout=aiohttp.ClientTimeout(
                    total=60 + frame.duration,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    raise await _http_error(res)
                data = await res.json()
        except APIStatusError:
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            # The request info carries the Authorization header: do not chain it.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"Audrey request failed ({type(e).__name__})") from None

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(config.language),
                    text=str(data.get("text") or ""),
                    start_time=0.0,
                    end_time=float(data.get("audio_seconds") or frame.duration),
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._sanitize_options(language=language),
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        endpoint_ms: NotGivenOr[int | None] = NOT_GIVEN,
        partial_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Change options. Open streams reconnect to apply them, which drops the
        utterance in progress."""
        _apply_options(
            self._opts,
            language=language,
            prompt=prompt,
            endpoint_ms=endpoint_ms,
            partial_ms=partial_ms,
        )
        for stream in self._streams:
            stream.update_options(
                language=language, prompt=prompt, endpoint_ms=endpoint_ms, partial_ms=partial_ms
            )


class SpeechStream(stt.RecognizeStream):
    """One streaming session. Reconnects on dropped connections and on
    retryable server errors, re-sending the configuration each time."""

    _FLUSH_MSG = json.dumps({"type": "flush"})
    _END_MSG = json.dumps({"type": "end"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._reconnect_event = asyncio.Event()
        self._unreported_secs = 0.0
        self._request_id = ""

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        endpoint_ms: NotGivenOr[int | None] = NOT_GIVEN,
        partial_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        _apply_options(
            self._opts,
            language=language,
            prompt=prompt,
            endpoint_ms=endpoint_ms,
            partial_ms=partial_ms,
        )
        self._reconnect_event.set()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio. Frames with more than one channel are mixed down to mono
        before resampling: Audrey takes mono audio only."""
        if frame.num_channels > 1:
            frame = _downmix(frame)
        super().push_frame(frame)

    def _config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "language": self._opts.language,
            "sample_rate": self._opts.sample_rate,
        }
        if self._opts.prompt:
            config["prompt"] = self._opts.prompt
        if self._opts.endpoint_ms is not None:
            config["endpoint_ms"] = self._opts.endpoint_ms
        if self._opts.partial_ms is not None:
            config["partial_ms"] = self._opts.partial_ms
        return config

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate * CHUNK_MS // 1000,
            )
            try:
                async for data in self._input_ch:
                    if isinstance(data, rtc.AudioFrame):
                        for frame in audio_bstream.write(data.data.tobytes()):
                            await self._send_audio(ws, frame)
                    elif isinstance(data, self._FlushSentinel):
                        for frame in audio_bstream.flush():
                            await self._send_audio(ws, frame)
                        self._report_usage()
                        await ws.send_str(self._FLUSH_MSG)

                # No more audio: the server finalises, sends "done" and closes.
                closing_ws = True
                self._report_usage()
                await ws.send_str(self._END_MSG)
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("Audrey connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if self._session.closed:
                        return
                    raise APIStatusError(
                        message="Audrey connection closed before the stream was done",
                        status_code=ws.close_code or -1,
                        body=None,
                    )
                if msg.type == aiohttp.WSMsgType.ERROR:
                    if self._session.closed:
                        return
                    raise APIConnectionError("Audrey connection lost") from ws.exception()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Audrey message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                except ValueError:
                    logger.warning("Audrey sent a message that is not JSON")
                    continue

                kind = data.get("type")
                if kind == "done":
                    closing_ws = True
                    return
                if kind == "error":
                    closing_ws = True
                    raise _stream_error(data)
                try:
                    self._process_stream_event(data)
                    if kind in ("partial", "final"):
                        self._mark_provider_progress()
                except Exception:
                    logger.exception("failed to process an Audrey message")

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    # The stream is complete when the server says "done" (recv_task
                    # returns); the send task ends earlier, after sending "end".
                    pending: set[asyncio.Task] = {*tasks, wait_reconnect_task}
                    reconnect = False
                    while True:
                        timeout = DONE_TIMEOUT_SECS if tasks[0].done() else None
                        done, pending = await asyncio.wait(
                            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                        )
                        if not done:
                            raise APIConnectionError(
                                "Audrey sent no 'done' after the end of the audio",
                                retryable=False,
                            )
                        for task in done:
                            if task is not wait_reconnect_task:
                                task.result()
                        if wait_reconnect_task in done:
                            reconnect = True
                            break
                        if tasks[1] in done:
                            break
                    if not reconnect:
                        break
                    self._reconnect_event.clear()
                    self._end_speech()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if ws is not None:
                    await ws.close()
                    ws = None
                self._report_usage()

    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse, frame: rtc.AudioFrame) -> None:
        self._unreported_secs += frame.duration
        await ws.send_bytes(frame.data.tobytes())
        if self._unreported_secs >= USAGE_REPORT_SECS:
            self._report_usage()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._opts.stream_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    heartbeat=30.0,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("timed out connecting to Audrey") from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                message=f"Audrey refused the WebSocket ({e.status})",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            # Not chained: aiohttp errors can carry the request headers, key included.
            raise APIConnectionError(f"failed to connect to Audrey ({type(e).__name__})") from None

        try:
            await ws.send_str(json.dumps(self._config()))
        except Exception as e:
            await ws.close()
            raise APIConnectionError(
                f"failed to configure the Audrey stream ({type(e).__name__})"
            ) from None

        self._report_connection_acquired(time.perf_counter() - t0, False)
        return ws

    def _report_usage(self) -> None:
        if self._unreported_secs <= 0:
            return
        secs, self._unreported_secs = self._unreported_secs, 0.0
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=secs),
            )
        )

    def _start_speech(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _end_speech(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _speech_data(self, data: dict) -> stt.SpeechData:
        start = float(data.get("start") or 0.0)
        end = float(data.get("end") or start)
        return stt.SpeechData(
            language=LanguageCode(self._opts.language),
            text=str(data.get("text") or "").strip(),
            start_time=start + self.start_time_offset,
            end_time=end + self.start_time_offset,
            confidence=1.0,
        )

    def _process_stream_event(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "partial":
            sd = self._speech_data(data)
            if not sd.text:
                return
            self._start_speech()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[sd],
                )
            )
        elif kind == "final":
            sd = self._speech_data(data)
            if sd.text:
                self._start_speech()
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[sd],
                    )
                )
            # "continues": the server cut a long utterance (about 25 s) while the
            # speaker was still talking, so this final is not the end of the speech.
            if not data.get("continues"):
                self._end_speech()
        else:
            logger.warning("unexpected Audrey message type %r", kind)


def _downmix(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    mono = samples.mean(axis=1).round().astype(np.int16)
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=len(mono),
    )


def _payment_reason(message: str, explicit: object = None) -> str:
    """Why a 402 happened: the server's ``reason`` field if present, else read from its message."""
    if explicit in ("no_plan", "no_credit", "payment_failed"):
        return str(explicit)
    lower = message.lower()
    if "payment failed" in lower or "update your card" in lower:
        return "payment_failed"
    if "choose a plan to continue" in lower:
        return "no_plan"
    return "no_credit"


def _refusal(status: int, message: str, detail: dict[str, Any]) -> APIStatusError:
    """The error for a refusal, with a machine-readable ``body``: ``code`` and, for a 402,
    ``reason`` and ``console_url``."""
    body: dict[str, Any] = {k: detail[k] for k in _ERROR_FIELDS if k in detail}
    code = body.get("code")
    body["code"] = code if isinstance(code, str) else ("unknown_key" if status == 401 else None)
    body["status"] = status
    lower = message.lower()
    retryable: bool | None = None
    if status == 402:
        body["reason"] = _payment_reason(message, body.get("reason"))
        body["console_url"] = str(body.get("top_up_url") or body.get("plans_url") or BILLING_URL)
        if body["console_url"] not in message:
            message = f"{message} ({body['console_url']})"
    elif status == 429:
        # The key's daily limit lasts until midnight Pacific time; a parallel-stream limit clears
        # when another stream of the same key ends.
        retryable = not (
            body["code"] in ("key_daily_limit", "research_capacity")
            or "capacity reached" in lower
            or any(s in lower for s in _FATAL_ERRORS)
        )
    elif status == -1:
        retryable = not any(s in lower for s in _FATAL_ERRORS)
    return APIStatusError(
        message=f"Audrey: {message}",
        status_code=status,
        request_id=None,
        body=body,
        retryable=retryable,
    )


def _stream_error(data: dict[str, Any]) -> APIStatusError:
    message = str(data.get("message") or "unknown error")
    status = data.get("code")
    if not isinstance(status, int):
        lower = message.lower()
        status = 401 if "api key" in lower else 429 if "allowance" in lower else -1
    return _refusal(status, message, data)


async def _http_error(res: aiohttp.ClientResponse) -> APIStatusError:
    detail: Any = None
    try:
        payload = await res.json(content_type=None)
        if isinstance(payload, dict):
            detail = payload.get("detail")
    except Exception:
        pass
    if isinstance(detail, dict):
        return _refusal(res.status, str(detail.get("message") or f"HTTP {res.status}"), detail)
    return _refusal(res.status, str(detail) if detail else f"HTTP {res.status}", {})


def _apply_options(
    opts: STTOptions,
    *,
    language: NotGivenOr[str],
    prompt: NotGivenOr[str | None],
    endpoint_ms: NotGivenOr[int | None],
    partial_ms: NotGivenOr[int | None],
) -> None:
    if is_given(language):
        opts.language = _validate_language(language)
    if is_given(prompt):
        opts.prompt = _validate_prompt(prompt) if prompt else None
    if is_given(endpoint_ms):
        opts.endpoint_ms = (
            None if endpoint_ms is None else _validate_range("endpoint_ms", endpoint_ms, 200, 3000)
        )
    if is_given(partial_ms):
        opts.partial_ms = (
            None if partial_ms is None else _validate_range("partial_ms", partial_ms, 250, 5000)
        )


def _validate_language(language: str) -> str:
    code = str(language).strip().lower().replace("_", "-")
    if code not in LANGUAGES and "-" in code:
        code = code.split("-")[0]  # "en-US" -> "en", "pt-BR" -> "pt"
    if code not in LANGUAGES:
        raise ValueError(
            f"Audrey does not support language {language!r}; use one of {', '.join(LANGUAGES)}"
        )
    return code


def _validate_sample_rate(sample_rate: int) -> int:
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, not {sample_rate}")
    return sample_rate


def _validate_prompt(prompt: str) -> str:
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")
    return prompt


def _validate_range(name: str, value: int, low: int, high: int) -> int:
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}, not {value}")
    return int(value)


def _to_stream_url(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path + "/v1/stream", "", ""))
