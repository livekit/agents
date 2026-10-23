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

"""Lokutor speech-to-text.

Streaming uses ``WS /ws/stt``. Wire contract:

    client -> server   {"language": "en"}   ("" lets the model detect it)
    server -> client   {"type": "status", "data": "ready"}
    client -> server   binary frames: 16-bit PCM, 16 kHz, mono
    server -> client   {"type": "transcript", "data": ..., "isFinal": bool}
    client -> server   {"type": "end"}
    server -> client   the last final, then {"type": "status", "data": "done"}

The server decides where an utterance ends (a pause of about 0.7 s). ``data`` is the
whole utterance so far on every partial, and the final carries the utterance once,
transcribed in one pass. Single-shot recognition uses ``POST /stt/transcribe``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import DEFAULT_BASE_URL, STT_LANGUAGES, STT_MODEL, STTLanguage

_SAMPLE_RATE = 16000


@dataclass
class _STTOptions:
    api_key: str
    language: str | None
    base_url: str


def _api_language(language: str | None) -> str:
    """The language as the API takes it ("en-US" -> "en"), or "" to have it detected.

    The recogniser writes the text by the rules of the language it is given, so a
    language it does not serve is left to detection rather than forced.
    """
    if not language:
        return ""
    code = LanguageCode(language).language
    return code if code in STT_LANGUAGES else ""


def _http_base(base_url: str) -> str:
    if base_url.startswith("wss://"):
        return "https://" + base_url[len("wss://") :]
    if base_url.startswith("ws://"):
        return "http://" + base_url[len("ws://") :]
    return base_url


class STT(stt.STT):
    """Lokutor Speech-to-Text integration.

    One model for nine languages: English, Spanish, Catalan, Galician, Basque,
    Portuguese, French, Italian and German. The text comes back written out, with
    punctuation, casing, digits and e-mail addresses.

    Args:
        api_key: Lokutor API key. If not provided, reads from ``LOKUTOR_API_KEY``
            environment variable.
        language: Language code (e.g. ``"en"``, ``"es-ES"``). Send it whenever it is
            known: it keeps one-word replies reliable and sets how the text is
            written. Left unset, or a language the model does not serve, the
            language is detected.
        base_url: Base URL for the Lokutor API. Defaults to ``"wss://api.lokutor.com"``.
        http_session: Optional shared ``aiohttp.ClientSession``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: NotGivenOr[STTLanguage | str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
            ),
        )
        lokutor_api_key = api_key or os.environ.get("LOKUTOR_API_KEY")
        if not lokutor_api_key:
            raise ValueError(
                "Lokutor API key is required, either as argument or set"
                " LOKUTOR_API_KEY environment variable"
            )
        self._opts = _STTOptions(
            api_key=lokutor_api_key,
            language=language if is_given(language) else None,
            base_url=base_url,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        """Returns the model name ``"conv-2.0"``."""
        return STT_MODEL

    @property
    def provider(self) -> str:
        """Returns the provider name ``"Lokutor"``."""
        return "Lokutor"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = _api_language(language if is_given(language) else self._opts.language)
        frame = rtc.combine_audio_frames(buffer)
        body: dict[str, str] = {
            "audio": base64.b64encode(frame.to_wav_bytes()).decode(),
            "format": "wav",
        }
        if lang:
            body["language"] = lang

        try:
            async with self._ensure_session().post(
                f"{_http_base(self._opts.base_url)}/stt/transcribe",
                json=body,
                headers={"X-API-Key": self._opts.api_key},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    raise APIStatusError(
                        "Lokutor transcription failed",
                        status_code=resp.status,
                        request_id=None,
                        body=None,
                    )
                data = await resp.json()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            # Keep only the exception type: a transport error can carry the request
            # URL, headers or a response body.
            raise APIConnectionError(type(e).__name__) from None

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid("lokutor-"),
            alternatives=[
                stt.SpeechData(
                    # Without a requested language, the one the API reports (detected), if it serves it.
                    language=LanguageCode(
                        lang or _api_language(str(data.get("language") or "")) or "multi"
                    ),
                    text=str(data.get("text") or "").strip(),
                    start_time=0.0,
                    end_time=frame.duration,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a streaming recognition session over ``WS /ws/stt``."""
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language if is_given(language) else self._opts.language,
        )


class SpeechStream(stt.SpeechStream):
    """Streaming recognition: interim transcripts while the user speaks, a final one
    when the server sees the utterance end."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str | None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=_SAMPLE_RATE)
        self._stt: STT = stt
        self._api_lang = _api_language(language)
        self._language = LanguageCode(self._api_lang or "multi")
        self._speaking = False
        self._request_id = ""
        self._input_done = False
        self._done = False
        self._audio_sent = 0.0
        self._audio_reported = 0.0

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        try:
            return await asyncio.wait_for(
                self._stt._ensure_session().ws_connect(
                    f"{self._stt._opts.base_url}/ws/stt",
                    max_msg_size=0,
                    # In a header, not the query string, so the key stays out of
                    # exception messages, tracing and access logs.
                    headers={"X-API-Key": self._stt._opts.api_key},
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            # 401 bad key, 402 no balance, 429 the account's concurrent sessions are in use.
            raise APIStatusError(
                "Lokutor STT refused the connection",
                status_code=e.status,
                request_id=None,
                body=None,
                retryable=e.status >= 500,
            ) from None
        except Exception as e:
            raise APIConnectionError(type(e).__name__) from None

    async def _run(self) -> None:
        self._input_done = False
        self._done = False
        ws = await self._connect_ws()
        try:
            await ws.send_str(json.dumps({"language": self._api_lang}))
            await asyncio.wait_for(self._await_ready(ws), self._conn_options.timeout)

            send_task = asyncio.create_task(self._send_task(ws))
            recv_task = asyncio.create_task(self._recv_task(ws))
            try:
                await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
                if send_task.done():
                    send_task.result()
                    # Input has ended: the last final transcript and "done" must follow within the
                    # timeout. A live stream waits on the caller for as long as it lasts; this
                    # bounds only the wait for a server that stops answering after "end".
                    await asyncio.wait_for(recv_task, self._conn_options.timeout)
                else:
                    recv_task.result()
            finally:
                await utils.aio.gracefully_cancel(send_task, recv_task)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        finally:
            self._report_usage()
            await ws.close()

    async def _await_ready(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise APIConnectionError("Lokutor STT closed before it was ready")
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "error":
                raise APIStatusError(
                    "Lokutor STT rejected the session",
                    status_code=-1,
                    request_id=None,
                    body=None,
                    retryable=False,
                )
            if data.get("type") == "status" and data.get("data") == "ready":
                self._note_language(data.get("lang"))
                return

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for data in self._input_ch:
            # A flush needs nothing here: the server ends an utterance on its own pause.
            if isinstance(data, rtc.AudioFrame):
                audio = data.data.tobytes()
                if audio:
                    self._audio_sent += data.samples_per_channel / data.sample_rate
                    await ws.send_bytes(audio)
        self._input_done = True
        await ws.send_str(json.dumps({"type": "end"}))

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # Only "done" ends a stream: a close before it, even after end_input(), may have
                # lost the last final transcript.
                if self._done:
                    return
                raise APIConnectionError("Lokutor STT closed before it finished")
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("Lokutor STT connection failed")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("Ignoring a non-JSON frame from Lokutor STT")
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type")
            if kind == "error":
                # Not the provider's message: it may echo customer content.
                raise APIStatusError(
                    "Lokutor STT returned an error",
                    status_code=-1,
                    request_id=None,
                    body=None,
                    retryable=False,
                )
            if kind == "status" and data.get("data") == "done":
                self._done = True
                return
            if kind == "transcript":
                self._note_language(data.get("lang"))
                self._on_transcript(
                    str(data.get("data") or "").strip(), data.get("isFinal") is True
                )

    def _note_language(self, lang: object) -> None:
        # With no language requested, label transcripts with the one the server reports, if it
        # serves it; "multi" otherwise.
        if not self._api_lang and isinstance(lang, str) and _api_language(lang):
            self._language = LanguageCode(_api_language(lang))

    def _on_transcript(self, text: str, final: bool) -> None:
        if not text:
            return
        if not self._speaking:
            self._speaking = True
            self._request_id = utils.shortuuid("lokutor-")
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=(
                    stt.SpeechEventType.FINAL_TRANSCRIPT
                    if final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                ),
                request_id=self._request_id,
                alternatives=[stt.SpeechData(language=self._language, text=text)],
            )
        )
        if final:
            self._event_ch.send_nowait(
                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
            )
            self._report_usage()
            self._speaking = False
            self._request_id = ""

    def _report_usage(self) -> None:
        duration = self._audio_sent - self._audio_reported
        if duration <= 0:
            return
        self._audio_reported = self._audio_sent
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )
