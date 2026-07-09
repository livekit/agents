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

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import uuid
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    TimedString,
)

from .log import logger

DEFAULT_ENGINE_MODEL_TYPE = "16k_zh_en"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_VOICE_FORMAT = 1
DEFAULT_HOST = "asr.cloud.tencent.com"


@dataclass
class _STTOptions:
    app_id: str
    secret_id: str
    secret_key: str
    engine_model_type: str
    language: str
    sample_rate: int
    voice_format: int
    needvad: int
    filter_modal: int
    filter_dirty: int
    filter_empty_result: int
    convert_num_mode: int
    chat_vad_enable: int
    vad_silence_time: int
    max_speak_time: int
    filter_punc: int
    word_info: int
    reinforce_hotword: int
    hotword_id: str | None
    hotword_list: str | None
    customization_id: str | None
    replace_text_id: str | None
    noise_threshold: float | None
    domain: int | None
    host: str
    proxy_url: str | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        secret_id: NotGivenOr[str] = NOT_GIVEN,
        secret_key: NotGivenOr[str] = NOT_GIVEN,
        engine_model_type: str = DEFAULT_ENGINE_MODEL_TYPE,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        voice_format: int = DEFAULT_VOICE_FORMAT,
        needvad: int = 1,
        filter_modal: int = 1,
        filter_dirty: int = 1,
        filter_empty_result: int = 1,
        convert_num_mode: int = 1,
        chat_vad_enable: int = 1,
        vad_silence_time: int = 500,
        max_speak_time: int = 15000,
        filter_punc: int = 0,
        word_info: int = 0,
        reinforce_hotword: int = 0,
        hotword_id: str | None = None,
        hotword_list: str | None = None,
        customization_id: str | None = None,
        replace_text_id: str | None = None,
        noise_threshold: float | None = None,
        domain: int | None = None,
        host: str = DEFAULT_HOST,
        proxy_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Tencent Cloud ASR streaming STT instance."""
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
            )
        )

        resolved_app_id = app_id if utils.is_given(app_id) else os.getenv("TENCENT_ASR_APP_ID", "")
        resolved_secret_id = (
            secret_id if utils.is_given(secret_id) else os.getenv("TENCENT_ASR_SECRET_ID", "")
        )
        resolved_secret_key = (
            secret_key if utils.is_given(secret_key) else os.getenv("TENCENT_ASR_SECRET_KEY", "")
        )

        if not resolved_app_id or not resolved_secret_id or not resolved_secret_key:
            raise ValueError(
                "Tencent ASR credentials are required: set TENCENT_ASR_APP_ID, "
                "TENCENT_ASR_SECRET_ID, and TENCENT_ASR_SECRET_KEY or pass them explicitly"
            )

        self._opts = _STTOptions(
            app_id=resolved_app_id,
            secret_id=resolved_secret_id,
            secret_key=resolved_secret_key,
            engine_model_type=engine_model_type,
            language=language,
            sample_rate=sample_rate,
            voice_format=voice_format,
            needvad=needvad,
            filter_modal=filter_modal,
            filter_dirty=filter_dirty,
            filter_empty_result=filter_empty_result,
            convert_num_mode=convert_num_mode,
            chat_vad_enable=chat_vad_enable,
            vad_silence_time=vad_silence_time,
            max_speak_time=max_speak_time,
            filter_punc=filter_punc,
            word_info=word_info,
            reinforce_hotword=reinforce_hotword,
            hotword_id=hotword_id,
            hotword_list=hotword_list,
            customization_id=customization_id,
            replace_text_id=replace_text_id,
            noise_threshold=noise_threshold,
            domain=domain,
            host=host,
            proxy_url=proxy_url,
        )
        self._http_session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.engine_model_type

    @property
    def provider(self) -> str:
        return "Tencent"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Tencent ASR does not support batch recognition in this plugin; use stream()"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language if utils.is_given(language) else self._opts.language,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt: STT = stt
        self._language = language
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id = ""
        self._speaking = False
        self._last_index = -1
        self._audio_duration = 0.0
        self._last_usage_report = time.monotonic()
        self._pending_audio = bytearray()
        self._send_threshold_bytes = max(1, stt._opts.sample_rate // 5 * 2)

    def _build_url(self, *, voice_id: str | None = None, now: int | None = None) -> str:
        opts = self._stt._opts
        voice_id = voice_id or str(uuid.uuid4())
        timestamp = int(time.time() if now is None else now)

        query: dict[str, str] = {
            "secretid": opts.secret_id,
            "timestamp": str(timestamp),
            "expired": str(timestamp + 24 * 60 * 60),
            "nonce": str(timestamp),
            "engine_model_type": opts.engine_model_type,
            "voice_id": voice_id,
            "voice_format": str(opts.voice_format),
            "needvad": str(opts.needvad),
            "filter_dirty": str(opts.filter_dirty),
            "filter_modal": str(opts.filter_modal),
            "filter_punc": str(opts.filter_punc),
            "filter_empty_result": str(opts.filter_empty_result),
            "convert_num_mode": str(opts.convert_num_mode),
            "word_info": str(opts.word_info),
            "reinforce_hotword": str(opts.reinforce_hotword),
            "max_speak_time": str(opts.max_speak_time),
        }

        if opts.hotword_id:
            query["hotword_id"] = opts.hotword_id
        if opts.hotword_list:
            query["hotword_list"] = opts.hotword_list
        if opts.customization_id:
            query["customization_id"] = opts.customization_id
        if opts.replace_text_id:
            query["replace_text_id"] = opts.replace_text_id
        if opts.vad_silence_time > 0:
            query["vad_silence_time"] = str(opts.vad_silence_time)
        if opts.noise_threshold is not None and opts.noise_threshold != 0:
            query["noise_threshold"] = f"{opts.noise_threshold:.3f}"
        if opts.chat_vad_enable > 0:
            query["chat_vad_enable"] = str(opts.chat_vad_enable)
        if opts.domain is not None and opts.domain > 0:
            query["domain"] = str(opts.domain)

        query_str = "&".join(f"{key}={query[key]}" for key in sorted(query))
        sign_url = f"{opts.host}/asr/v2/{opts.app_id}?{query_str}"
        digest = hmac.new(
            opts.secret_key.encode("utf-8"),
            sign_url.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        signature = urllib.parse.quote(base64.b64encode(digest).decode("utf-8"), safe="")
        return f"wss://{sign_url}&signature={signature}"

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        self._request_id = str(uuid.uuid4())
        ws_url = self._build_url(voice_id=self._request_id)

        try:
            ws = await asyncio.wait_for(
                self._stt._ensure_session().ws_connect(
                    ws_url,
                    proxy=self._stt._opts.proxy_url,
                ),
                timeout=self._conn_options.timeout,
            )
            msg = await asyncio.wait_for(ws.receive(), timeout=self._conn_options.timeout)
        except TimeoutError as e:
            raise APITimeoutError("Timeout connecting to Tencent ASR API") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=self._request_id,
                body=None,
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Tencent ASR connection error: {e}") from e

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise APIConnectionError(f"Unexpected Tencent ASR handshake message: {msg.type}")

        data = _json_loads(msg.data)
        self._raise_for_tencent_error(data)
        logger.debug("Tencent ASR WebSocket connection established")
        return ws

    async def _run(self) -> None:
        ws = await self._connect_ws()
        self._ws = ws
        send_task = asyncio.create_task(self._send_audio_task())
        recv_task = asyncio.create_task(self._recv_messages_task())
        tasks: list[asyncio.Task[Any]] = [send_task, recv_task]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if send_task in done:
                await recv_task
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Tencent ASR WebSocket error: {e}") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            if self._ws is not None:
                await self._ws.close()
                self._ws = None

    async def _send_audio_task(self) -> None:
        if not self._ws:
            return

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._send_pending_audio()
                self._send_usage(flush=True)
                continue

            if isinstance(data, rtc.AudioFrame):
                self._audio_duration += data.duration
                self._pending_audio.extend(data.data.tobytes())
                if len(self._pending_audio) >= self._send_threshold_bytes:
                    await self._send_pending_audio()
                self._send_usage()

        await self._send_pending_audio()
        self._send_usage(flush=True)
        await self._ws.send_str('{"type":"end"}')

    async def _send_pending_audio(self) -> None:
        if not self._pending_audio or not self._ws:
            return
        await self._ws.send_bytes(bytes(self._pending_audio))
        self._pending_audio.clear()

    def _send_usage(self, *, flush: bool = False) -> None:
        if self._audio_duration <= 0:
            return
        now = time.monotonic()
        if not flush and now - self._last_usage_report < 5.0:
            return

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=self._audio_duration),
            )
        )
        self._audio_duration = 0.0
        self._last_usage_report = now

    async def _recv_messages_task(self) -> None:
        if not self._ws:
            return

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = _json_loads(msg.data)
                if self._process_stream_event(data):
                    return
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                return
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"Tencent ASR WebSocket error: {self._ws.exception()}")

    def _process_stream_event(self, data: dict[str, Any]) -> bool:
        self._raise_for_tencent_error(data)

        if data.get("final") == 1:
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        request_id=self._request_id,
                    )
                )
            return True

        result = data.get("result")
        if not isinstance(result, dict):
            return False

        index = _as_int(result.get("index"), default=-1)
        slice_type = _as_int(result.get("slice_type"), default=-1)
        text = str(result.get("voice_text_str") or "")

        if index != self._last_index or slice_type == 0:
            self._last_index = index
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.START_OF_SPEECH,
                        request_id=self._request_id,
                    )
                )

        if text:
            event_type = (
                SpeechEventType.FINAL_TRANSCRIPT
                if slice_type == 2
                else SpeechEventType.INTERIM_TRANSCRIPT
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=event_type,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=text,
                            language=LanguageCode(self._language),
                            start_time=_as_int(result.get("start_time"), default=0) / 1000.0,
                            end_time=_as_int(result.get("end_time"), default=0) / 1000.0,
                            words=_parse_words(result.get("word_list")),
                        )
                    ],
                )
            )

        if slice_type == 2 and self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            )

        return False

    def _raise_for_tencent_error(self, data: dict[str, Any]) -> None:
        code = _as_int(data.get("code"), default=0)
        if code == 0:
            return

        raise APIStatusError(
            message=str(data.get("message") or "Tencent ASR error"),
            status_code=code,
            request_id=str(data.get("voice_id") or self._request_id),
            body=data,
        )


def _json_loads(data: str) -> dict[str, Any]:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as e:
        raise APIConnectionError(f"Invalid Tencent ASR JSON message: {data}") from e
    if not isinstance(decoded, dict):
        raise APIConnectionError(f"Unexpected Tencent ASR JSON payload: {decoded!r}")
    return decoded


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_words(value: Any) -> list[TimedString] | None:
    if not isinstance(value, list) or not value:
        return None

    words: list[TimedString] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        if not isinstance(word, str) or not word:
            continue
        words.append(
            TimedString(
                text=word,
                start_time=_as_int(item.get("start_time"), default=0) / 1000.0,
                end_time=_as_int(item.get("end_time"), default=0) / 1000.0,
            )
        )
    return words or None
