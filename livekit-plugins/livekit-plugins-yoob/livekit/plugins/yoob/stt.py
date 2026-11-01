from __future__ import annotations

import asyncio
import io
import json
import os
import wave
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import DEFAULT_BASE_URL, STT_SAMPLE_RATE, STTModels

# The transcription models are English-first; transcripts are tagged "en".
_LANGUAGE = LanguageCode("en")


@dataclass
class _STTOptions:
    model: STTModels | str
    interim_results: bool
    base_url: str
    api_key: str

    def ws_url(self) -> str:
        return self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + (
            "/audio/transcriptions/stream"
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "yoob-asr-1",
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Yoob STT (Undertone).

        Args:
            model: ``yoob-asr-1`` (default) or ``yoob-asr-2``. Used for batch ``recognize()`` of up to 60 s;
                streaming always uses the live model.
            interim_results: Emit partial transcripts while the user speaks.
            api_key: Defaults to the ``YOOB_API_KEY`` environment variable. Keys come from https://app.yoob.com.
            base_url: Defaults to ``YOOB_BASE_URL`` or https://tts-api.yoob.com/v1.
            http_session: An existing aiohttp session to reuse.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                aligned_transcript=False,
                manual_flush=True,
            )
        )
        key = api_key or os.environ.get("YOOB_API_KEY")
        if not key:
            raise ValueError("Yoob API key is required: pass api_key or set YOOB_API_KEY")
        self._opts = _STTOptions(
            model=model,
            interim_results=interim_results,
            base_url=(base_url or os.environ.get("YOOB_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            api_key=key,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Yoob"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._opts.api_key}"}

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(frame.num_channels)
            w.setsampwidth(2)
            w.setframerate(frame.sample_rate)
            w.writeframes(frame.data.tobytes())
        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/audio/transcriptions",
                params={"model": self._opts.model, "response_format": "verbose_json"},
                data=wav.getvalue(),
                headers={**self._headers(), "Content-Type": "audio/wav"},
                timeout=aiohttp.ClientTimeout(total=60, sock_connect=conn_options.timeout),
            ) as resp:
                body = await resp.text()
                request_id = resp.headers.get("x-request-id") or ""
                if resp.status >= 400:
                    raise APIStatusError(
                        _error_message(body), status_code=resp.status, request_id=request_id, body=body
                    )
                data = json.loads(body)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        segments = data.get("segments") or []
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[
                stt.SpeechData(
                    language=_LANGUAGE,
                    text=data.get("text") or "",
                    start_time=float(segments[0].get("start", 0)) if segments else 0.0,
                    end_time=float(segments[-1].get("end", 0)) if segments else 0.0,
                    confidence=1.0,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, opts=replace(self._opts), conn_options=conn_options)


def _error_message(body: str) -> str:
    try:
        err = json.loads(body).get("error") or {}
        return str(err.get("message") or body)
    except (ValueError, AttributeError):
        return body


class SpeechStream(stt.SpeechStream):
    """Live transcription over Yoob's transcription socket (16 kHz s16le mono in, events out)."""

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=STT_SAMPLE_RATE)
        self._stt: STT = stt
        self._opts = opts
        self._request_id = ""

    async def _run(self) -> None:
        session = self._stt._ensure_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(self._opts.ws_url(), headers=self._stt._headers()),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status) from None
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        try:
            ready = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
            if ready.type != aiohttp.WSMsgType.TEXT:
                raise APIStatusError(
                    "Yoob transcription socket closed before it was ready",
                    status_code=ws.close_code or -1,
                )
            ready_msg = json.loads(ready.data)
            if ready_msg.get("type") == "error":
                raise _socket_error(ready_msg)
            self._request_id = str(ready_msg.get("id") or utils.shortuuid())

            async def send_task() -> None:
                # 100 ms frames; the socket takes frames of at most 1 s.
                bstream = utils.audio.AudioByteStream(
                    sample_rate=STT_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=STT_SAMPLE_RATE // 10,
                )
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        for f in bstream.flush():
                            await ws.send_bytes(f.data.tobytes())
                        # close the current utterance now instead of waiting for the server's VAD
                        await ws.send_str(json.dumps({"type": "end"}))
                        continue
                    for f in bstream.write(data.data.tobytes()):
                        await ws.send_bytes(f.data.tobytes())
                for f in bstream.flush():
                    await ws.send_bytes(f.data.tobytes())
                await ws.send_str(json.dumps({"type": "stop"}))

            async def recv_task() -> None:
                speaking = False
                pending_end = False  # VAD said the utterance ended; its final is still coming

                def end_of_speech() -> None:
                    nonlocal speaking, pending_end
                    if speaking:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                            )
                        )
                    speaking = pending_end = False

                def start_of_speech() -> None:
                    nonlocal speaking
                    if pending_end:
                        end_of_speech()
                    if not speaking:
                        speaking = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                            )
                        )

                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        end_of_speech()
                        if self._input_ch.closed:
                            return
                        raise APIStatusError(
                            "Yoob transcription socket closed unexpectedly",
                            status_code=ws.close_code or -1,
                        )
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data: dict[str, Any] = json.loads(msg.data)
                    kind = data.get("type")
                    if kind == "error":
                        raise _socket_error(data)
                    if kind == "vad":
                        if data.get("speech"):
                            start_of_speech()
                        elif speaking:
                            pending_end = True
                    elif kind == "partial":
                        if not self._opts.interim_results or not data.get("text"):
                            continue
                        start_of_speech()
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                request_id=self._request_id,
                                alternatives=[self._speech_data(data)],
                            )
                        )
                    elif kind == "final":
                        if data.get("text"):
                            start_of_speech()
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    request_id=self._request_id,
                                    alternatives=[self._speech_data(data)],
                                )
                            )
                        if pending_end:
                            end_of_speech()
                    elif kind == "done":
                        end_of_speech()
                        return

            tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            await ws.close()

    def _speech_data(self, ev: dict[str, Any]) -> stt.SpeechData:
        return stt.SpeechData(
            language=_LANGUAGE,
            text=str(ev.get("text") or ""),
            start_time=float(ev.get("start") or 0.0) + self.start_time_offset,
            end_time=float(ev.get("end") or 0.0) + self.start_time_offset,
            confidence=1.0,
        )


def _socket_error(msg: dict[str, Any]) -> APIStatusError:
    code = msg.get("code") or "error"
    status = {"invalid_api_key": 401, "insufficient_credits": 402, "busy": 429}.get(code, 400)
    logger.debug("yoob transcription socket error", extra={"code": code})
    return APIStatusError(
        f"Yoob: {msg.get('message') or code}", status_code=status, body=json.dumps(msg)
    )
