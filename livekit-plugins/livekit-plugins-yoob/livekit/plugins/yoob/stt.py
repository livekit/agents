from __future__ import annotations

import asyncio
import io
import json
import os
import wave
import weakref
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp
import numpy as np

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

# Without a language, a stream transcribes English (the API's default).
_DEFAULT_LANGUAGE = "en"

# English names the API reports in verbose_json, mapped to codes.
_LANGUAGE_CODES = {
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "dutch": "nl",
    "portuguese": "pt",
    "polish": "pl",
    "russian": "ru",
    "ukrainian": "uk",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "czech": "cs",
    "slovak": "sk",
    "slovenian": "sl",
    "croatian": "hr",
    "bulgarian": "bg",
    "romanian": "ro",
    "hungarian": "hu",
    "greek": "el",
    "estonian": "et",
    "latvian": "lv",
    "lithuanian": "lt",
    "maltese": "mt",
    "arabic": "ar",
    "hebrew": "he",
    "hindi": "hi",
    "bengali": "bn",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "turkish": "tr",
    "persian": "fa",
}


def _language_code(name: str | None) -> LanguageCode:
    name = (name or _DEFAULT_LANGUAGE).strip()
    return LanguageCode(_LANGUAGE_CODES.get(name.lower(), name))


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str | None
    interim_results: bool
    base_url: str
    api_key: str

    def ws_url(self) -> str:
        # The stream reads its model and language from the URL when it opens.
        query = {"model": self.model}
        if self.language:
            query["language"] = self.language
        return (
            self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            + "/audio/transcriptions/stream?"
            + urlencode(query)
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "yoob-asr-turbo-v1",
        language: str | None = None,
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Yoob STT (Undertone).

        Args:
            model: ``yoob-asr-turbo-v1`` (default, fastest; 25 languages) or ``yoob-asr-pro-v1`` (most accurate;
                47 languages), for both streaming and batch ``recognize()`` of up to 60 s. ``recognize()`` waits
                ``conn_options.timeout`` for the answer.
            language: The spoken language as a code (``de``), a regional tag (``de-DE``) or an English name.
                Streams default to English; ``recognize()`` without it detects the language (Pro).
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
            language=language,
            interim_results=interim_results,
            base_url=(base_url or os.environ.get("YOOB_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            api_key=key,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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
        lang = language if is_given(language) else self._opts.language
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
                params={
                    "model": self._opts.model,
                    "response_format": "verbose_json",
                    **({"language": lang} if lang else {}),
                },
                data=wav.getvalue(),
                headers={**self._headers(), "Content-Type": "audio/wav"},
                # one budget for the pooled connection, the upload and the whole answer
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                body = await resp.text()
                request_id = resp.headers.get("x-request-id") or ""
                if resp.status >= 400:
                    raise _http_error(resp.status, body, request_id)
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
                    language=_language_code(data.get("language") or lang),
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
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


def _http_error(status: int, body: str, request_id: str) -> APIStatusError:
    # Only the API's error code: no provider text and no raw body, so retry logs never carry request content.
    code = None
    try:
        code = (json.loads(body).get("error") or {}).get("code")
    except (ValueError, AttributeError):
        pass
    return APIStatusError(
        f"Yoob transcription request failed ({code or f'HTTP {status}'})",
        status_code=status,
        request_id=request_id,
        body={"code": code} if code else None,
    )


class SpeechStream(stt.SpeechStream):
    """Live transcription over Yoob's transcription socket (16 kHz s16le mono in, events out)."""

    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=STT_SAMPLE_RATE)
        self._stt: STT = stt
        self._opts = opts
        self._request_id = ""

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        # The socket takes mono. Mix multichannel frames down before the base class resamples and queues
        # them; sent as is, interleaved stereo would arrive as one channel at twice its length.
        if frame.num_channels > 1:
            pcm = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
            frame = rtc.AudioFrame(
                data=pcm.mean(axis=1).astype(np.int16).tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=1,
                samples_per_channel=frame.samples_per_channel,
            )
        super().push_frame(frame)

    async def _run(self) -> None:
        session = self._stt._ensure_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._opts.ws_url(),
                    headers=self._stt._headers(),
                    # The monitor answers pings and probes an idle socket, so a half-open
                    # connection fails within conn_options.liveness_timeout.
                    autoping=False,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status) from None
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        monitor = utils.aio.WebSocketMonitor(
            ws, liveness_timeout=self._conn_options.liveness_timeout
        )
        try:
            ready = await asyncio.wait_for(monitor.receive(), self._conn_options.timeout)
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
                last_end: float | None = None  # where this utterance's latest final ends

                def end_of_speech() -> None:
                    nonlocal speaking, pending_end, last_end
                    if speaking:
                        started_at = self._input_ch.input_started_at
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                request_id=self._request_id,
                                speech_end_time=(
                                    started_at + last_end
                                    if started_at is not None and last_end is not None
                                    else None
                                ),
                            )
                        )
                    speaking = pending_end = False
                    last_end = None

                def ensure_started() -> None:
                    # A transcript belongs to the open utterance, even one VAD has already
                    # closed, or opens one: it never ends a turn itself.
                    nonlocal speaking
                    if not speaking:
                        speaking = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.START_OF_SPEECH,
                                request_id=self._request_id,
                            )
                        )

                while True:
                    msg = await monitor.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        # closed before "done". Partials are never committed as finals: they are unconfirmed,
                        # and a retry could repeat or contradict them. After end_input() the results are lost
                        # for good, so the caller gets a non-retryable error instead of a quiet success.
                        end_of_speech()
                        if self._input_ch.closed:
                            raise APIConnectionError(
                                "Yoob transcription socket closed before the last results",
                                retryable=False,
                            )
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
                            if pending_end:
                                # new speech before the last utterance's final: that one is over
                                end_of_speech()
                            ensure_started()
                        elif speaking:
                            pending_end = True
                    elif kind == "partial":
                        if not self._opts.interim_results or not data.get("text"):
                            continue
                        ensure_started()
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                request_id=self._request_id,
                                alternatives=[self._speech_data(data)],
                            )
                        )
                    elif kind == "final":
                        if data.get("text"):
                            ensure_started()
                            final = self._speech_data(data)
                            if final.end_time > 0:
                                last_end = final.end_time
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    request_id=self._request_id,
                                    alternatives=[final],
                                )
                            )
                        if pending_end:
                            end_of_speech()
                    elif kind == "done":
                        end_of_speech()
                        return

            sender = asyncio.create_task(send_task())
            receiver = asyncio.create_task(recv_task())
            try:
                await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if not sender.done():
                    receiver.result()
                else:
                    sender.result()
                    # Input has ended: the last finals and "done" must follow within the timeout. A live
                    # stream waits on the caller for as long as it lasts; this bounds only the wait for a
                    # server that stops answering after "stop". The audio is spent, so this is final.
                    try:
                        await asyncio.wait_for(receiver, self._conn_options.timeout)
                    except asyncio.TimeoutError:
                        raise APITimeoutError(
                            "Yoob transcription sent no last results in time", retryable=False
                        ) from None
            except APIError as e:
                # Once input has ended, the audio this connection consumed cannot be replayed: a retry would
                # finish without its results, and without an error. A retryable failure (a lost connection, Yoob
                # busy) is final then. While input is still open, a retry carries on with the audio still to come.
                if e.retryable and self._input_ch.closed:
                    raise _final(e) from e
                raise
            finally:
                await utils.aio.gracefully_cancel(sender, receiver)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            # A send on a socket that dropped mid-stream raises before the receiver sees the close. Once input has
            # ended, the frames this connection consumed cannot be replayed: a retry would quietly lose them.
            raise APIConnectionError(retryable=not self._input_ch.closed) from e
        finally:
            await ws.close()

    def _speech_data(self, ev: dict[str, Any]) -> stt.SpeechData:
        return stt.SpeechData(
            language=_language_code(self._opts.language),
            text=str(ev.get("text") or ""),
            start_time=float(ev.get("start") or 0.0) + self.start_time_offset,
            end_time=float(ev.get("end") or 0.0) + self.start_time_offset,
            confidence=1.0,
        )


def _final(e: APIError) -> APIError:
    """`e`, not retryable: the stream's audio was spent on the connection that failed."""
    if isinstance(e, APIStatusError):
        return APIStatusError(
            e.message,
            status_code=e.status_code,
            request_id=e.request_id,
            body=e.body,
            retryable=False,
        )
    if isinstance(e, APITimeoutError):
        return APITimeoutError(e.message, retryable=False)
    if isinstance(e, APIConnectionError):
        return APIConnectionError(
            "Yoob transcription connection lost before the last results", retryable=False
        )
    return APIError(e.message, body=e.body, retryable=False)


def _socket_error(msg: dict[str, Any]) -> APIStatusError:
    code = msg.get("code") or "error"
    status = {"invalid_api_key": 401, "insufficient_credits": 402, "busy": 429}.get(code, 400)
    logger.debug("yoob transcription socket error", extra={"code": code})
    return APIStatusError(
        f"Yoob transcription error ({code})", status_code=status, body={"code": code}
    )
