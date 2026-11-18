from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

DEFAULT_BASE_URL = "https://api.giggy.ai/v1"
SAMPLE_RATE = 16000
LANGUAGES = {
    LanguageCode(name).language: name
    for name in (
        "Arabic",
        "Cantonese",
        "Chinese",
        "Czech",
        "Dutch",
        "English",
        "Filipino",
        "French",
        "German",
        "Greek",
        "Hindi",
        "Hungarian",
        "Indonesian",
        "Italian",
        "Japanese",
        "Korean",
        "Malay",
        "Persian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Russian",
        "Spanish",
        "Swedish",
        "Thai",
        "Turkish",
        "Vietnamese",
    )
}
LANGUAGES.update({"yue": "Cantonese", "fil": "Filipino"})
LANGUAGE_CODES = {name: code for code, name in LANGUAGES.items()}


class STT(stt.STT):
    """Giggy speech-to-text.

    Streaming uses one WebSocket per stream; Giggy detects the end of each
    utterance on the server and returns interim and final transcripts.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Giggy STT instance.

        Args:
            api_key: Giggy API key. Defaults to the ``GIGGY_API_KEY`` environment variable.
            language: Supported language name or BCP-47 code. Defaults to automatic detection.
            base_url: Giggy API v1 base URL.
            http_session: Optional aiohttp session to use for requests.
        """
        resolved_key = api_key or os.getenv("GIGGY_API_KEY")
        if not resolved_key:
            raise ValueError("Giggy API key is required, either as argument or set GIGGY_API_KEY")
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._api_key = resolved_key
        self._language = _resolve_language(language)
        self._base_url = base_url.rstrip("/")
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Giggy"

    @property
    def model(self) -> str:
        return "giggyspeech-stt"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=_resolve_language(language) if is_given(language) else self._language,
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        selected = _resolve_language(language) if is_given(language) else self._language
        frame = rtc.combine_audio_frames(buffer)
        if frame.sample_rate != SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                frame.sample_rate, SAMPLE_RATE, num_channels=frame.num_channels
            )
            frame = rtc.combine_audio_frames(resampler.push(frame) + resampler.flush())

        params = {"mode": "streaming", "client_request_id": str(uuid4())}
        if selected:
            params["language"] = selected

        try:
            async with self._ensure_session().post(
                f"{self._base_url}/audio/transcriptions",
                params=params,
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "audio/wav",
                    "Accept": "text/event-stream",
                },
                data=frame.to_wav_bytes(),
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=conn_options.timeout,
                    sock_read=conn_options.timeout,
                ),
            ) as resp:
                if resp.status >= 400:
                    raise APIStatusError(
                        message=resp.reason or "Giggy STT request failed",
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=await resp.text(),
                    )
                event = ""
                async for raw in resp.content:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("event:"):
                        event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    status = data.get("status") or (data.get("generation") or {}).get("status")
                    if event == "error" or (
                        event == "generation_updated"
                        and status in {"failed", "cancelled", "admission_failed"}
                    ):
                        raise APIConnectionError("Giggy STT recognition failed")
                    if event == "transcript_chunk" and data.get("complete") is True:
                        return stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=data.get("generation_uuid") or "",
                            alternatives=[
                                stt.SpeechData(
                                    text=data.get("text") or "",
                                    language=LanguageCode(
                                        data.get("language")
                                        or LANGUAGE_CODES.get(selected or "", "")
                                    ),
                                    end_time=frame.duration,
                                )
                            ],
                        )
            raise APIConnectionError("Giggy STT ended without a final transcript")
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SpeechStream(stt.SpeechStream):
    """Streaming recognition: interim transcripts while the user speaks, a final one
    when Giggy detects the end of the utterance."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str | None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt: STT = stt
        self._code = LANGUAGE_CODES.get(language or "", "")
        self._speaking = False
        self._request_id = ""
        self._stopped = False
        self._audio_sent = 0.0
        self._audio_reported = 0.0
        self._utterance_floor = 0.0

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = self._stt._base_url.replace("http", "ws", 1) + "/audio/transcriptions"
        try:
            return await asyncio.wait_for(
                self._stt._ensure_session().ws_connect(
                    url,
                    autoping=False,
                    headers={"Authorization": f"Bearer {self._stt._api_key}"},
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "Giggy STT refused the connection",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError(type(e).__name__) from None

    async def _run(self) -> None:
        # A retry opens a new connection that knows nothing of an utterance the last one
        # cut off: end it for the consumer and let the new connection open its own.
        if self._speaking:
            self._event_ch.send_nowait(
                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
            )
        self._speaking = False
        self._request_id = ""
        self._stopped = False
        self._audio_sent = 0.0
        self._audio_reported = 0.0
        self._utterance_floor = 0.0
        ws = await self._connect_ws()
        try:
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            await ws.send_str(
                json.dumps(
                    {
                        "type": "start",
                        "format": "raw",
                        "encoding": "LINEAR16",
                        "sampleRateHz": SAMPLE_RATE,
                        "interimResults": True,
                        **({"language": self._code} if self._code else {}),
                    }
                )
            )
            send_task = asyncio.create_task(self._send_task(ws))
            recv_task = asyncio.create_task(self._recv_task(monitor))
            try:
                await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
                if send_task.done():
                    send_task.result()
                    # Input has ended: the last final transcript and the close must follow.
                    await asyncio.wait_for(recv_task, self._conn_options.timeout)
                else:
                    recv_task.result()
            finally:
                await utils.aio.gracefully_cancel(send_task, recv_task)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(type(e).__name__) from None
        finally:
            self._report_usage()
            await ws.close()

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # Giggy accepts at most 1 s of PCM per message, so send fixed 50 ms chunks.
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=SAMPLE_RATE // 20
        )
        async for data in self._input_ch:
            # A flush needs no message: Giggy ends an utterance on its own pause.
            if isinstance(data, rtc.AudioFrame):
                frames = audio_bstream.write(data.data.tobytes())
            else:
                frames = audio_bstream.flush()
            for frame in frames:
                self._audio_sent += frame.duration
                await ws.send_bytes(frame.data.tobytes())
        for frame in audio_bstream.flush():
            self._audio_sent += frame.duration
            await ws.send_bytes(frame.data.tobytes())
        self._stopped = True
        await ws.send_str(json.dumps({"type": "stop"}))

    async def _recv_task(self, monitor: utils.aio.WebSocketMonitor) -> None:
        while True:
            msg = await monitor.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # Giggy closes normally only after "stop" and the last final transcript.
                if self._stopped and msg.data == aiohttp.WSCloseCode.OK:
                    return
                raise APIConnectionError("Giggy STT connection closed unexpectedly")
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("Giggy STT connection failed")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("ignoring a non-JSON frame from Giggy STT")
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type")
            if kind == "transcription":
                self._mark_provider_progress()
                alternatives = data.get("alternatives") or [{}]
                self._on_transcript(
                    str(alternatives[0].get("transcript") or "").strip(),
                    data.get("is_final") is True,
                    data.get("generation_uuid"),
                    data.get("language"),
                )
            elif kind == "error":
                raise APIConnectionError(f"Giggy STT error: {data.get('error')}")
            elif kind not in {
                "generation_created",
                "ready",
                "input_complete",
                "generation_updated",
            }:
                logger.debug("ignoring unknown Giggy STT event", extra={"type": kind})

    def _on_transcript(self, text: str, final: bool, request_id: object, language: object) -> None:
        if not text:
            # An empty final after interims (a cough, noise) still closes the utterance.
            if final and self._speaking:
                self._end_utterance()
            return
        if not self._speaking:
            self._speaking = True
            self._request_id = request_id if isinstance(request_id, str) else utils.shortuuid()
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=(
                    stt.SpeechEventType.FINAL_TRANSCRIPT
                    if final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                ),
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        language=LanguageCode(
                            language if isinstance(language, str) and language else self._code
                        ),
                        text=text,
                        start_time=self.start_time_offset + self._utterance_floor,
                        end_time=self.start_time_offset + self._audio_sent,
                    )
                ],
            )
        )
        if final:
            self._end_utterance()

    def _end_utterance(self) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
        )
        self._report_usage()
        self._utterance_floor = self._audio_sent
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


def _resolve_language(language: str | None) -> str | None:
    """Return Giggy's language name for a name or BCP-47 code, or None for detection."""
    if not language:
        return None
    name = LANGUAGES.get(LanguageCode(language).language)
    if name is None:
        raise ValueError(f"Unsupported Giggy STT language: {language}")
    return name
