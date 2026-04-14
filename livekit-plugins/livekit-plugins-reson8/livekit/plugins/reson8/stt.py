from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, replace
from urllib.parse import urlencode

import httpx
import websockets
from livekit.agents import stt, utils
from livekit.agents._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given
from websockets.asyncio.client import ClientConnection

from livekit import rtc


@dataclass
class STTOptions:
    api_key: str
    api_url: str
    language: str
    custom_model_id: str | None
    sample_rate: int
    include_timestamps: bool
    include_words: bool
    include_confidence: bool

    def query_params(self, *, interim: bool) -> dict[str, str]:
        params: dict[str, str] = {
            "language": self.language,
            "encoding": "pcm_s16le",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
        }
        if interim:
            params["include_interim"] = "true"
        if self.custom_model_id:
            params["custom_model_id"] = self.custom_model_id
        if self.include_timestamps:
            params["include_timestamps"] = "true"
        if self.include_words:
            params["include_words"] = "true"
        if self.include_confidence:
            params["include_confidence"] = "true"
        return params


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        language: str = "nl",
        custom_model_id: str | None = None,
        sample_rate: int = 16000,
        include_timestamps: bool = False,
        include_words: bool = False,
        include_confidence: bool = False,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        resolved_key = api_key or os.environ.get("RESON8_API_KEY", "")
        if not resolved_key:
            raise ValueError("Reson8 API key is required. Set RESON8_API_KEY or pass api_key=")
        self._opts = STTOptions(
            api_key=resolved_key,
            api_url=api_url or os.environ.get("RESON8_API_URL", "https://api.reson8.dev"),
            language=language,
            custom_model_id=custom_model_id,
            sample_rate=sample_rate,
            include_timestamps=include_timestamps,
            include_words=include_words,
            include_confidence=include_confidence,
        )

    @property
    def model(self) -> str:
        return "reson8-stt"

    @property
    def provider(self) -> str:
        return "reson8"

    def _resolve_opts(self, language: NotGivenOr[str]) -> STTOptions:
        if is_given(language):
            return replace(self._opts, language=language)
        return self._opts

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, opts=self._resolve_opts(language), conn_options=conn_options)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = self._resolve_opts(language)
        audio = rtc.combine_audio_frames(buffer).data.tobytes()
        url = f"{opts.api_url.rstrip('/')}/v1/speech-to-text/prerecorded?{urlencode(opts.query_params(interim=False))}"

        try:
            async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                resp = await client.post(
                    url,
                    content=audio,
                    headers={
                        "Authorization": f"ApiKey {opts.api_key}",
                        "Content-Type": "application/octet-stream",
                    },
                )
        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except httpx.HTTPError as e:
            raise APIConnectionError(str(e)) from e

        if resp.status_code != 200:
            raise APIStatusError(
                message=f"Reson8 prerecorded error {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=resp.json().get("text", ""), language=LanguageCode(opts.language))],
        )


class SpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts

    async def _run(self) -> None:
        base = self._opts.api_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        url = f"{base}/v1/speech-to-text/realtime?{urlencode(self._opts.query_params(interim=True))}"

        try:
            ws = await websockets.connect(
                url, additional_headers={"Authorization": f"ApiKey {self._opts.api_key}"}, proxy=None
            )
        except websockets.exceptions.InvalidStatus as e:
            raise APIStatusError(
                message=f"Reson8 WebSocket rejected: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise APIConnectionError(f"Failed to connect to Reson8: {e}") from e

        async with ws:
            send_task = asyncio.create_task(self._send_loop(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))
            try:
                done, _ = await asyncio.wait(
                    [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
            finally:
                for t in (send_task, recv_task):
                    if not t.done():
                        t.cancel()
                await ws.close()

    async def _send_loop(self, ws: ClientConnection) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                await ws.send(data.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                await ws.send(json.dumps({"type": "flush_request", "id": str(uuid.uuid4())}))

    async def _recv_loop(self, ws: ClientConnection) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") != "transcript":
                continue
            event_type = (
                stt.SpeechEventType.FINAL_TRANSCRIPT
                if msg.get("is_final", True)
                else stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=event_type,
                    alternatives=[
                        stt.SpeechData(text=msg.get("text", ""), language=LanguageCode(self._opts.language)),
                    ],
                )
            )
