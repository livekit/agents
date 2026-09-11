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

"""Speech-to-text over Model Studio's realtime WebSocket (``qwen3-asr-flash-realtime``).

Model Studio's realtime endpoint borrows OpenAI's event *names* but not its schema: the
model goes in the query string, the session config is flat (``input_audio_format`` /
``sample_rate`` rather than OpenAI's nested ``audio.input``), interim text arrives as
``...input_audio_transcription.text`` instead of ``.delta``, and the sample rate must be
16 kHz where the OpenAI plugin hardcodes 24 kHz. There is also no ``/audio/transcriptions``
endpoint, so ``openai.STT(base_url=...)`` cannot be reused for either path.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._realtime import (
    CLOSE_TYPES,
    DEFAULT_FINISH_TIMEOUT,
    RealtimeSocket,
    connect,
    status_error_from,
)
from ._utils import resolve_api_key, resolve_realtime_url
from .log import logger
from .models import DEFAULT_REGION, DEFAULT_STT_MODEL, STT_SAMPLE_RATE, QwenRegion, STTModels

# 100 ms per append: small enough to keep transcription latency down, large enough to
# stay well clear of the 15 MiB-per-event manual-mode cap.
_CHUNK_SAMPLES = STT_SAMPLE_RATE // 10


@dataclass
class _STTOptions:
    model: str
    language: str | None
    vad_silence_duration_ms: int | None
    vad_threshold: float | None
    finish_timeout: float

    def turn_detection(self) -> dict[str, Any]:
        """Model Studio's server VAD block; unset fields keep the server defaults."""
        turn_detection: dict[str, Any] = {"type": "server_vad"}
        if self.vad_silence_duration_ms is not None:
            turn_detection["silence_duration_ms"] = self.vad_silence_duration_ms
        if self.vad_threshold is not None:
            turn_detection["threshold"] = self.vad_threshold
        return turn_detection


class STT(stt.STT):
    """Streaming speech-to-text on Alibaba Cloud Model Studio (Qwen3-ASR realtime)."""

    def __init__(
        self,
        *,
        model: STTModels | str = DEFAULT_STT_MODEL,
        language: str | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_threshold: float | None = None,
        finish_timeout: float = DEFAULT_FINISH_TIMEOUT,
        region: QwenRegion = DEFAULT_REGION,
        base_url: str | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Qwen realtime STT.

        Args:
            model: Model Studio realtime ASR model id.
            language: ISO code such as ``"zh"`` or ``"en"``. Leave ``None`` to let the model
                detect the language, which is also Model Studio's documented setting for
                mixed-language speech.
            vad_silence_duration_ms: Silence that ends an utterance on the server;
                ``None`` keeps the server default (800 ms).
            vad_threshold: Server VAD speech-probability threshold; ``None`` keeps the
                server default (0.2).
            finish_timeout: Seconds to wait for ``session.finished`` after asking the server
                to finish. A server that never answers raises ``APITimeoutError`` instead of
                stalling the session.
            region: ``"intl"`` (Singapore) or ``"cn"`` (Beijing). API keys are region-bound.
            base_url: Full realtime WebSocket URL, for example a workspace-dedicated domain
                ``wss://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime``.
                Overrides ``region``.
            api_key: Model Studio API key; falls back to ``DASHSCOPE_API_KEY``.
            http_session: Optional aiohttp session to reuse.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._api_key = resolve_api_key(api_key)
        self._base_url = resolve_realtime_url(base_url, region)
        self._opts = _STTOptions(
            model=model,
            language=language,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_threshold=vad_threshold,
            finish_timeout=finish_timeout,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Qwen"

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        vad_silence_duration_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update options for streams opened after this call; the session config is per socket."""
        if is_given(language):
            self._opts.language = language
        if is_given(vad_silence_duration_ms):
            self._opts.vad_silence_duration_ms = vad_silence_duration_ms
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _options_for(self, language: NotGivenOr[str]) -> _STTOptions:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        return opts

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """One-shot recognition over the streaming transport.

        Model Studio has no OpenAI-style ``/audio/transcriptions`` endpoint to batch
        against. The utterance boundary is known here, so the buffer is committed
        explicitly rather than waiting on the server's silence window.
        """
        opts = self._options_for(language)
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options, manual_commit=True)
        stream.push_frame(rtc.combine_audio_frames(buffer))
        stream.end_input()
        final: stt.SpeechEvent | None = None
        try:
            async for event in stream:
                if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT:
                    final = event
        finally:
            await stream.aclose()

        if final is not None:
            return final
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode(opts.language or ""), text="")],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, opts=self._options_for(language), conn_options=conn_options)


class SpeechStream(stt.RecognizeStream):
    """One realtime ASR session: audio frames in, speech events out."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        manual_commit: bool = False,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=STT_SAMPLE_RATE)
        self._qwen = stt
        self._opts = opts
        # Manual mode (turn_detection null) commits on each flush sentinel. Only
        # recognize() asks for it: that path calls end_input(), which is the one place
        # LiveKit ever flushes an STT stream. Live streams always run with server VAD.
        self._manual_commit = manual_commit

    def _session_config(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {}
        if self._opts.language:
            # Left unset the model detects the language, which is also the documented way
            # to handle mixed-language speech.
            transcription["language"] = self._opts.language
        return {
            "input_audio_format": "pcm",
            "sample_rate": STT_SAMPLE_RATE,
            "input_audio_transcription": transcription,
            "turn_detection": None if self._manual_commit else self._opts.turn_detection(),
        }

    def _emit_transcript(self, kind: stt.SpeechEventType, text: str, event: dict[str, Any]) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=kind,
                request_id=event.get("event_id") or "",
                alternatives=[
                    stt.SpeechData(
                        # Absent when the model was told the language up front.
                        language=LanguageCode(event.get("language") or self._opts.language or ""),
                        text=text,
                    )
                ],
            )
        )

    async def _run(self) -> None:
        socket = RealtimeSocket(
            await connect(
                self._qwen._ensure_session(),
                base_url=self._qwen._base_url,
                model=self._opts.model,
                api_key=self._qwen._api_key,
                timeout=self._conn_options.timeout,
            ),
            finish_timeout=self._opts.finish_timeout,
        )
        uncommitted = False
        pushed_samples = 0
        reported_samples = 0

        async def append(frames: list[rtc.AudioFrame]) -> None:
            nonlocal uncommitted, pushed_samples
            for frame in frames:
                await socket.send(
                    "input_audio_buffer.append",
                    audio=base64.b64encode(frame.data.tobytes()).decode(),
                )
                uncommitted = True
                pushed_samples += frame.samples_per_channel

        def report_usage() -> None:
            # Model Studio bills ASR per second of input audio and sends no usage of its
            # own, so report what we streamed. It goes out with each final rather than at
            # the end: a live session is torn down with aclose(), which cancels this task
            # before any epilogue runs.
            nonlocal reported_samples
            unreported = pushed_samples - reported_samples
            if unreported <= 0:
                return
            reported_samples = pushed_samples
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=unreported / STT_SAMPLE_RATE
                    ),
                )
            )

        async def commit() -> None:
            nonlocal uncommitted
            # end_input() flushes on its way out, so a flush()+end_input() pair reaches us
            # as two sentinels. Committing an empty buffer is an error on Model Studio, so
            # only commit what was actually appended.
            if not uncommitted:
                return
            await socket.send("input_audio_buffer.commit")
            uncommitted = False

        async def send() -> None:
            chunker = utils.audio.AudioByteStream(
                sample_rate=STT_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=_CHUNK_SAMPLES,
            )
            try:
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        # Flush the partial chunk first, else the tail of the utterance is
                        # committed without ever being sent.
                        await append(chunker.flush())
                        if self._manual_commit:
                            await commit()
                    else:
                        await append(chunker.write(item.data.tobytes()))
                # No tail to flush here: end_input() sends a flush sentinel before closing
                # the channel, so the branch above has run.
                await socket.finish()
            except ConnectionResetError:
                # The socket died first; recv() owns reporting that.
                pass

        async def recv() -> None:
            speaking = False
            while True:
                msg = await socket.receive()
                if msg.type in CLOSE_TYPES:
                    # Only `session.finished` ends a stream cleanly. Treating a bare close
                    # as success would hide an upstream failure and rob the
                    # FallbackAdapter of its cue to try the next STT.
                    raise APIConnectionError(
                        "Model Studio closed the ASR socket before session.finished"
                    )
                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Model Studio ASR websocket error")
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                event: dict[str, Any] = json.loads(msg.data)
                kind = event.get("type")

                if kind == "error":
                    raise status_error_from(event)
                elif kind == "input_audio_buffer.speech_started":
                    speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                elif kind == "conversation.item.input_audio_transcription.text":
                    # `text` is the confirmed prefix, `stash` the tail the model may still
                    # revise; an interim result wants both.
                    text = (event.get("text") or "") + (event.get("stash") or "")
                    if text:
                        self._emit_transcript(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, event)
                elif kind == "conversation.item.input_audio_transcription.completed":
                    self._emit_transcript(
                        stt.SpeechEventType.FINAL_TRANSCRIPT, event.get("transcript") or "", event
                    )
                    report_usage()
                    if speaking:
                        # Model Studio has no end-of-speech event, but LiveKit's turn
                        # detection needs the pair closed.
                        speaking = False
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                elif kind == "conversation.item.input_audio_transcription.failed":
                    # Per-utterance failure. Raising here would drop the socket and send the
                    # FallbackAdapter to the next provider over one bad segment; LiveKit
                    # falls back to the interim text.
                    logger.warning(
                        "qwen asr failed to transcribe an utterance; keeping the stream open",
                        extra={"lk.pii.data": event},
                    )
                elif kind == "session.finished":
                    return

        cancelled = False
        try:
            await socket.send("session.update", session=self._session_config())
            send_task = asyncio.create_task(send(), name="qwen-stt-send")
            recv_task = asyncio.create_task(recv(), name="qwen-stt-recv")
            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                await utils.aio.cancel_and_wait(send_task, recv_task)

            # Audio streamed after the last final (or with no final at all) on a clean
            # end, e.g. the one-shot path.
            report_usage()
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if cancelled:
                # aclose() is how LiveKit ends a live stream, and it lands here before
                # send() ever reached session.finish. Model Studio books a socket dropped
                # without the handshake as a failed request, so finish it on the way out,
                # bounded, since this is on the session-teardown path. Error exits skip
                # it: the FallbackAdapter is waiting to try the next provider.
                await socket.close_gracefully()
            else:
                await socket.close()
