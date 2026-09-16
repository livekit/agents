# Copyright 2025 LiveKit, Inc.
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

"""Qwen3-ASR protocol support for `STT` (``model="qwen3-asr"``).

Qwen3-ASR takes base64 audio in OpenAI-realtime frames and replies with
`type: "transcription"` / `segments[].text`, where the streaming-Whisper truss
`STT` also serves sends raw binary PCM and reads `message_type`/`transcript`.
Different protocol, so it gets its own backend and stream implementation here;
`STT` picks between them.

Protocol: a handshake frame, then
`{"type": "input_audio_buffer.append", "audio": <base64 PCM16 16kHz mono>}`,
ended by `{"type": "input_audio_buffer.commit"}`. The server runs Silero VAD, so
turns also end on their own; each turn emits revisable partials
(`is_final: false`, replace-style) then one final.

Word timestamps need `STREAM_ALIGNER=mms` + `STREAM_ALIGNER_DEVICE=cuda` set on
the deployment; without them the server ignores the request.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt as stt_module,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer, is_given

from ._endpoint import resolve_endpoint

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
_SAMPLES_PER_CHUNK = SAMPLE_RATE // 10  # 100 ms, matching the reference client

# LanguageCode already maps most language *names* to codes ("English" -> "en");
# these are the Qwen3-ASR languages it leaves untouched.
_LANGUAGE_OVERRIDES = {"cantonese": "yue", "filipino": "fil"}


def _language_code(name: str | None) -> LanguageCode:
    """Normalize a Qwen3-ASR language name into a LanguageCode."""
    if not name:
        return LanguageCode("")
    key = name.strip().lower()
    return LanguageCode(_LANGUAGE_OVERRIDES.get(key, key))


@dataclass
class _STTOptions:
    audio_language: str
    enable_partial_transcripts: bool
    partial_transcript_interval_s: float
    final_transcript_max_duration_s: float
    vad_threshold: float
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    word_timestamps: bool

    def handshake(self) -> dict[str, Any]:
        return {
            "whisper_params": {
                "audio_language": self.audio_language,
                "show_word_timestamps": self.word_timestamps,
            },
            "streaming_params": {
                "enable_partial_transcripts": self.enable_partial_transcripts,
                "partial_transcript_interval_s": self.partial_transcript_interval_s,
                "final_transcript_max_duration_s": self.final_transcript_max_duration_s,
            },
            "streaming_vad_config": {
                "threshold": self.vad_threshold,
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            },
        }


class _Qwen3Backend:
    """Connection settings for a Qwen3-ASR deployment, owned by `STT`.

    Not an `stt.STT` itself — `STT` selects this when ``model="qwen3-asr"`` and
    keeps the public surface.
    """

    def __init__(
        self,
        *,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        api_key: str | None = None,
        language: str = "auto",
        interim_results: bool = True,
        partial_transcript_interval_s: float = 0.5,
        final_transcript_max_duration_s: float = 30,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 500,
        vad_speech_pad_ms: int = 100,
        word_timestamps: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            model_endpoint: Full ``wss://`` URL of the deployment. Takes priority
                over ``model_id`` and ``chain_id``; falls back to
                ``BASETEN_MODEL_ENDPOINT``.
            model_id: Baseten truss model ID; builds the endpoint URL for you.
            chain_id: Baseten chain ID; builds the endpoint URL for you.
            api_key: Baseten API key, or ``BASETEN_API_KEY``.
            language: A Qwen3-ASR canonical language name (``"English"``,
                ``"Cantonese"``) or ``"auto"``. Forcing a language also sets the
                *output* language and can translate, so use it only to pin what
                is actually being spoken.
            interim_results: Emit revisable partials during a turn. Partials are
                replace-style: each re-states the whole current hypothesis.
            partial_transcript_interval_s: Cadence of those partials.
            final_transcript_max_duration_s: Longest turn before the server
                forces a final.
            vad_threshold: Server-side Silero VAD speech probability threshold.
            vad_min_silence_duration_ms: Silence needed to end a turn.
            vad_speech_pad_ms: Padding added around detected speech.
            word_timestamps: Request word-level alignment on finals. Requires
                ``STREAM_ALIGNER=mms`` and ``STREAM_ALIGNER_DEVICE=cuda`` on the
                deployment; otherwise the server ignores it.
            http_session: Optional aiohttp session to reuse.

        Raises:
            ValueError: If no API key or endpoint can be resolved.
        """
        api_key = api_key or os.environ.get("BASETEN_API_KEY")
        if not api_key:
            raise ValueError("Pass `api_key` or set BASETEN_API_KEY.")

        model_endpoint = resolve_endpoint(
            model_endpoint, model_id, chain_id, "BASETEN_MODEL_ENDPOINT"
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._session = http_session
        self._opts = _STTOptions(
            audio_language=language,
            enable_partial_transcripts=interim_results,
            partial_transcript_interval_s=partial_transcript_interval_s,
            final_transcript_max_duration_s=final_transcript_max_duration_s,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            word_timestamps=word_timestamps,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        partial_transcript_interval_s: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Applies to streams opened after this call; the handshake is per-socket."""
        if is_given(language):
            self._opts.audio_language = language
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(partial_transcript_interval_s):
            self._opts.partial_transcript_interval_s = partial_transcript_interval_s

    def make_stream(
        self,
        owner: stt_module.STT,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> Qwen3SpeechStream:
        """Build a recognition stream owned by the public `STT`."""
        opts = replace(self._opts)
        if is_given(language):
            opts.audio_language = language
        return Qwen3SpeechStream(stt=owner, backend=self, opts=opts, conn_options=conn_options)

    async def recognize_via_stream(
        self,
        owner: stt_module.STT,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt_module.SpeechEvent:
        """One-shot recognition, run over the same streaming session."""
        stream = self.make_stream(owner, language=language, conn_options=conn_options)
        stream.push_frame(rtc.combine_audio_frames(buffer))
        stream.end_input()

        texts: list[str] = []
        code = ""
        try:
            async for ev in stream:
                if ev.type is stt_module.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                    texts.append(ev.alternatives[0].text)
                    code = ev.alternatives[0].language or code
        finally:
            await stream.aclose()

        return stt_module.SpeechEvent(
            type=stt_module.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt_module.SpeechData(language=LanguageCode(code), text=" ".join(texts).strip())
            ],
        )


class Qwen3SpeechStream(stt_module.SpeechStream):
    """A single recognition session: audio frames in, speech events out."""

    def __init__(
        self,
        *,
        stt: stt_module.STT,
        backend: _Qwen3Backend,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._backend = backend
        self._opts = opts

    async def _run(self) -> None:
        try:
            ws = await asyncio.wait_for(
                self._backend._ensure_session().ws_connect(
                    self._backend._model_endpoint,
                    headers={"Authorization": f"Api-Key {self._backend._api_key}"},
                ),
                timeout=self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        input_ended = asyncio.Event()

        async def _send() -> None:
            # LiveKit resamples to SAMPLE_RATE for us; this only re-chunks into
            # the ~100ms frames the server expects.
            chunker = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=_SAMPLES_PER_CHUNK,
            )

            async def append(frames: list[rtc.AudioFrame]) -> None:
                for frame in frames:
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(frame.data.tobytes()).decode(),
                            }
                        )
                    )

            # `end_input()` pushes a flush sentinel *and then* closes the channel,
            # so this has to remember whether the last thing it did was commit —
            # otherwise the trailing commit below doubles it and the server
            # answers an extra, empty turn.
            committed = False
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        await append(chunker.flush())
                        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
                        committed = True
                    else:
                        await append(chunker.write(data.data.tobytes()))
                        committed = False
                if not committed:
                    await append(chunker.flush())
                    await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            finally:
                input_ended.set()

        async def _recv() -> None:
            speaking = False
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if input_ended.is_set():
                        return
                    raise APIConnectionError("Baseten closed the STT websocket")
                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"websocket error: {ws.exception()}")
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                if event.get("type") == "error":
                    raise APIStatusError(
                        message=str(event.get("message", "unknown STT error")),
                        status_code=500,
                        request_id=None,
                        body=None,
                    )
                if event.get("type") != "transcription":
                    continue

                segments = event.get("segments") or []
                text = " ".join(s.get("text", "") for s in segments).strip()
                is_final = bool(event.get("is_final"))

                # Only real words open a turn. An empty final (VAD closed on
                # noise, recognizer returned nothing) would otherwise emit
                # START/END_OF_SPEECH and, under turn_detection="stt", commit
                # the user's turn so the agent answers silence.
                if not speaking and text:
                    speaking = True
                    self._event_ch.send_nowait(
                        stt_module.SpeechEvent(type=stt_module.SpeechEventType.START_OF_SPEECH)
                    )

                if text:
                    self._event_ch.send_nowait(
                        stt_module.SpeechEvent(
                            type=stt_module.SpeechEventType.FINAL_TRANSCRIPT
                            if is_final
                            else stt_module.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[self._speech_data(event, segments, text)],
                        )
                    )

                if is_final:
                    if speaking:
                        self._event_ch.send_nowait(
                            stt_module.SpeechEvent(type=stt_module.SpeechEventType.END_OF_SPEECH)
                        )
                        speaking = False
                    # Only the commit-triggered final ends the session; VAD
                    # finals just close a turn and the stream keeps going.
                    if event.get("is_end_of_audio_flush") and input_ended.is_set():
                        return

        try:
            await ws.send_str(json.dumps(self._opts.handshake()))
            send_task = asyncio.create_task(_send(), name="qwen3-stt-send")
            recv_task = asyncio.create_task(_recv(), name="qwen3-stt-recv")
            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                await utils.aio.gracefully_cancel(send_task, recv_task)
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await ws.close()

    def _speech_data(
        self, event: dict[str, Any], segments: list[dict[str, Any]], text: str
    ) -> stt_module.SpeechData:
        words = None
        if self._opts.word_timestamps:
            words = [
                TimedString(
                    text=w.get("word", ""),
                    start_time=float(w.get("start_time", 0.0)) + self.start_time_offset,
                    end_time=float(w.get("end_time", 0.0)) + self.start_time_offset,
                    confidence=float(w.get("prob", 0.0)),
                    start_time_offset=self.start_time_offset,
                )
                for s in segments
                for w in (s.get("word_timestamps") or [])
            ] or None

        return stt_module.SpeechData(
            language=_language_code(event.get("language_code")),
            text=text,
            # Server times are socket-relative; the framework grows
            # start_time_offset on every reconnect so consumers see one timeline.
            start_time=(float(segments[0].get("start_time", 0.0)) if segments else 0.0)
            + self.start_time_offset,
            end_time=(float(segments[-1].get("end_time", 0.0)) if segments else 0.0)
            + self.start_time_offset,
            confidence=1.0,
            words=words,
        )
