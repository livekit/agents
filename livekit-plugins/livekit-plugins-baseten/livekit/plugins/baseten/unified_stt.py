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

"""Baseten unified streaming STT support for `STT` (``model="unified"``).

One wire protocol in front of every model Baseten serves through its unified
STT server (Whisper, Qwen3-ASR, and later ones), so a deployment's model no
longer decides which plugin code talks to it.

Protocol: the server speaks first with ``session.created``. The client asks for
its configuration with ``session.update`` and is answered by
``session.updated``, whose ``refusals`` list anything it did not grant. Audio
goes up as ``{"type": "input_audio_buffer.append", "audio": <base64 PCM16 16kHz
mono>}``; ``input_audio_buffer.flush`` ends a turn, and ``close`` asks the
server to deliver what it still owes and close the socket. Every event is
strict: an unknown field ends the session with ``error`` and a 1008 close.

Each turn emits revisable partials (``transcription.delta``, replace-style) and
then one ``transcription.done``, or ``transcription.failed`` if that turn alone
could not be decoded. Who ends a turn is negotiated per session:
``server_vad`` (trailing silence) or ``client_flush`` (only a flush does).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._endpoint import resolve_endpoint
from .log import logger

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
_SAMPLES_PER_CHUNK = SAMPLE_RATE // 10  # 100 ms; the server re-frames any size

SegmentationMode = Literal["server_vad", "client_flush"]

# The error codes the server sends before it closes, as the status a retry
# decision is made on: a malformed request fails the same way every time, while
# a full replica or a server fault may not on the next connection.
_ERROR_STATUS = {"invalid_event": 400, "at_capacity": 503, "internal_error": 500}


@dataclass
class _STTOptions:
    segmentation_mode: SegmentationMode
    language: str | None
    enable_partial_transcripts: bool
    partial_transcript_interval_s: float
    vad_threshold: float | None
    vad_min_silence_duration_ms: int | None

    def session_update(self) -> dict[str, Any]:
        # Omitted fields take the deployment's defaults, which differ per model,
        # so only what the caller actually chose is sent. `language` especially:
        # servers older than v0.2.0 have no such field and would close on it.
        session: dict[str, Any] = {
            "input_format": {
                "encoding": "pcm_s16le",
                "sample_rate_hz": SAMPLE_RATE,
                "channels": NUM_CHANNELS,
            },
            "segmentation_mode": self.segmentation_mode,
        }
        if self.enable_partial_transcripts:
            session["partial_interval_s"] = self.partial_transcript_interval_s
        if self.language is not None:
            session["language"] = self.language
        # The server refuses VAD settings under client_flush, where no silence
        # ends a turn, so they are only sent where they do something.
        vad: dict[str, Any] = {}
        if self.vad_threshold is not None:
            vad["threshold"] = self.vad_threshold
        if self.vad_min_silence_duration_ms is not None:
            vad["silence_s"] = self.vad_min_silence_duration_ms / 1000
        if vad and self.segmentation_mode == "server_vad":
            session["vad"] = vad
        return {"type": "session.update", "session": session}


class _UnifiedBackend:
    """Connection settings for a Baseten unified STT deployment, owned by `STT`.

    Not an `stt.STT` itself — `STT` selects this when ``model="unified"`` and
    keeps the public surface.
    """

    def __init__(
        self,
        *,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        api_key: str | None = None,
        segmentation_mode: SegmentationMode = "server_vad",
        language: str | None = None,
        interim_results: bool = True,
        partial_transcript_interval_s: float = 0.5,
        vad_threshold: float | None = None,
        vad_min_silence_duration_ms: int | None = None,
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
            segmentation_mode: Who ends a turn. ``server_vad`` (the default):
                the server's VAD, on trailing silence. ``client_flush``: only a
                flush does, which `STT` requests from LiveKit's own turn
                handling through ``manual_flush``.
            language: A BCP-47 code (``"en"``, ``"en-US"``) or ``"auto"``.
                ``None`` takes the deployment's default. Needs a server at
                v0.2.0 or newer; older ones close the session on it.
            interim_results: Emit revisable partials during a turn. Partials are
                replace-style: each re-states the whole current hypothesis.
            partial_transcript_interval_s: Cadence of those partials. The
                server raises it to its own floor and says so in a refusal.
            vad_threshold: Server-side VAD speech probability threshold.
                ``None`` takes the deployment's default. ``server_vad`` only.
            vad_min_silence_duration_ms: Silence needed to end a turn. ``None``
                takes the deployment's default. ``server_vad`` only.
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
            segmentation_mode=segmentation_mode,
            language=language,
            enable_partial_transcripts=interim_results,
            partial_transcript_interval_s=partial_transcript_interval_s,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
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
        partial_transcript_interval_s: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Applies to streams opened after this call; the session is per-socket."""
        if is_given(language):
            self._opts.language = language
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(partial_transcript_interval_s):
            self._opts.partial_transcript_interval_s = partial_transcript_interval_s

    def make_stream(
        self,
        owner: stt_module.STT,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> UnifiedSpeechStream:
        """Build a recognition stream owned by the public `STT`."""
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        return UnifiedSpeechStream(stt=owner, backend=self, opts=opts, conn_options=conn_options)

    async def recognize_via_stream(
        self,
        owner: stt_module.STT,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt_module.SpeechEvent:
        """One-shot recognition, run over the same streaming session."""
        # The buffer is queued once, and a stream's own reconnect would resume
        # after the audio its first socket consumed, so it could finish cleanly
        # on nothing. No retries here: STT.recognize() retries with a fresh
        # stream and the whole buffer instead.
        stream = self.make_stream(
            owner, language=language, conn_options=replace(conn_options, max_retry=0)
        )
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


class UnifiedSpeechStream(stt_module.SpeechStream):
    """A single recognition session: audio frames in, speech events out."""

    def __init__(
        self,
        *,
        stt: stt_module.STT,
        backend: _UnifiedBackend,
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
                    autoping=False,
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
            # LiveKit resamples to SAMPLE_RATE for us; this only re-chunks.
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
            # so this has to remember whether the last thing it did was flush —
            # otherwise the trailing flush below doubles it and the server
            # answers an extra, empty turn.
            flushed = False
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        await append(chunker.flush())
                        await ws.send_str(json.dumps({"type": "input_audio_buffer.flush"}))
                        flushed = True
                    else:
                        await append(chunker.write(data.data.tobytes()))
                        flushed = False
                if not flushed:
                    await append(chunker.flush())
                    await ws.send_str(json.dumps({"type": "input_audio_buffer.flush"}))
                # The server delivers every final it still owes, then closes
                # with 1000; that close is what ends `_recv`.
                await ws.send_str(json.dumps({"type": "close"}))
            finally:
                input_ended.set()

        async def _recv() -> None:
            speaking = False
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # The server closes with 1000 only in answer to `close`, after
                    # delivering every final it owed. Any other close, even once the
                    # input has ended, may have taken a final with it.
                    if input_ended.is_set() and ws.close_code == aiohttp.WSCloseCode.OK:
                        return
                    raise APIConnectionError(
                        f"Baseten closed the STT websocket (code {ws.close_code})"
                    )
                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"websocket error: {ws.exception()}")
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                kind = event.get("type")

                if kind == "error":
                    # Always followed by the server's close. at_capacity is the
                    # one a caller should retry elsewhere, so it keeps a 5xx.
                    code = event.get("code", "")
                    raise APIStatusError(
                        message=f"{code}: {event.get('message', 'unknown STT error')}",
                        status_code=_ERROR_STATUS.get(code, 500),
                        request_id=None,
                        body=None,
                    )

                if kind == "session.updated":
                    # `granted` is authoritative; each refusal says what was
                    # changed from the request and why. The session goes on.
                    for refusal in event.get("refusals") or []:
                        logger.warning(
                            "Baseten STT did not grant %s=%r (got %r): %s",
                            refusal.get("parameter"),
                            refusal.get("requested"),
                            refusal.get("granted"),
                            refusal.get("reason"),
                        )
                    continue

                if kind == "transcription.failed":
                    # One turn could not be decoded; the session goes on, so
                    # close the turn its partials opened and keep reading.
                    logger.warning(
                        "Baseten STT could not transcribe turn %s: %s",
                        event.get("turn_id"),
                        event.get("message"),
                    )
                    if speaking:
                        self._event_ch.send_nowait(
                            stt_module.SpeechEvent(type=stt_module.SpeechEventType.END_OF_SPEECH)
                        )
                        speaking = False
                    continue

                if kind not in ("transcription.delta", "transcription.done"):
                    continue

                transcript = event.get("transcript") or {}
                segments = transcript.get("segments") or []
                # Each segment carries its own leading separator, so joining
                # with "" is right for every script, spaced or not.
                text = "".join(s.get("text", "") for s in segments).strip()
                is_final = kind == "transcription.done"
                self._mark_provider_progress()
                # Leaving out partial_interval_s is how partials are turned off,
                # but a deployment may send them anyway; interim_results=False
                # is a promise to the caller, so it is kept here too.
                if not is_final and not self._opts.enable_partial_transcripts:
                    continue

                # Only real words open a turn. An empty final (a flush with no
                # speech behind it) would otherwise emit START/END_OF_SPEECH and,
                # under turn_detection="stt", commit the user's turn so the agent
                # answers silence.
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
                            alternatives=[self._speech_data(transcript, segments, text)],
                        )
                    )

                # A turn cut at the server's length limit is not the end of
                # speech: the speaker carries straight on into the next turn.
                if is_final and speaking and event.get("ended_by") != "max_turn_length":
                    self._event_ch.send_nowait(
                        stt_module.SpeechEvent(type=stt_module.SpeechEventType.END_OF_SPEECH)
                    )
                    speaking = False

        try:
            await ws.send_str(json.dumps(self._opts.session_update()))
            send_task = asyncio.create_task(_send(), name="unified-stt-send")
            recv_task = asyncio.create_task(_recv(), name="unified-stt-recv")
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
        self, transcript: dict[str, Any], segments: list[dict[str, Any]], text: str
    ) -> stt_module.SpeechData:
        return stt_module.SpeechData(
            # The language this turn was decoded in; absent when the server
            # cannot say (detection that does not report what it found).
            language=LanguageCode(transcript.get("language") or ""),
            text=text,
            # Server times are seconds from the start of the socket; the framework
            # grows start_time_offset on every reconnect so consumers see one timeline.
            start_time=(float(segments[0].get("start", 0.0)) if segments else 0.0)
            + self.start_time_offset,
            end_time=(float(segments[-1].get("end", 0.0)) if segments else 0.0)
            + self.start_time_offset,
            confidence=1.0,
        )
