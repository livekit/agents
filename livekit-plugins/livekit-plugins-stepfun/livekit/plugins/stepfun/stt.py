from __future__ import annotations

import asyncio
import json
import time
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_STT_SAMPLE_RATE,
    DEFAULT_VAD_SILENCE_DURATION_MS,
    DEFAULT_VAD_THRESHOLD,
    INCREMENTAL_DELTA_STT_MODELS,
    NUM_CHANNELS,
    PCM_BITS,
    PCM_CODEC,
    STT_FRAME_MS,
    STT_MAX_SESSION_DURATION,
    STT_RECOGNIZE_PATH,
    STT_STREAM_PATH,
    TIMESTAMP_ALIGN_STT_MODELS,
    Cluster,
    Language,
    STTBatchModels,
    STTModels,
)

_EV_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
_EV_TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
_EV_SPEECH_STARTED = "input_audio_buffer.speech_started"
_EV_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
_EV_SESSION_CREATED = "session.created"
_EV_SESSION_UPDATED = "session.updated"
_EV_COMMITTED = "input_audio_buffer.committed"
_EV_ITEM_CREATED = "conversation.item.created"
_EV_ERROR = "error"

#: How long to wait for the server to acknowledge our session.update before
#: giving up on the connection.
_SESSION_ACK_TIMEOUT = 10.0

#: Window after a commit acknowledgement for a transcription item to appear.
#: Silent commits produce no final; this timing heuristic is not a protocol guarantee.
_COMMIT_ITEM_WINDOW = 1.0

#: Quiet period after the last server event before a finalized stream ends.
_QUIET_SETTLE = 1.0

#: How long to wait, once input ends, before reporting that completion cannot be
#: confirmed because no final arrived. Bounded so a silent session shuts down
#: promptly; the protocol offers no receipt for audio that produced no speech.
_NO_EVIDENCE_DEADLINE = 2.0

if TYPE_CHECKING:
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    _WebSocket = aiohttp.ClientWebSocketResponse


@dataclass
class _STTOptions:
    model: STTModels | str
    batch_model: STTBatchModels | str
    language: NotGivenOr[Language | str]
    sample_rate: int
    #: Context or terminology hints for the streaming endpoint's `prompt` field.
    prompt: NotGivenOr[str]
    #: Hotword list for the batch endpoint.
    hotwords: list[str]
    enable_itn: bool
    enable_timestamp_align: bool
    server_vad: bool
    vad_silence_duration_ms: int
    vad_threshold: float
    cluster: Cluster
    ws_base_url: str
    http_base_url: str

    @property
    def deltas_are_cumulative(self) -> bool:
        """Whether a transcription delta replaces the previous text or extends it."""
        return self.model not in INCREMENTAL_DELTA_STT_MODELS

    @property
    def timestamps_available(self) -> bool:
        return self.enable_timestamp_align and self.model in TIMESTAMP_ALIGN_STT_MODELS


class STT(stt.STT):
    """StepFun streaming and batch speech recognition."""

    def __init__(
        self,
        *,
        model: STTModels | str = "stepaudio-2.5-asr-stream",
        batch_model: STTBatchModels | str = "stepaudio-3-asr-max",
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        http_base_url: str | None = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        hotwords: list[str] | None = None,
        enable_itn: bool = True,
        enable_timestamp_align: bool = False,
        server_vad: bool = True,
        vad_silence_duration_ms: int = DEFAULT_VAD_SILENCE_DURATION_MS,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        use_streaming: bool = True,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a streaming and batch speech recognizer.

        Args:
            model: WebSocket model ID.
            batch_model: HTTP + SSE model used by recognize() and batch sessions.
            language: Recognition language; the service defaults to English.
            api_key: API key, falling back to STEPFUN_API_KEY.
            cluster: "global" or "china" (mainland China); defaults to
                STEPFUN_CLUSTER, then global. Credentials are cluster-specific.
            base_url: Override the WebSocket base URL.
            http_base_url: Override the HTTP base URL.
            sample_rate: Streaming input rate; LiveKit resamples incoming audio.
                Batch recognize() also resamples VAD segments to this rate before
                upload, so a lower rate (16 kHz default) cuts the request size.
            prompt: Context hints for streaming recognition.
            hotwords: Batch recognition hints on the mainland-China cluster.
            enable_itn: Enable inverse text normalization.
            enable_timestamp_align: Request word alignment on supported streaming models.
            server_vad: Use server endpointing. Disable for manual flush() commits.
            vad_silence_duration_ms: Server VAD silence window.
            vad_threshold: Server VAD detection threshold.
            use_streaming: Set False for VAD-segmented batch recognition; AgentSession
                then requires a local vad to create a StreamAdapter.
            http_session: Caller-owned session; otherwise use the LiveKit job session.
        """
        resolved_cluster = _utils.resolve_cluster(cluster)
        aligned: Literal["word", False] = (
            "word"
            if use_streaming and enable_timestamp_align and model in TIMESTAMP_ALIGN_STT_MODELS
            else False
        )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                interim_results=use_streaming,
                offline_recognize=True,
                # Not supported on the streaming endpoint.
                diarization=False,
                aligned_transcript=aligned,
                keyterms=False,
                # Streaming ASR does not consume conversation context.
                chat_context=False,
            ),
        )

        if enable_timestamp_align and not aligned:
            if not use_streaming:
                logger.warning(
                    "enable_timestamp_align applies to the streaming endpoint "
                    "only; ignoring it because use_streaming=False"
                )
            else:
                logger.warning(
                    "enable_timestamp_align is only supported by %s; ignoring it",
                    ", ".join(sorted(TIMESTAMP_ALIGN_STT_MODELS)),
                    extra={"model": model},
                )

        self._use_streaming = use_streaming
        self._api_key = _utils.resolve_api_key(api_key)
        self._opts = _STTOptions(
            model=model,
            batch_model=batch_model,
            language=language,
            sample_rate=sample_rate,
            prompt=prompt,
            hotwords=hotwords or [],
            enable_itn=enable_itn,
            enable_timestamp_align=enable_timestamp_align,
            server_vad=server_vad,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_threshold=vad_threshold,
            cluster=resolved_cluster,
            ws_base_url=_utils.resolve_ws_base_url(base_url, resolved_cluster),
            http_base_url=_utils.resolve_http_base_url(http_base_url, resolved_cluster),
        )

        self._session = http_session
        self._streams = weakref.WeakSet[RecognizeStream]()
        self._pool = utils.ConnectionPool[_WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=STT_MAX_SESSION_DURATION,
        )

    @property
    def model(self) -> str:
        if not self._use_streaming:
            return str(self._opts.batch_model)
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "StepFun"

    # -- connections -------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            # Reuse the framework's session: it pools connections and is scoped
            # to the job lifecycle.
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> _WebSocket:
        url = f"{self._opts.ws_base_url}{STT_STREAM_PATH}"
        try:
            return await asyncio.wait_for(
                self._ensure_session().ws_connect(url, headers=_utils.auth_headers(self._api_key)),
                timeout,
            )
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun stt connect")

    async def _close_ws(self, ws: _WebSocket) -> None:
        await ws.close()

    def prewarm(self) -> None:
        """Complete DNS, TLS and the WebSocket handshake ahead of time, so cold
        start cost does not land on the first turn.
        """
        self._ensure_session()
        if self._use_streaming:
            self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        await self._pool.aclose()

        # Both caller-supplied and job-scoped HTTP sessions have external owners.
        self._session = None

    # -- configuration -----------------------------------------------------

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update options in place."""
        if is_given(model):
            self._opts.model = model
            self._capabilities.aligned_transcript = (
                "word" if self._use_streaming and self._opts.timestamps_available else False
            )
        if is_given(language):
            self._opts.language = language
        if is_given(prompt):
            self._opts.prompt = prompt

        # Pooled sockets were configured by a session.update built from the old
        # options, so stop handing them out.
        self._pool.invalidate()

    # -- streaming ---------------------------------------------------------

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        if not self._use_streaming:
            raise RuntimeError(
                "this STT was constructed with use_streaming=False, so it has no "
                "streaming transport. Either construct it with use_streaming=True, "
                "or pass it to an AgentSession with a vad= so the framework wraps "
                "it in a stt.StreamAdapter and uses the batch endpoint."
            )

        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        stream = RecognizeStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    # -- batch -------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """One-shot recognition over HTTP + SSE.

        Serves batch workloads, StreamAdapter setups and FallbackAdapter's
        offline escape hatch. The audio goes up in a single request and
        transcripts come back incrementally; only the final one matters here.
        """
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        frame = rtc.combine_audio_frames(buffer)

        if frame.sample_rate != self._opts.sample_rate:
            resampler = rtc.AudioResampler(
                frame.sample_rate,
                self._opts.sample_rate,
                num_channels=frame.num_channels,
            )
            resampled = list(resampler.push(frame))
            resampled.extend(resampler.flush())
            if resampled:
                frame = rtc.combine_audio_frames(resampled)

        transcription: dict[str, Any] = {
            "model": opts.batch_model,
            "enable_itn": opts.enable_itn,
        }
        if is_given(opts.language):
            transcription["language"] = opts.language
        if opts.hotwords and opts.cluster == "china":
            transcription["hotwords"] = opts.hotwords

        payload = {
            "audio": {
                "data": _utils.pcm_to_b64(frame.data.tobytes()),
                "input": {
                    "transcription": transcription,
                    "format": {
                        "type": "pcm",
                        "codec": PCM_CODEC,
                        "rate": frame.sample_rate,
                        "bits": PCM_BITS,
                        "channel": frame.num_channels,
                    },
                },
            }
        }

        url = f"{opts.http_base_url}{STT_RECOGNIZE_PATH}"
        headers = {
            **_utils.auth_headers(self._api_key),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        text = ""
        usage: dict[str, Any] = {}
        session_id = ""
        completed = False

        try:
            async with self._ensure_session().post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=conn_options.timeout,
                    sock_read=conn_options.timeout,
                ),
            ) as resp:
                await _utils.raise_for_status(resp)

                async for event in _utils.iter_sse_events(resp):
                    etype = event.get("type")
                    if etype == "transcript.text.delta":
                        # Incremental on this route, unlike the WebSocket's
                        # cumulative deltas.
                        text += event.get("delta") or ""
                    elif etype == "transcript.text.done":
                        completed = True
                        text = event.get("text", text) or ""
                        usage = event.get("usage") or {}
                        session_id = (event.get("meta") or {}).get("session_id", "")
                        break
                    elif etype == "error":
                        raise _utils.status_error(
                            status_code=-1,
                            message="stepfun stt recognize: server returned an error",
                            request_id=(event.get("meta") or {}).get("session_id"),
                            body=event,
                        )
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun stt recognize")

        if not completed:
            raise APIConnectionError("stepfun stt: SSE stream ended before transcript.text.done")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=session_id or utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    # This route echoes no detected language, so report what was
                    # asked for; English is the server-side default.
                    language=LanguageCode(opts.language if is_given(opts.language) else "en"),
                    text=text,
                )
            ],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=utils.audio.calculate_audio_duration(buffer),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )


def _close_detail(ws: _WebSocket, msg: Any) -> str:
    """Describe why a socket closed, as far as the transport can tell.

    A bare "closed unexpectedly" sends people to check their network, which is
    the wrong place when the real cause is a configuration the server refused —
    an unsupported model, for instance, which this endpoint signals by dropping
    the connection rather than returning an error frame.
    """
    code = ws.close_code
    if code is None and msg.type is aiohttp.WSMsgType.CLOSE:
        code = msg.data
    parts = []
    if code is not None:
        parts.append(f"code={code}")
    reason = msg.extra or None
    if reason:
        parts.append(f"reason={reason!r}")
    return f" ({', '.join(parts)})" if parts else ""


@dataclass(eq=False)
class _PendingCommit:
    """A manual commit awaiting acknowledgement and possible transcription."""

    acknowledged_at: float | None = None


class RecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._stt_impl = stt
        self._opts = opts  # already replace()d in STT.stream(); this is a snapshot
        self._request_id = ""
        self._interim_texts: dict[str, str] = {}
        self._input_done = False
        # Track transcription items, not the commit bookkeeping items.
        self._open_items: set[str] = set()
        self._anonymous_open = False  # speech reported without an item_id
        self._speech_active = False
        # Commit and transcription IDs differ, so each acknowledgement keeps
        # its own item-arrival window.
        self._commits: list[_PendingCommit] = []
        self._expected_finals = 0
        self._finals_received = 0
        self._consumed_audio = False
        self._input_finished_at = 0.0
        self._last_event_at = 0.0
        self._progress = asyncio.Event()
        self._attempt_audio = 0.0

        self._audio_duration = 0.0
        self._pending_tokens = (0, 0)

    # -- main loop ---------------------------------------------------------

    async def _run(self) -> None:
        try:
            await self._run_impl()
        except Exception as exc:
            translated = _utils.translate_exception(exc, context="stepfun stt stream")
            if isinstance(translated, APIError) and translated.retryable and self._speech_at_risk():
                # A fresh session cannot recover consumed audio, so do not
                # silently retry once any was sent.
                raise APIConnectionError(
                    "stepfun stt: connection lost before final transcription; "
                    "in-flight speech cannot be recovered by a retry",
                    retryable=False,
                ) from translated
            if translated is exc:
                raise
            raise translated from exc

    async def _run_impl(self) -> None:
        self._request_id = utils.shortuuid()
        self._interim_texts.clear()
        self._input_done = False
        self._open_items.clear()
        self._anonymous_open = False
        self._speech_active = False
        self._commits.clear()
        self._expected_finals = 0
        self._finals_received = 0
        self._consumed_audio = False
        self._input_finished_at = 0.0
        self._last_event_at = 0.0
        self._uncommitted_audio = False
        self._attempt_audio = 0.0

        async with self._stt_impl._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._report_connection_acquired(
                self._stt_impl._pool.last_acquire_time,
                self._stt_impl._pool.last_connection_reused,
            )

            try:
                await self._configure_session(ws)
            except BaseException:
                await ws.close()
                self._stt_impl._pool.remove(ws)
                raise

            tasks = [
                asyncio.create_task(self._send_task(ws), name="stepfun_stt_send"),
                asyncio.create_task(self._recv_task(ws), name="stepfun_stt_recv"),
            ]
            keepalive = asyncio.create_task(self._keepalive_task(ws), name="stepfun_stt_keepalive")
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                send_task, recv_task = tasks
                if send_task.done() and not recv_task.done():
                    # Surface a failed upload immediately instead of waiting for a final.
                    send_task.result()
                    await self._await_finalization(recv_task)
                    if not recv_task.done():
                        # Finished without a closing event: settle usage here,
                        # since the receiver is about to be cancelled.
                        self._report_usage()
            finally:
                await utils.aio.cancel_and_wait(*tasks, keepalive)

                # Discarding a pooled connection does not close it immediately.
                await ws.close()
                self._stt_impl._pool.remove(ws)
                self._stt_impl._pool.prewarm()

            for task in tasks:
                if not task.cancelled():
                    task.result()

    async def _configure_session(self, ws: _WebSocket) -> None:
        """Send session.update and wait for the server to acknowledge it.

        This endpoint accepts no query parameters: the model, audio format and
        turn detection are configured entirely by this frame. Skipping it leaves
        the session unconfigured, which is why pumping audio straight after
        connecting cannot work.
        """
        transcription: dict[str, Any] = {
            "model": self._opts.model,
            "enable_itn": self._opts.enable_itn,
        }
        if is_given(self._opts.language):
            transcription["language"] = self._opts.language
        if is_given(self._opts.prompt):
            transcription["prompt"] = self._opts.prompt
        if self._opts.timestamps_available:
            transcription["enable_timestamp_align"] = True

        audio_input: dict[str, Any] = {
            "format": {
                "type": "pcm",
                "codec": PCM_CODEC,
                "rate": self._opts.sample_rate,
                "bits": PCM_BITS,
                "channel": NUM_CHANNELS,
            },
            "transcription": transcription,
        }
        if self._opts.server_vad:
            audio_input["turn_detection"] = {
                "type": "server_vad",
                "silence_duration_ms": self._opts.vad_silence_duration_ms,
                "threshold": self._opts.vad_threshold,
            }

        await ws.send_str(
            json.dumps(
                {
                    "event_id": _utils.event_id(),
                    "type": "session.update",
                    "session": {"audio": {"input": audio_input}},
                }
            )
        )

        # Wait for the ack before sending audio, so a rejected configuration
        # surfaces as a configuration error rather than as silence.
        try:
            await asyncio.wait_for(self._await_session_ack(ws), _SESSION_ACK_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise APIConnectionError(
                "stepfun stt: timed out waiting for session.update acknowledgement"
            ) from exc

    async def _await_session_ack(self, ws: _WebSocket) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIConnectionError(
                    "stepfun stt: the server closed the connection while "
                    "configuring the session"
                    f"{_close_detail(ws, msg)}. This usually means the session "
                    "configuration was refused — most often an unsupported "
                    f"model (model={self._opts.model!r}) or audio format — "
                    "rather than a network problem."
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"stepfun stt: websocket error: {msg.data}")

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = json.loads(msg.data)
            etype = event.get("type")
            if etype == _EV_SESSION_UPDATED:
                logger.debug(
                    "stepfun stt: session ready",
                    extra={
                        "type": etype,
                        "session_id": (event.get("meta") or {}).get("session_id"),
                    },
                )
                return
            if etype == _EV_ERROR:
                raise _utils.translate_asr_ws_error(event, context="stepfun stt session.update")
            # session.created can arrive unprompted before our ack; anything
            # else at this stage is noise.
            logger.debug("stepfun stt: ignoring event during setup", extra={"type": etype})

    # -- upstream ----------------------------------------------------------

    async def _send_task(self, ws: _WebSocket) -> None:
        self._uncommitted_audio = False
        samples_per_frame = self._opts.sample_rate * STT_FRAME_MS // 1000
        bstream = utils.audio.AudioByteStream(
            self._opts.sample_rate,
            NUM_CHANNELS,
            samples_per_channel=samples_per_frame,
        )

        async for data in self._input_ch:
            await self._send_item(ws, bstream, data)

        await self._finish_input(ws, samples_per_frame)

    async def _send_item(
        self,
        ws: _WebSocket,
        bstream: utils.audio.AudioByteStream,
        data: rtc.AudioFrame | stt.RecognizeStream._FlushSentinel,
    ) -> None:
        frames: list[rtc.AudioFrame] = []
        flushing = False

        if isinstance(data, rtc.AudioFrame):
            # Even a sub-frame tail has left the input channel and cannot be
            # recovered by the base class on retry.
            self._consumed_audio = self._consumed_audio or data.samples_per_channel > 0
            frames.extend(bstream.write(data.data.tobytes()))
        elif isinstance(data, self._FlushSentinel):
            # Drain the sub-frame tail held by AudioByteStream, or that
            # audio is lost.
            frames.extend(bstream.flush())
            flushing = True

        for frame in frames:
            self._uncommitted_audio = True
            duration = frame.samples_per_channel / frame.sample_rate
            self._audio_duration += duration
            self._attempt_audio += duration
            await self._append(ws, frame.data.tobytes())

        if flushing:
            await self._maybe_commit(ws)

    async def _append(self, ws: _WebSocket, pcm: bytes) -> None:
        await ws.send_str(
            json.dumps(
                {
                    "event_id": _utils.event_id(),
                    "type": "input_audio_buffer.append",
                    "audio": _utils.pcm_to_b64(pcm),
                }
            )
        )

    async def _finish_input(self, ws: _WebSocket, samples_per_frame: int) -> None:
        if self._opts.server_vad and self._attempt_audio > 0:
            # Append silence so endpointing can close the last real-time turn.
            silence_samples = (
                self._opts.sample_rate * (self._opts.vad_silence_duration_ms + STT_FRAME_MS) // 1000
            )
            silence = b"\x00\x00" * silence_samples
            step = samples_per_frame * 2
            for offset in range(0, len(silence), step):
                chunk = silence[offset : offset + step]
                await self._append(ws, chunk)
            # Best-effort finalize: the public gateway ignores commits under
            # server VAD today, and promises no acknowledgement, so it is not
            # tracked. The silence above is what closes the turn meanwhile.
            await self._send_commit(ws)
            self._uncommitted_audio = False

        await self._maybe_commit(ws)
        self._input_finished_at = time.monotonic()
        self._last_event_at = self._input_finished_at
        self._input_done = True
        self._progress.set()

    def _final_pending(self) -> bool:
        """Whether the server may still owe a final for audio already sent."""
        return bool(
            self._speech_active
            or self._open_items
            or self._anonymous_open
            or self._interim_texts
            or any(commit.acknowledged_at is None for commit in self._commits)
        )

    def _end_input_grace_deadline(self) -> float:
        """Wait for a commit's transcription item, unless it already arrived.

        A commit yields at most one final, so once as many finals as
        acknowledgements have arrived nothing is outstanding. Unmatched
        commitments can only be silent audio, which waits out one window.
        """
        if self._expected_finals <= 0:
            return self._input_finished_at
        return max(
            (
                commit.acknowledged_at + _COMMIT_ITEM_WINDOW
                for commit in self._commits
                if commit.acknowledged_at is not None
            ),
            default=self._input_finished_at,
        )

    def _unconfirmed(self) -> bool:
        """Whether ending now could lose a transcript the server still owes."""
        if self._final_pending():
            return True
        if self._opts.server_vad:
            # A final is the only evidence that sent audio was processed.
            return self._consumed_audio and not self._finals_received
        return self._expected_finals > 0 and time.monotonic() < self._end_input_grace_deadline()

    def _speech_at_risk(self) -> bool:
        """A fresh session cannot recover any audio consumed by this attempt.

        Speech detection is asynchronous; absent speech events do not establish
        that consumed audio was silent or safely finalized.
        """
        return self._consumed_audio or self._final_pending()

    async def _await_finalization(self, recv_task: asyncio.Task[None]) -> None:
        """Finish only on evidence that the sent audio was processed.

        A delivered final is that evidence: with it, a short quiet period is
        enough. Without it, silence is indistinguishable from an unreported
        turn, so the stream waits a bounded interval and then reports that
        completion cannot be confirmed. A manual commit whose item window
        elapsed inside the deadline completes without a transcript.
        """
        timeout_at = self._input_finished_at + self._conn_options.timeout
        if not self._consumed_audio and not self._final_pending():
            return
        while not recv_task.done():
            self._progress.clear()
            now = time.monotonic()
            if self._final_pending():
                # A slow turn is still alive as long as events keep arriving,
                # so the limit is idle time rather than time since input ended.
                idle_at = self._last_event_at + self._conn_options.timeout
                if now >= idle_at:
                    raise APIConnectionError(
                        "stepfun stt: timed out waiting for final transcription"
                    )
                wake_at = idle_at
            elif self._opts.server_vad:
                if self._finals_received:
                    wake_at = min(self._last_event_at + _QUIET_SETTLE, timeout_at)
                    if now >= wake_at:
                        return
                else:
                    unconfirmed_at = min(
                        timeout_at, self._input_finished_at + _NO_EVIDENCE_DEADLINE
                    )
                    if now >= unconfirmed_at:
                        raise APIConnectionError(
                            "stepfun stt: cannot confirm the sent audio was transcribed"
                        )
                    wake_at = unconfirmed_at
            else:
                grace_at = self._end_input_grace_deadline()
                if now >= timeout_at and grace_at > timeout_at:
                    raise APIConnectionError(
                        "stepfun stt: cannot confirm the sent audio was transcribed"
                    )
                if now >= grace_at:
                    return
                wake_at = min(grace_at, timeout_at)

            progress = asyncio.create_task(self._progress.wait())
            try:
                await asyncio.wait(
                    [recv_task, progress],
                    timeout=max(0.0, wake_at - now),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                await utils.aio.cancel_and_wait(progress)

    async def _send_commit(self, ws: _WebSocket) -> None:
        """Ask the server to finalize the current audio buffer."""
        await ws.send_str(
            json.dumps({"event_id": _utils.event_id(), "type": "input_audio_buffer.commit"})
        )

    async def _maybe_commit(self, ws: _WebSocket) -> None:
        """Tell the server the current utterance is finished."""
        if self._opts.server_vad or not self._uncommitted_audio:
            return

        now = time.monotonic()
        self._commits = [
            pending
            for pending in self._commits
            if pending.acknowledged_at is None
            or now < pending.acknowledged_at + _COMMIT_ITEM_WINDOW
        ]
        commit = _PendingCommit()
        # Register before yielding: the receiver may process the ack while
        # send_str is suspended. Roll back only if still unacked.
        self._commits.append(commit)
        try:
            await self._send_commit(ws)
        except BaseException:
            if commit.acknowledged_at is None:
                self._commits.remove(commit)
            raise
        self._uncommitted_audio = False

    # -- downstream --------------------------------------------------------

    async def _recv_task(self, ws: _WebSocket) -> None:
        while True:
            msg = await ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # A close ends the session, so nothing further can arrive: the
                # stream is complete once nothing it owes is outstanding.
                if self._input_done and not self._unconfirmed():
                    self._report_usage()
                    return
                if self._input_done:
                    raise APIConnectionError(
                        "stepfun stt: websocket closed before final transcription"
                    )
                raise APIConnectionError(
                    "stepfun stt: the server closed the connection mid-stream"
                    f"{_close_detail(ws, msg)}. If this happens immediately "
                    "after the first audio, the server is most likely "
                    f"rejecting the model (model={self._opts.model!r}) — the "
                    "bidirectional endpoint documents support for "
                    "stepaudio-2.5-asr-stream only."
                )

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"stepfun stt: websocket error: {msg.data}")

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.debug("stepfun stt: unexpected ws frame type %s", msg.type)
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("stepfun stt: failed to decode ws frame")
                continue

            self._handle_event(event)
            self._last_event_at = time.monotonic()
            self._progress.set()

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")

        if etype == _EV_ERROR:
            raise _utils.translate_asr_ws_error(event, context="stepfun stt")

        if etype == _EV_TRANSCRIPT_DELTA:
            self._handle_delta(event)
            return

        if etype == _EV_TRANSCRIPT_COMPLETED:
            self._handle_completed(event)
            return

        if etype == _EV_SPEECH_STARTED:
            self._speech_active = True
            self._open_item(event.get("item_id"))
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        if etype == _EV_SPEECH_STOPPED:
            self._speech_active = False
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        if etype == _EV_COMMITTED:
            for commit in self._commits:
                if commit.acknowledged_at is None:
                    commit.acknowledged_at = time.monotonic()
                    self._expected_finals += 1
                    break
            return

        if etype == _EV_ITEM_CREATED:
            item = event.get("item") or {}
            # Only an in_progress item is transcribed and finalized; the
            # completed-status one is the committed buffer's bookkeeping.
            if item.get("status") == "in_progress":
                self._open_item(item.get("id") or event.get("item_id"))
            return

        if etype in (_EV_SESSION_CREATED, _EV_SESSION_UPDATED):
            logger.debug("stepfun stt: %s", etype)
            return

        logger.debug("stepfun stt: unhandled event type %s", etype)

    def _open_item(self, item_id: str | None) -> None:
        """Record an utterance the server reported, identified or not.

        An unidentified report is folded into the next identified one, so a
        `speech_started` without an id cannot outlive the final for its item.
        """
        if item_id:
            self._open_items.add(item_id)
            self._anonymous_open = False
        else:
            self._anonymous_open = True

    def _handle_delta(self, event: dict[str, Any]) -> None:
        text = event.get("text") or ""
        item_id = event.get("item_id") or ""

        if self._opts.deltas_are_cumulative:
            # Replaces, never appends — the payload is the full transcript so
            # far, including revisions to text already emitted.
            self._interim_texts[item_id] = text
        else:
            self._interim_texts[item_id] = self._interim_texts.get(item_id, "") + text

        interim_text = self._interim_texts[item_id]
        if not interim_text:
            return

        stash = event.get("stash") or ""
        if stash and interim_text.endswith(stash):
            stable = interim_text[: -len(stash)]
            if stable:
                self._emit(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, stable, event, words=None)

        self._emit(
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            interim_text,
            event,
            words=self._timed_words(event),
        )

    def _handle_completed(self, event: dict[str, Any]) -> None:
        # The completed frame carries the final text under `transcript`, not
        # `text`.
        item_id = event.get("item_id") or ""
        self._open_items.discard(item_id)
        self._expected_finals = max(0, self._expected_finals - 1)
        self._finals_received += 1
        # A final closes any unidentified report too: the server finalizes
        # utterances in order, and an id-less start belongs to this one.
        self._anonymous_open = False
        # Clear both keys: an anonymous interim belongs to this utterance, as
        # assumed for _anonymous_open.
        identified = self._interim_texts.pop(item_id, None)
        anonymous = self._interim_texts.pop("", None)
        interim_text = identified if identified is not None else (anonymous or "")
        text = event.get("transcript", interim_text) or ""

        usage = event.get("usage") or {}
        self._pending_tokens = (
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        self._emit(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            text,
            event,
            words=self._timed_words(event),
        )
        # Settle usage right after each final so metrics align with turns.
        self._report_usage()

    def _timed_words(self, event: dict[str, Any]) -> list[TimedString] | None:
        """Translate the per-word `words` array, when timestamp alignment is on."""
        raw = event.get("words")
        if not raw or not self._opts.timestamps_available:
            return None

        words = [
            TimedString(
                entry.get("word", ""),
                start_time=entry.get("start", NOT_GIVEN),
                end_time=entry.get("end", NOT_GIVEN),
                start_time_offset=self.start_time_offset,
            )
            for entry in raw
            if entry.get("final", True)
        ]
        return words or None

    def _emit(
        self,
        event_type: stt.SpeechEventType,
        text: str,
        raw: dict[str, Any],
        *,
        words: list[TimedString] | None,
    ) -> None:
        if not text:
            return

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        language=LanguageCode(
                            raw.get("language")
                            or (self._opts.language if is_given(self._opts.language) else "en")
                        ),
                        text=text,
                        words=words,
                    )
                ],
            )
        )

    def _report_usage(self) -> None:
        """Emit accumulated usage, then reset.

        Audio duration is accumulated client-side; the token counts come from
        the server's own `usage` block on the completed event, so they match
        what is billed. Emitting nothing when there is nothing to report keeps
        the close-time settle from producing an empty metrics record.
        """
        input_tokens, output_tokens = self._pending_tokens
        if not self._audio_duration and not input_tokens and not output_tokens:
            return

        audio_duration = self._audio_duration
        self._audio_duration = 0.0
        self._pending_tokens = (0, 0)

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=audio_duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
        )

    # -- keepalive ---------------------------------------------------------

    async def _keepalive_task(self, ws: _WebSocket) -> None:
        """Keep the socket alive through long silences."""
        while True:
            await asyncio.sleep(10.0)
            await ws.ping()


# Deprecated alias. The base class renamed SpeechStream to RecognizeStream and
# kept its own alias; this mirrors that so existing imports keep working.
SpeechStream = RecognizeStream
