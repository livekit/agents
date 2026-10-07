"""Opt-in native Oruk Realtime transcription with completed, keyed turn receipts."""

from __future__ import annotations

import asyncio
import copy
import os
import time
import uuid
import weakref
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, LanguageCode, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._realtime_protocol import (
    ENDPOINT,
    MODEL,
    TRANSCRIPT,
    RealtimeError,
    RealtimeOptions,
    TurnResult,
    new_http_session,
    run_turn,
    validate_endpoint,
)


class RealtimeSTT(stt.STT):
    """Native partial transcription; each flush commits one independent socket.

    Input is mono PCM16 at 16 kHz. The batch ``oruk.STT`` default is unchanged.
    A final event is released only after final text, usage and clean close;
    phrase events may arrive after final text. A caller must supply real turn
    boundaries: LiveKit's default STT node does not flush on VAD silence.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        endpoint: str = ENDPOINT,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        key = api_key if is_given(api_key) else os.environ.get("ORUK_API_KEY")
        if not key or any(c.isspace() for c in key):
            raise ValueError("Set a nonempty ORUK_API_KEY without whitespace")
        validate_endpoint(endpoint)
        self._key = key
        self._endpoint = endpoint
        self._session: aiohttp.ClientSession | None = None
        self._streams: weakref.WeakSet[RealtimeStream] = weakref.WeakSet()
        self._receipts: OrderedDict[str, tuple[float, TurnResult]] = OrderedDict()
        self._closed = False

    @property
    def model(self) -> str:
        return MODEL

    @property
    def provider(self) -> str:
        return "Oruk"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Use RealtimeSTT.stream(), or STT for file transcription")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RealtimeStream:
        if self._closed:
            raise RuntimeError("RealtimeSTT is closed")
        if len(self._streams) >= 4:
            raise RuntimeError("Close existing Realtime streams before creating more than four")
        if is_given(language) and language:
            raise ValueError("Oruk Realtime selects language automatically")
        stream = RealtimeStream(self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def take_turn(self, request_id: str) -> TurnResult | None:
        """Consume an exact receipt once; never guess by text, speaker or recency.

        At most four completed turns are cached per instance and are retrievable
        for 30 seconds. Expired data is purged lazily on read/write, or on close;
        this is not a timed-deletion guarantee. Missing, expired or evicted IDs
        return None. Use one instance per agent session. Normal LiveKit
        ChatMessages do not contain this request ID.
        """
        results = self.take_turns([request_id])
        return results[0] if results else None

    def take_turns(self, request_ids: Sequence[str]) -> list[TurnResult] | None:
        """Atomically consume an ordered group of up to four exact receipts.

        A user turn can contain several provider turns. Missing/expired/evicted
        IDs fail the whole lookup, without consuming the available subset.
        This method does not infer which IDs belong to a user message.
        """
        self._expire_receipts()
        if (
            isinstance(request_ids, str)
            or not 1 <= len(request_ids) <= 4
            or any(not isinstance(key, str) or not key for key in request_ids)
            or len(set(request_ids)) != len(request_ids)
            or any(key not in self._receipts for key in request_ids)
        ):
            return None
        results = [copy.deepcopy(self._receipts[key][1]) for key in request_ids]
        for key in request_ids:
            self._receipts.pop(key)
        return results

    def _expire_receipts(self) -> None:
        now = time.monotonic()
        while self._receipts:
            first = next(iter(self._receipts))
            if now - self._receipts[first][0] <= 30:
                break
            self._receipts.pop(first)

    def _record_turn(self, result: TurnResult) -> None:
        self._expire_receipts()
        self._receipts[result.request_id] = (time.monotonic(), copy.deepcopy(result))
        while len(self._receipts) > 4:
            self._receipts.popitem(last=False)

    async def aclose(self) -> None:
        self._closed = True
        try:
            results = await asyncio.gather(
                *(stream.aclose() for stream in list(self._streams)), return_exceptions=True
            )
        finally:
            self._receipts.clear()
            if self._session is not None:
                await self._session.close()
        for result in results:
            if isinstance(result, BaseException):
                raise result


class RealtimeStream(stt.RecognizeStream):
    """Bounded PCM input, one commit per segment, no framework audio replay."""

    def __init__(self, owner: RealtimeSTT, *, conn_options: APIConnectOptions) -> None:
        self._owner = owner
        self._connect_options = conn_options
        self._queued_bytes = 0
        self._boundary_times: deque[float] = deque()
        self._outstanding_events = 0
        # Only run_turn may retry an upgrade, and only before a send attempt.
        super().__init__(stt=owner, conn_options=replace(conn_options, max_retry=0))

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.sample_rate != 16000 or frame.num_channels != 1:
            raise ValueError("Realtime input must be mono PCM16 at 16 kHz")
        data = bytes(frame.data)
        if not data or len(data) > 10_240:
            raise ValueError("Realtime frames must contain 1 through 5120 mono samples")
        if self._queued_bytes + len(data) > 320_000:
            raise RuntimeError("Realtime input backlog exceeds ten audio seconds")
        self._check_input_not_ended()
        self._check_not_closed()
        self._queued_bytes += len(data)
        # Own the queued bytes; callers can reuse or mutate their frame buffers.
        super().push_frame(rtc.AudioFrame(data, 16000, 1, len(data) // 2))

    def flush(self) -> None:
        if len(self._boundary_times) >= 8:
            raise RuntimeError("Realtime input has too many pending turn boundaries")
        # The input boundary may be consumed much later, after another turn's
        # drain/upgrade. Capture the caller's boundary now, not at socket commit.
        boundary_time = time.time()
        super().flush()
        self._boundary_times.append(boundary_time)

    def _emit_event(self, event: stt.SpeechEvent) -> None:
        # The framework metrics tee drains into the user peer's deque. Only a
        # user __anext__ releases credit, so a paused user cannot grow that deque
        # across arbitrarily many completed turns.
        if self._outstanding_events >= 64:
            raise APIError(
                "Oruk Realtime output backlog; consume or close the stream", retryable=False
            )
        self._outstanding_events += 1
        self._event_ch.send_nowait(event)

    async def __anext__(self) -> stt.SpeechEvent:
        event = await super().__anext__()
        self._outstanding_events -= 1
        return event

    async def aclose(self) -> None:
        await super().aclose()
        self._owner._streams.discard(self)

    async def _run(self) -> None:
        async for first in self._input_ch:
            if isinstance(first, self._FlushSentinel):
                self._boundary_times.popleft()
                continue
            self._queued_bytes -= first.samples_per_channel * 2
            request_id = str(uuid.uuid4())
            provisional = ""
            interim_count = 0
            committed_at: float | None = None
            self._emit_event(stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH))

            async def audio(first_frame: rtc.AudioFrame = first) -> AsyncIterator[bytes]:
                nonlocal committed_at
                yield bytes(first_frame.data)
                async for frame in self._input_ch:
                    if isinstance(frame, self._FlushSentinel):
                        committed_at = self._boundary_times.popleft()
                        break
                    self._queued_bytes -= frame.samples_per_channel * 2
                    yield bytes(frame.data)

            def on_event(event: dict[str, Any]) -> None:
                nonlocal provisional, interim_count
                if event["type"] == TRANSCRIPT + "delta":
                    provisional += event["delta"]
                    # Bound framework queue growth even for one-character deltas.
                    # The complete final transcript is never truncated.
                    if interim_count >= 128:
                        return
                    interim_count += 1
                    self._emit_event(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=event["request_id"],
                            alternatives=[
                                stt.SpeechData(text=provisional, language=LanguageCode(""))
                            ],
                        )
                    )

            if self._owner._session is None:
                self._owner._session = new_http_session()
            try:
                result = await run_turn(
                    session=self._owner._session,
                    api_key=self._owner._key,
                    endpoint=self._owner._endpoint,
                    audio=audio(),
                    request_id=request_id,
                    options=RealtimeOptions(),
                    on_event=on_event,
                    connect_timeout=self._connect_options.timeout,
                    max_connect_retries=self._connect_options.max_retry,
                    retry_interval=self._connect_options.retry_interval,
                )
            except RealtimeError as exc:
                # The UUID supports reconciliation; no provider body/transcript is logged.
                raise APIError(
                    f"Oruk Realtime {exc.code}; request_id={request_id}; do not replay audio",
                    retryable=False,
                ) from None

            # FINAL, END and usage must fit together, before exposing a receipt.
            if self._outstanding_events + 3 > 64:
                raise APIError(
                    "Oruk Realtime output backlog; consume or close the stream", retryable=False
                )
            self._owner._record_turn(result)
            metadata = {
                "request_id": result.request_id,
                "model": MODEL,
                "phrases": copy.deepcopy(result.phrases),
                "usage": copy.deepcopy(result.usage),
                "asr_confidence": None,
                "detected_language": None,
                "input_boundary_time": committed_at,
                "transport_completion": "final_usage_clean_close",
                "phrase_coverage": "observed_events_only",
            }
            self._emit_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=result.request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=result.transcript,
                            language=LanguageCode(""),
                            metadata={"oruk": metadata},
                        )
                    ],
                )
            )
            self._emit_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=result.request_id,
                )
            )
            self._emit_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=result.request_id,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=result.usage["audio_seconds"]
                    ),
                )
            )
