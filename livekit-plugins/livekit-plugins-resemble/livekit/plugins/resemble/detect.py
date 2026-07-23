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

from __future__ import annotations

import asyncio
import io
import math
import os
import struct
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, utils

from .log import logger

RESEMBLE_DETECT_API_URL = "https://app.resemble.ai/api/v2"

DETECT_SAMPLE_RATE = 16000
"""Sample rate (Hz) the monitor resamples participant audio to before analysis."""

DetectionMode = Literal["sampled", "first_n", "continuous"]
DetectionLabel = Literal["real", "fake", "inconclusive"]
DetectionNormalizedLabel = Literal["real", "synthetic", "inconclusive"]
DetectionSecurity = Literal["spot", "standard", "high"]
DetectionAction = Literal["continue", "watch", "verify", "block"]
DetectFormFieldValue = str | int | float | bool

EventTypes = Literal["result", "fake_detected", "synthetic_detected", "verdict", "error"]


class DetectTransport(Protocol):
    """Transport used by :class:`DetectionMonitor` to submit one audio window.

    The default implementation calls Resemble Detect's REST endpoint. Advanced users can
    pass their own transport to use a streaming Detect backend, a gateway, or a test double
    without changing the LiveKit-facing monitor logic.
    """

    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Submit mono 16 kHz PCM16 audio and return a completed Detect ``item`` payload."""


@dataclass(frozen=True)
class _SecurityPreset:
    mode: DetectionMode
    samples: int
    sample_interval_seconds: float
    analysis_budget_seconds: float
    agreement_window: int
    min_fake_results: int


_SECURITY_PRESETS: dict[DetectionSecurity, _SecurityPreset] = {
    "spot": _SecurityPreset(
        mode="sampled",
        samples=1,
        sample_interval_seconds=0.0,
        analysis_budget_seconds=4.0,
        agreement_window=1,
        min_fake_results=1,
    ),
    "standard": _SecurityPreset(
        mode="sampled",
        samples=3,
        sample_interval_seconds=30.0,
        analysis_budget_seconds=16.0,
        agreement_window=3,
        min_fake_results=2,
    ),
    "high": _SecurityPreset(
        mode="continuous",
        samples=3,
        sample_interval_seconds=0.0,
        analysis_budget_seconds=0.0,
        agreement_window=3,
        min_fake_results=2,
    ),
}

_DEFAULT_WINDOW_SECONDS = 4.0
_DEFAULT_FAKE_THRESHOLD = 0.7
_DEFAULT_LIKELY_SYNTHETIC_THRESHOLD = 0.5
_DEFAULT_UNCLEAR_THRESHOLD = 0.3


@dataclass
class DetectionResult:
    """Outcome of analyzing one audio window with Resemble Detect."""

    label: DetectionLabel
    """``"fake"`` or ``"real"`` as returned by the API."""
    aggregated_score: float
    """Aggregated fake-probability for the window (0.0 = real, 1.0 = fake)."""
    scores: list[float]
    """Per-frame fake-probabilities inside the window (one per ``frame_length`` seconds)."""
    consistency: float | None
    """Model consistency metric for the window, when provided by the API."""
    window_index: int
    """0-based index of the analyzed window within this monitor's stream."""
    window_start: float
    """Start of the window, in seconds since monitoring began."""
    window_end: float
    """End of the window, in seconds since monitoring began."""
    participant_identity: str | None
    """Identity of the monitored participant, when known."""
    detection_uuid: str | None
    """UUID of the detection job on Resemble's side (for follow-up queries)."""
    latency: float
    """Round-trip seconds spent on the API call for this window."""
    forced: bool = False
    """True if this check was triggered on demand via :meth:`DetectionMonitor.check_now`
    rather than by the ambient schedule."""
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    """Raw ``item`` payload returned by the API."""

    @property
    def score(self) -> float:
        """Alias for ``aggregated_score``; a fake/synthetic probability in ``[0, 1]``."""
        return self.aggregated_score

    @property
    def normalized_label(self) -> DetectionNormalizedLabel:
        """Developer-facing label: ``"synthetic"`` instead of Detect's raw ``"fake"``."""
        return _normalize_label(self.label)

    @property
    def confidence(self) -> float:
        """Best-effort confidence in ``[0, 1]``.

        Resemble Detect can return ``consistency`` as either a percentage or a fraction. If it
        is absent, fall back to distance from the decision boundary.
        """
        if self.consistency is not None:
            return _clamp01(self.consistency / 100 if self.consistency > 1 else self.consistency)
        return _clamp01(abs(self.aggregated_score - 0.5) * 2)

    @property
    def scan_index(self) -> int:
        """1-based scan index for UI/event payloads."""
        return self.window_index + 1

    @property
    def window_ts(self) -> float:
        """End timestamp of the analyzed window, seconds since monitoring began."""
        return self.window_end

    @property
    def is_final(self) -> bool:
        """Per-window results are interim; finality is represented by ``DetectionVerdict``."""
        return False

    @property
    def recommended_action(self) -> DetectionAction:
        """Action band using the default Resemble Detect score strategy."""
        if self.aggregated_score < _DEFAULT_UNCLEAR_THRESHOLD:
            return "continue"
        if self.aggregated_score < _DEFAULT_LIKELY_SYNTHETIC_THRESHOLD:
            return "watch"
        if self.aggregated_score < _DEFAULT_FAKE_THRESHOLD:
            return "verify"
        return "block"

    def to_dict(self) -> dict[str, Any]:
        """Return a small, stable payload suitable for app events or dashboards."""
        return {
            "label": self.normalized_label,
            "raw_label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "window_ts": self.window_ts,
            "scan_index": self.scan_index,
            "is_final": self.is_final,
            "recommended_action": self.recommended_action,
            "participant_identity": self.participant_identity,
            "detection_uuid": self.detection_uuid,
            "latency": self.latency,
            "forced": self.forced,
        }


@dataclass
class DetectionVerdict:
    """Aggregate verdict across all analyzed windows of a monitoring session."""

    label: DetectionLabel
    """``"fake"`` if the security policy confirmed a synthetic caller, ``"real"`` if speech
    was analyzed and stayed below the unclear band, ``"inconclusive"`` otherwise."""
    max_score: float | None
    """Highest window ``aggregated_score`` observed."""
    analyzed_seconds: float
    """Total seconds of speech submitted for analysis."""
    results: list[DetectionResult]
    """All per-window results that informed the verdict."""

    @property
    def score(self) -> float:
        """Highest fake/synthetic probability observed, or ``0.0`` if no speech was analyzed."""
        return self.max_score or 0.0

    @property
    def normalized_label(self) -> DetectionNormalizedLabel:
        """Developer-facing label: ``"synthetic"`` instead of Detect's raw ``"fake"``."""
        return _normalize_label(self.label)

    @property
    def confidence(self) -> float:
        """Confidence from the highest-scoring result, or ``0.0`` if no result exists."""
        if not self.results:
            return 0.0
        result = max(self.results, key=lambda r: r.aggregated_score)
        return result.confidence

    @property
    def scan_index(self) -> int:
        """1-based index of the latest scan represented by this verdict."""
        return len(self.results)

    @property
    def window_ts(self) -> float:
        """End timestamp of the latest analyzed window."""
        if not self.results:
            return 0.0
        return self.results[-1].window_end

    @property
    def is_final(self) -> bool:
        """Verdicts are final for the completed ambient detection budget."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a small, stable payload suitable for app events or dashboards."""
        return {
            "label": self.normalized_label,
            "raw_label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "window_ts": self.window_ts,
            "scan_index": self.scan_index,
            "is_final": self.is_final,
            "analyzed_seconds": self.analyzed_seconds,
        }


@dataclass
class _DetectOptions:
    security: DetectionSecurity
    window_seconds: float
    mode: DetectionMode
    analysis_budget_seconds: float
    samples: int
    sample_interval_seconds: float
    fake_threshold: float
    likely_synthetic_threshold: float
    unclear_threshold: float
    agreement_window: int
    min_fake_results: int
    force_immediate_fake: bool
    frame_length: int
    silence_rms_threshold: float
    request_timeout: float


class DetectionMonitor(rtc.EventEmitter[EventTypes]):
    """Real-time deepfake detection for LiveKit participants, powered by Resemble Detect.

    The monitor taps a participant's audio in parallel with the agent pipeline (it never
    modifies or delays frames), buffers it into short windows, and submits each window to
    the Resemble Detect API (https://docs.resemble.ai/detect). Results are surfaced as
    events so an agent can warn the user, flag the session, or end the call:

    - ``"result"`` (:class:`DetectionResult`): a window was analyzed
    - ``"fake_detected"`` (:class:`DetectionResult`): a raw window crossed ``fake_threshold``
      (kept for backwards compatibility)
    - ``"synthetic_detected"`` (:class:`DetectionResult`): the configured security policy
      has enough agreement to treat the caller as synthetic
    - ``"verdict"`` (:class:`DetectionVerdict`): the analysis budget completed (``first_n``
      mode) or the stream ended
    - ``"error"`` (Exception): an API call failed (monitoring continues)

    By default, ``security="standard"`` checks about four seconds of real speech early,
    repeats spot-checks across the call, and only emits ``"synthetic_detected"`` after
    two suspicious windows in the last three checks. Set ``security="spot"`` for a single
    low-cost check or ``security="high"`` for continuous monitoring.
    """

    def __init__(
        self,
        *,
        security: DetectionSecurity = "standard",
        api_key: str | None = None,
        base_url: str = RESEMBLE_DETECT_API_URL,
        window_seconds: float | None = None,
        mode: DetectionMode | None = None,
        samples: int | None = None,
        sample_interval_seconds: float | None = None,
        analysis_budget_seconds: float | None = None,
        fake_threshold: float = _DEFAULT_FAKE_THRESHOLD,
        likely_synthetic_threshold: float = _DEFAULT_LIKELY_SYNTHETIC_THRESHOLD,
        unclear_threshold: float = _DEFAULT_UNCLEAR_THRESHOLD,
        agreement_window: int | None = None,
        min_fake_results: int | None = None,
        force_immediate_fake: bool = False,
        frame_length: int = 2,
        silence_rms_threshold: float = 0.0035,
        request_timeout: float = 30.0,
        http_session: aiohttp.ClientSession | None = None,
        transport: DetectTransport | None = None,
        zero_retention_mode: bool = True,
        extra_form_fields: Mapping[str, DetectFormFieldValue] | None = None,
    ) -> None:
        """Create a new deepfake-detection monitor.

        Args:
            security (DetectionSecurity, optional): Preset detection strategy. ``"spot"``
                performs one low-cost check. ``"standard"`` (default) samples early and
                across the call with 2-of-3 agreement. ``"high"`` monitors continuously.
            api_key (str, optional): Resemble API key. If not provided, it will be read
                from the RESEMBLE_API_KEY environment variable. Not required if
                ``transport`` is provided.
            base_url (str, optional): Override the Resemble REST Detect API base URL.
            window_seconds (float, optional): Seconds of audio per detection request.
                Defaults to 4.0 (minimum 2.0). Overrides the selected preset.
            mode (DetectionMode, optional): ``"sampled"`` takes ``samples``
                spot-checks of ``window_seconds`` each, spaced ``sample_interval_seconds``
                apart, then emits a ``"verdict"`` and stops — low, predictable cost per call.
                ``"first_n"`` analyzes the first ``analysis_budget_seconds`` of speech back
                to back, then stops. ``"continuous"`` analyzes for the lifetime of the track.
                Overrides the selected preset.
            samples (int, optional): Number of spot-checks in ``"sampled"`` mode.
                Overrides the selected preset.
            sample_interval_seconds (float, optional): Idle gap between spot-checks in
                ``"sampled"`` mode (audio in the gap is not analyzed). Overrides the preset.
            analysis_budget_seconds (float, optional): Speech budget for ``"first_n"`` mode.
                Overrides the selected preset.
            fake_threshold (float, optional): ``aggregated_score`` at or above which a
                window emits ``"fake_detected"``. Defaults to 0.7.
            likely_synthetic_threshold (float, optional): Score at or above which a final
                verdict is inconclusive unless agreement confirms a synthetic caller.
            unclear_threshold (float, optional): Score at or above which a final verdict is
                inconclusive instead of real.
            agreement_window (int, optional): Number of recent checks considered for
                agreement. Overrides the selected preset.
            min_fake_results (int, optional): Suspicious results needed within
                ``agreement_window`` before emitting ``"synthetic_detected"``. Overrides
                the selected preset.
            force_immediate_fake (bool, optional): If true, an on-demand ``check_now()``
                result crossing ``fake_threshold`` immediately emits ``"synthetic_detected"``.
            frame_length (int, optional): Analysis sub-window in seconds (1-4) passed to
                the API. Defaults to 2.
            silence_rms_threshold (float, optional): Windows whose normalized RMS falls
                below this are treated as silence and skipped (no API call). Set to 0 to
                analyze everything.
            request_timeout (float, optional): Per-request timeout in seconds.
            http_session (aiohttp.ClientSession, optional): An existing aiohttp
                ClientSession to use. If not provided, a new session will be created.
            transport (DetectTransport, optional): Custom Detect transport. Use this for a
                streaming Detect backend, gateway, or tests.
            zero_retention_mode (bool, optional): Enable Detect's zero-retention mode for
                the default REST transport. Defaults to True.
            extra_form_fields (Mapping[str, str | int | float | bool], optional): Extra
                Detect form fields sent by the default REST transport, for example
                ``{"use_ood_detector": True}``. Ignored when ``transport`` is provided.
        """
        super().__init__()

        if security not in _SECURITY_PRESETS:
            raise ValueError(f"security must be one of {sorted(_SECURITY_PRESETS)}")

        preset = _SECURITY_PRESETS[security]
        resolved_window_seconds = (
            window_seconds if window_seconds is not None else _DEFAULT_WINDOW_SECONDS
        )
        resolved_mode = mode if mode is not None else preset.mode
        resolved_samples = samples if samples is not None else preset.samples
        resolved_sample_interval_seconds = (
            sample_interval_seconds
            if sample_interval_seconds is not None
            else preset.sample_interval_seconds
        )
        resolved_analysis_budget_seconds = (
            analysis_budget_seconds
            if analysis_budget_seconds is not None
            else preset.analysis_budget_seconds
        )
        resolved_agreement_window = (
            agreement_window if agreement_window is not None else preset.agreement_window
        )
        resolved_min_fake_results = (
            min_fake_results if min_fake_results is not None else preset.min_fake_results
        )

        if resolved_mode not in ("sampled", "first_n", "continuous"):
            raise ValueError("mode must be one of 'sampled', 'first_n', or 'continuous'")
        if resolved_window_seconds < 2.0:
            raise ValueError("window_seconds must be >= 2.0 (Detect scores 1-4s frames)")
        if not 1 <= frame_length <= 4:
            raise ValueError("frame_length must be between 1 and 4")
        if resolved_samples < 1:
            raise ValueError("samples must be >= 1")
        if resolved_sample_interval_seconds < 0:
            raise ValueError("sample_interval_seconds must be >= 0")
        if resolved_mode == "first_n" and resolved_analysis_budget_seconds <= 0:
            raise ValueError("analysis_budget_seconds must be > 0 in first_n mode")
        _validate_threshold("fake_threshold", fake_threshold)
        _validate_threshold("likely_synthetic_threshold", likely_synthetic_threshold)
        _validate_threshold("unclear_threshold", unclear_threshold)
        if not unclear_threshold <= likely_synthetic_threshold <= fake_threshold:
            raise ValueError(
                "thresholds must satisfy unclear_threshold <= likely_synthetic_threshold "
                "<= fake_threshold"
            )
        if resolved_agreement_window < 1:
            raise ValueError("agreement_window must be >= 1")
        if resolved_min_fake_results < 1:
            raise ValueError("min_fake_results must be >= 1")
        if resolved_min_fake_results > resolved_agreement_window:
            raise ValueError("min_fake_results must be <= agreement_window")

        if transport is None:
            api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Resemble API key is required, either as argument or set RESEMBLE_API_KEY"
                    " environment variable"
                )
            transport = RestDetectTransport(
                api_key=api_key,
                base_url=base_url,
                http_session=http_session,
                zero_retention_mode=zero_retention_mode,
                extra_form_fields=extra_form_fields,
            )

        self._opts = _DetectOptions(
            security=security,
            window_seconds=resolved_window_seconds,
            mode=resolved_mode,
            analysis_budget_seconds=resolved_analysis_budget_seconds,
            samples=resolved_samples,
            sample_interval_seconds=resolved_sample_interval_seconds,
            fake_threshold=fake_threshold,
            likely_synthetic_threshold=likely_synthetic_threshold,
            unclear_threshold=unclear_threshold,
            agreement_window=resolved_agreement_window,
            min_fake_results=resolved_min_fake_results,
            force_immediate_fake=force_immediate_fake,
            frame_length=frame_length,
            silence_rms_threshold=silence_rms_threshold,
            request_timeout=request_timeout,
        )

        self._transport = transport
        self._tasks: set[asyncio.Task[Any]] = set()
        self._analysis_tasks: set[asyncio.Task[Any]] = set()
        self._results: list[DetectionResult] = []
        self._analyzed_seconds = 0.0
        self._window_index = 0
        self._samples_taken = 0
        self._verdict_emitted = False
        self._synthetic_alert_emitted = False
        self._closed = False
        self._paused = False
        self._force_pending = False

    def check_now(self) -> None:
        """Force an immediate spot-check of the next speech window, on demand.

        Independent of the ``sampled`` schedule — useful to verify the caller right before
        a sensitive action (a payment, a password reset), or to drive a "verify now" button.
        The monitor keeps serving these even after its ambient sample budget is spent.
        """
        self._force_pending = True

    def pause(self) -> None:
        """Pause collection — incoming audio is dropped until :meth:`resume`.

        Use this to analyze only the monitored party's own speech. In a voice agent,
        pause while *your* agent is speaking so its (always-synthetic) voice is never
        analyzed, avoiding false positives from echo or shared/mixed audio.
        """
        self._paused = True

    def resume(self) -> None:
        """Resume collection after :meth:`pause`."""
        self._paused = False

    @property
    def paused(self) -> bool:
        """Whether collection is currently paused."""
        return self._paused

    @property
    def security(self) -> DetectionSecurity:
        """Configured security preset."""
        return self._opts.security

    @property
    def samples(self) -> int:
        """Configured number of ambient samples for sampled mode."""
        return self._opts.samples

    @property
    def window_seconds(self) -> float:
        """Seconds of audio submitted per detection request."""
        return self._opts.window_seconds

    @property
    def results(self) -> list[DetectionResult]:
        """All per-window results collected so far."""
        return list(self._results)

    @property
    def verdict(self) -> DetectionVerdict:
        """Aggregate verdict over everything analyzed so far."""
        max_score = max((r.aggregated_score for r in self._results), default=None)
        if max_score is None:
            label: DetectionLabel = "inconclusive"
        elif self._synthetic_alert_emitted or self._has_confirmed_fake():
            label = "fake"
        elif max_score >= self._opts.unclear_threshold:
            label = "inconclusive"
        else:
            label = "real"
        return DetectionVerdict(
            label=label,
            max_score=max_score,
            analyzed_seconds=self._analyzed_seconds,
            results=self.results,
        )

    def start(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | None = None,
        *,
        track_source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE,
    ) -> None:
        """Begin monitoring a participant's audio in the given room.

        Args:
            room (rtc.Room): The room the agent is connected to.
            participant (rtc.RemoteParticipant, optional): The participant to monitor.
                If not provided, the first existing or future remote participant is
                monitored.
            track_source (rtc.TrackSource.ValueType, optional): Which audio source to
                monitor. Defaults to the participant's microphone.
        """
        if self._closed:
            raise RuntimeError("DetectionMonitor is closed")

        self._spawn(self._run(room, participant, track_source))

    def attach(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | None = None,
        *,
        track_source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE,
    ) -> None:
        """Alias for :meth:`start`, matching the high-level Resemble Detect examples."""
        self.start(room, participant, track_source=track_source)

    def monitor_track(self, track: rtc.Track, *, participant_identity: str | None = None) -> None:
        """Begin monitoring a specific audio track (lower-level alternative to start())."""
        if self._closed:
            raise RuntimeError("DetectionMonitor is closed")

        stream = rtc.AudioStream(track, sample_rate=DETECT_SAMPLE_RATE, num_channels=1)
        self._spawn(self._consume(stream, participant_identity))

    async def aclose(self) -> None:
        """Stop monitoring and release resources."""
        self._closed = True
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _spawn(self, coro: Any, *, analysis: bool = False) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if analysis:
            self._analysis_tasks.add(task)
            task.add_done_callback(self._analysis_tasks.discard)

    async def _run(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | None,
        track_source: rtc.TrackSource.ValueType,
    ) -> None:
        if participant is None:
            participant = await self._wait_for_participant(room)

        stream = rtc.AudioStream.from_participant(
            participant=participant,
            track_source=track_source,
            sample_rate=DETECT_SAMPLE_RATE,
            num_channels=1,
        )
        await self._consume(stream, participant.identity)

    async def _wait_for_participant(self, room: rtc.Room) -> rtc.RemoteParticipant:
        if room.remote_participants:
            return next(iter(room.remote_participants.values()))

        fut: asyncio.Future[rtc.RemoteParticipant] = asyncio.get_running_loop().create_future()

        def _on_join(p: rtc.RemoteParticipant) -> None:
            if not fut.done():
                fut.set_result(p)

        room.on("participant_connected", _on_join)
        try:
            return await fut
        finally:
            room.off("participant_connected", _on_join)

    async def _consume(self, stream: rtc.AudioStream, participant_identity: str | None) -> None:
        window_samples = int(self._opts.window_seconds * DETECT_SAMPLE_RATE)
        buf = bytearray()
        stream_pos = 0.0  # seconds of audio consumed from the stream
        cooldown_until = 0.0  # sampled mode: ignore audio until this stream position

        try:
            async for event in stream:
                # paused (e.g. while our own agent is speaking): drop audio and never let
                # a window straddle the pause boundary, so the agent's voice is never scored
                if self._paused:
                    if buf:
                        buf.clear()
                    continue

                frame = event.frame
                buf.extend(frame.data)
                if len(buf) // 2 < window_samples:
                    continue

                window = bytes(buf[: window_samples * 2])
                del buf[: window_samples * 2]
                window_start = stream_pos
                stream_pos += self._opts.window_seconds

                forced = self._force_pending  # check_now(): analyze the next window right away

                if self._opts.mode == "sampled" and not forced:
                    # between spot-checks, drain audio without analyzing it
                    if stream_pos <= cooldown_until:
                        continue
                    # ambient budget spent: idle, but keep listening for on-demand checks
                    if self._samples_taken >= self._opts.samples:
                        continue

                if self._budget_exhausted():
                    if self._opts.mode == "first_n":
                        await self._flush_and_emit_verdict()
                        return
                    continue

                if self._is_silence(window) and not forced:
                    logger.debug("skipping silent window", extra={"window_start": window_start})
                    continue

                self._analyzed_seconds += self._opts.window_seconds
                index = self._window_index
                self._window_index += 1
                self._spawn(
                    self._analyze_window(
                        window,
                        index=index,
                        window_start=window_start,
                        participant_identity=participant_identity,
                        forced=forced,
                    ),
                    analysis=True,
                )

                if forced:
                    self._force_pending = False

                if self._opts.mode == "sampled":
                    if not forced:
                        self._samples_taken += 1
                        cooldown_until = stream_pos + self._opts.sample_interval_seconds
                        if self._samples_taken >= self._opts.samples:
                            # settle the ambient verdict but stay alive for check_now()
                            await self._flush_and_emit_verdict()
        finally:
            await stream.aclose()
            if not self._closed:
                await self._flush_and_emit_verdict()

    async def _flush_and_emit_verdict(self) -> None:
        # let in-flight window analyses land before settling the verdict
        while self._analysis_tasks:
            await asyncio.gather(*list(self._analysis_tasks), return_exceptions=True)
        self._emit_verdict()

    def _budget_exhausted(self) -> bool:
        return (
            self._opts.mode == "first_n"
            and self._analyzed_seconds >= self._opts.analysis_budget_seconds
        )

    def _is_silence(self, pcm16: bytes) -> bool:
        if self._opts.silence_rms_threshold <= 0:
            return False
        samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        return rms < self._opts.silence_rms_threshold

    def _emit_verdict(self) -> None:
        if self._verdict_emitted:
            return
        self._verdict_emitted = True
        self.emit("verdict", self.verdict)

    async def _analyze_window(
        self,
        pcm16: bytes,
        *,
        index: int,
        window_start: float,
        participant_identity: str | None,
        forced: bool = False,
    ) -> None:
        try:
            started = time.monotonic()
            item = await self._submit(pcm16)
            latency = time.monotonic() - started
        except Exception as exc:
            logger.warning("resemble detect request failed", exc_info=exc)
            self.emit("error", exc)
            return

        metrics = item.get("metrics") or {}
        try:
            # the API pads the score array with None for a trailing partial frame
            scores = [float(s) for s in metrics.get("score") or [] if s is not None]
            aggregated = float(metrics["aggregated_score"])
            label: DetectionLabel = "fake" if metrics["label"] == "fake" else "real"
        except (KeyError, TypeError, ValueError):
            self.emit("error", APIConnectionError(f"unexpected detect response shape: {item}"))
            return

        result = DetectionResult(
            label=label,
            aggregated_score=aggregated,
            scores=scores,
            consistency=_opt_float(metrics.get("consistency")),
            window_index=index,
            window_start=window_start,
            window_end=window_start + self._opts.window_seconds,
            participant_identity=participant_identity,
            detection_uuid=item.get("uuid"),
            latency=latency,
            forced=forced,
            raw=item,
        )
        self._results.append(result)
        # Concurrent API calls can finish out of order; agreement is defined over audio
        # windows, not request completion order.
        self._results.sort(key=lambda detection: detection.window_index)
        logger.debug(
            "detect window analyzed",
            extra={
                "label": result.label,
                "aggregated_score": result.aggregated_score,
                "window_index": index,
                "latency": round(latency, 3),
            },
        )
        self.emit("result", result)
        if aggregated >= self._opts.fake_threshold:
            self.emit("fake_detected", result)
        if self._should_emit_synthetic_detected(result):
            self._synthetic_alert_emitted = True
            self.emit("synthetic_detected", result)

    async def _submit(self, pcm16: bytes) -> dict[str, Any]:
        return await self._transport.submit(
            pcm16,
            frame_length=self._opts.frame_length,
            request_timeout=self._opts.request_timeout,
        )

    def _has_confirmed_fake(self) -> bool:
        if not self._results:
            return False
        recent = self._results[-self._opts.agreement_window :]
        fake_results = [r for r in recent if r.aggregated_score >= self._opts.fake_threshold]
        return len(fake_results) >= self._opts.min_fake_results

    def _should_emit_synthetic_detected(self, result: DetectionResult) -> bool:
        if self._synthetic_alert_emitted:
            return False
        if result.aggregated_score < self._opts.fake_threshold:
            return False
        if result.forced and self._opts.force_immediate_fake:
            return True
        return self._has_confirmed_fake()


class ResembleDetect(DetectionMonitor):
    """High-level LiveKit Detect component.

    This is a semantic alias for :class:`DetectionMonitor` with the developer-facing name
    used in integration examples:

    .. code-block:: python

        detect = resemble.ResembleDetect(security="standard")
        detect.attach(ctx.room)
        detect.on("synthetic_detected", on_synthetic)
    """


class RestDetectTransport:
    """Default transport for Resemble Detect's REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = RESEMBLE_DETECT_API_URL,
        http_session: aiohttp.ClientSession | None = None,
        zero_retention_mode: bool = True,
        extra_form_fields: Mapping[str, DetectFormFieldValue] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = http_session
        self._extra_form_fields = {
            "zero_retention_mode": _form_value(zero_retention_mode),
            **{key: _form_value(value) for key, value in (extra_form_fields or {}).items()},
        }

    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        form = aiohttp.FormData()
        form.add_field(
            "file",
            _wav_bytes(pcm16),
            filename="window.wav",
            content_type="audio/wav",
        )
        form.add_field("modality", "audio")
        form.add_field("frame_length", str(frame_length))
        for key, value in self._extra_form_fields.items():
            form.add_field(key, value)

        async with self._ensure_session().post(
            f"{self._base_url}/detect",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Prefer": "wait",
            },
            data=form,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                raise APIStatusError(
                    message="resemble detect request failed",
                    status_code=resp.status,
                    request_id=None,
                    body=body[:500],
                )
            payload = await resp.json()

        item: dict[str, Any] = payload.get("item") or {}
        if item.get("status") == "completed":
            return item

        # `Prefer: wait` normally completes synchronously; poll as a fallback
        uuid = item.get("uuid")
        if not uuid:
            raise APIConnectionError(f"detect response missing uuid: {payload}")
        return await self._poll(uuid, request_timeout=request_timeout)

    async def _poll(self, uuid: str, *, request_timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + request_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            async with self._ensure_session().get(
                f"{self._base_url}/detect/{uuid}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as resp:
                if resp.status != 200:
                    continue
                payload = await resp.json()
            item: dict[str, Any] = payload.get("item") or {}
            if item.get("status") == "completed":
                return item
        raise APIConnectionError(f"detection {uuid} did not complete in time")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session


def _wav_bytes(pcm16: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(DETECT_SAMPLE_RATE)
        wf.writeframes(pcm16)
    return out.getvalue()


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_label(label: DetectionLabel) -> DetectionNormalizedLabel:
    if label == "fake":
        return "synthetic"
    return label


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _form_value(value: DetectFormFieldValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
