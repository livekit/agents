# SPDX-License-Identifier: Apache-2.0
"""Session-bound output accounting through the public LiveKit audio contract."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rimelabs_sdk import realtime as r

from livekit import rtc
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, PlaybackFinishedEvent

_KEY = "rime.prism.output"


@dataclass(eq=False)
class _Output:
    owner: Playback
    response_id: str
    generated: int = 0
    accepted: int = 0
    played: int = 0
    invalid: bool = False


@dataclass(frozen=True)
class _Frame:
    output: _Output
    offset: int
    samples: int


@dataclass
class _Response:
    ref: r.ResponseRef
    outputs: list[_Output] = field(default_factory=list)
    ended: bool = False
    interrupted: bool = False
    reporting: bool = False
    hold: Any = None
    deadline: asyncio.TimerHandle | None = None


class Playback:
    def __init__(
        self,
        sdk: r.RealtimeSession,
        mode: Literal["auto", "estimated", "discard"],
        request_timeout: float,
        report: Callable[[r.PlaybackReport], Awaitable[None]],
        spawn: Callable[..., asyncio.Task[Any]],
        measured_enabled: Callable[[], bool],
    ):
        self._sdk = sdk
        self._mode = mode
        self._timeout = request_timeout
        self._report = report
        self._spawn = spawn
        self._measured_enabled = measured_enabled
        self._responses: OrderedDict[str, _Response] = OrderedDict()
        self._closed = False
        self._worker: asyncio.Task[Any] | None = None
        self._queue: deque[r.PlaybackReport] = deque()
        self._finalized: OrderedDict[str, None] = OrderedDict()

    def start(self, ref: r.ResponseRef) -> None:
        if self._mode == "estimated" or (self._mode == "auto" and not self._measured_enabled()):
            return
        if len(self._responses) >= 128:
            # Drop only a local observation. This never reports playback completion.
            for key, value in self._responses.items():
                if value.ended and not value.reporting:
                    self._release(key)
                    break
            else:
                return
        state = _Response(ref)
        hold = getattr(self._sdk, "hold_response", None)
        if hold is not None:
            state.hold = hold(ref)
        self._responses[ref.response_id] = state

    def output(self, ref: r.OutputRef) -> _Output | None:
        state = self._responses.get(ref.response.response_id)
        if state is None:
            return None
        token = _Output(self, ref.response.response_id)
        state.outputs.append(token)
        return token

    def tag(self, token: _Output | None, frame: rtc.AudioFrame) -> None:
        if token is not None:
            frame.userdata[_KEY] = _Frame(token, token.generated, frame.samples_per_channel)
            token.generated += frame.samples_per_channel

    def end(self, ref: r.ResponseRef, *, interrupted: bool = False) -> None:
        state = self._responses.get(ref.response_id)
        if state is None or state.ended:
            return
        state.ended = True
        state.interrupted |= interrupted
        duration = sum(o.generated for o in state.outputs) / 24000
        state.deadline = asyncio.get_running_loop().call_later(
            duration + self._timeout, self._expire, ref.response_id
        )
        self._check(state)

    def interrupt(self, report: r.PlaybackInterrupted) -> None:
        if report.output.response.response_id in self._finalized:
            return
        state = self._responses.get(report.output.response.response_id)
        if state is not None:
            state.interrupted = True
            # An interruption can wait behind another report. Keep its SDK
            # reference until that operation settles, including after the normal
            # unobserved-playback cleanup deadline.
            state.reporting = True
        if not self._closed:
            self._finalize(report.output.response.response_id)
            self._queue.append(report)
            self._wake()

    def played(self, token: _Output, samples: int) -> None:
        state = self._responses.get(token.response_id)
        if self._closed or state is None or token not in state.outputs:
            return
        token.played += samples
        self._check(state)

    def _check(self, state: _Response) -> None:
        if self._closed or not state.ended or state.interrupted or state.reporting:
            return
        if self._mode != "discard" and not (
            state.outputs
            and all(
                not o.invalid and o.generated > 0 and o.generated == o.accepted == o.played
                for o in state.outputs
            )
        ):
            return
        state.reporting = True
        self._queue.append(
            r.PlaybackFinished(
                response=state.ref,
                played_ms=0
                if self._mode == "discard"
                else sum(o.played for o in state.outputs) / 24,
            )
        )
        self._wake()

    def _wake(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = self._spawn(self._drain())

    async def _drain(self) -> None:
        while self._queue and not self._closed:
            report = self._queue.popleft()
            if isinstance(report, r.PlaybackFinished):
                state = self._responses.get(report.response.response_id)
                if state is None or state.interrupted:
                    continue
                self._finalize(report.response.response_id)
            try:
                await self._report(report)
            finally:
                key = (
                    report.response.response_id
                    if isinstance(report, r.PlaybackFinished)
                    else report.output.response.response_id
                )
                self._release(key)

    async def settle(self) -> None:
        if self._worker is not None:
            await asyncio.shield(self._worker)

    def _expire(self, key: str) -> None:
        state = self._responses.get(key)
        if state is not None and not state.reporting:
            self._release(key)

    def _release(self, key: str) -> None:
        state = self._responses.pop(key, None)
        if state is not None:
            if state.deadline is not None:
                state.deadline.cancel()
            if state.hold is not None:
                state.hold.release()

    def _finalize(self, key: str) -> None:
        self._finalized[key] = None
        while len(self._finalized) > 256:
            self._finalized.popitem(last=False)

    def close(self) -> None:
        self._closed = True
        self._queue.clear()
        for key in list(self._responses):
            self._release(key)


@dataclass
class _Segment:
    token: _Output | None
    samples: int = 0
    duration: float = 0
    sealed: bool = False
    invalid: bool = False


class PrismAudioOutput(AudioOutput):
    """Read identity before resampling. Unproven segments use server estimates."""

    def __init__(self, output: AudioOutput):
        # Manage the leaf directly: no private proxy or core state is required.
        super().__init__(
            label="PrismAudioOutput",
            capabilities=AudioOutputCapabilities(pause=output.can_pause),
            sample_rate=None,
        )
        self._output = output
        self._segments: deque[_Segment] = deque()
        self._segment: _Segment | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._format: tuple[int, int] | None = None
        self._unproven = False
        self._flush_task: asyncio.Task[None] | None = None
        output.on("playback_finished", self._finished)
        output.on("playback_started", lambda ev: self.on_playback_started(created_at=ev.created_at))
        # Older released cores have no progress event. Event registration is still safe.
        output.on("playback_progressed", self._progressed)

    def _progressed(self, ev: Any) -> None:
        method = getattr(self, "on_playback_progressed", None)
        if method is not None:
            method(started_at=ev.started_at, offset=ev.offset, duration=ev.duration)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self._flush_task is not None:
            await asyncio.shield(self._flush_task)
            self._flush_task = None
        await super().capture_frame(frame)
        identity = frame.userdata.get(_KEY)
        token = identity.output if isinstance(identity, _Frame) else None
        if token is not None:
            assert isinstance(identity, _Frame)
            if token.owner._closed:
                token = None
            elif (
                identity.offset != token.accepted
                or identity.samples != frame.samples_per_channel
                or frame.sample_rate != 24000
                or frame.num_channels != 1
            ):
                token.invalid = True
        if self._segment is None:
            self._segment = _Segment(token)
            if len(self._segments) >= 128:
                self._unproven = True
                for segment in self._segments:
                    if segment.token is not None:
                        segment.token.invalid = True
                self._segments.clear()
            self._segments.append(self._segment)
        segment = self._segment
        if token is not segment.token or self._unproven:
            segment.invalid = True
            if token is not None:
                token.invalid = True
            if segment.token is not None:
                segment.token.invalid = True
        # Register before capture: the sink may emit an event synchronously.
        segment.samples += frame.samples_per_channel
        segment.duration += frame.duration
        if token is not None:
            token.accepted += frame.samples_per_channel
        try:
            rate = self._output.sample_rate
            if rate is not None and rate != frame.sample_rate:
                fmt = (frame.sample_rate, frame.num_channels)
                if self._resampler is None:
                    self._resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=rate,
                        num_channels=frame.num_channels,
                    )
                    self._format = fmt
                elif fmt != self._format:
                    segment.invalid = True
                    raise ValueError("Audio format changed inside a playback segment")
                for converted in self._resampler.push(frame):
                    await self._output.capture_frame(converted)
            else:
                await self._output.capture_frame(frame)
        except BaseException:
            segment.invalid = True
            if token is not None:
                token.invalid = True
            raise

    def flush(self) -> None:
        super().flush()
        segment, self._segment = self._segment, None
        if segment is not None:
            segment.sealed = True
        # AudioResampler has synchronous flush but the sink capture is async.
        # The final resampler tail must be delivered before downstream flush.
        if self._resampler is not None:
            resampler, self._resampler = self._resampler, None
            task = self._flush_task = asyncio.create_task(self._flush_resampled(resampler, segment))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        else:
            self._output.flush()

    async def _flush_resampled(
        self, resampler: rtc.AudioResampler, segment: _Segment | None
    ) -> None:
        try:
            for frame in resampler.flush():
                await self._output.capture_frame(frame)
            self._output.flush()
        except Exception:
            if segment is not None and segment.token is not None:
                segment.token.invalid = True
            self.on_playback_finished(playback_position=0, interrupted=True)

    def clear_buffer(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        for segment in self._segments:
            segment.invalid = True
            if segment.token is not None:
                segment.token.invalid = True
        self._resampler = None
        self._output.clear_buffer()

    def _finished(self, event: PlaybackFinishedEvent) -> None:
        segment = self._segments.popleft() if self._segments else None
        if segment is not None and segment.token is not None:
            token = segment.token
            if (
                self._unproven
                or segment.invalid
                or not segment.sealed
                or event.interrupted
                or not math.isfinite(event.playback_position)
                or event.playback_position + 1 / 24000 < segment.duration
            ):
                token.invalid = True
            else:
                token.owner.played(token, segment.samples)
        self.on_playback_finished(
            playback_position=event.playback_position,
            interrupted=event.interrupted,
            synchronized_transcript=event.synchronized_transcript,
        )

    def on_attached(self) -> None:
        self._output.on_attached()

    def on_detached(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        for segment in self._segments:
            if segment.token is not None:
                segment.token.invalid = True
        self._output.on_detached()

    def pause(self) -> None:
        self._output.pause()

    def resume(self) -> None:
        self._output.resume()
