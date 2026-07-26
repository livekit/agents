"""Regression tests: overlap state must survive events that are not a new agent turn.

``_overlap_started`` is the gate that lets user audio reach the bargein gateway at all,
and the only thing that ever raises it is an ``_OverlapSpeechStartedSentinel``, which in
turn only comes from a VAD start-of-speech. VAD does not re-announce speech that is
already under way, so once the flag is cleared mid-overlap nothing can re-arm it: every
remaining frame of the interruption is dropped, no verdict is emitted, and the agent
talks straight through the user.

One agent turn can span several speech segments — the reply that follows a tool call, or
a queued ``say()`` — and those later segments raise ``_AgentSpeechStartedSentinel`` again
with no end sentinel in between. Treating that as a new turn used to reset the overlap
the user was in the middle of (see issue #6548, ported from livekit/agents-js#2117).
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from unittest.mock import AsyncMock

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionWebSocketStream,
    OverlappingSpeechEvent,
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.types import APIConnectOptions

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)
BATCH_SAMPLES = 1600  # detection_interval (0.1s) * sample_rate (16000)


def _make_audio_frame(*, num_samples: int = BATCH_SAMPLES) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.zeros(num_samples, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=num_samples,
    )


class _FakeGatewayWS:
    """Stand-in for the bargein gateway: answers every audio batch promptly.

    Without prompt answers the transport's own inference timeout would tear the
    stream down mid-test. Set ``bargein_next`` to answer the next batch with a
    ``bargein_detected`` verdict instead of ``inference_done``.
    """

    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self.sent_audio: list[bytes] = []
        self.bargein_next = False
        self._rx: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self._push_json({"type": "session.created", "default_threshold": 0.5})

    def _push_json(self, payload: dict) -> None:
        self._rx.put_nowait(
            aiohttp.WSMessage(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload), extra=None)
        )

    async def send_str(self, data: str) -> None:  # session.create / session.close
        pass

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio.append(data)
        created_at = struct.unpack("<Q", data[:8])[0]
        if self.bargein_next:
            self.bargein_next = False
            self._push_json(
                {
                    "type": "bargein_detected",
                    "created_at": created_at,
                    "probabilities": [0.91, 0.93, 0.95],
                    "prediction_duration": 0.02,
                }
            )
        else:
            self._push_json(
                {
                    "type": "inference_done",
                    "created_at": created_at,
                    "probabilities": [0.01, 0.02],
                    "prediction_duration": 0.02,
                }
            )

    async def receive(self) -> aiohttp.WSMessage:
        return await self._rx.get()

    async def close(self) -> None:
        self.closed = True


def _make_harness() -> tuple[
    _FakeGatewayWS, InterruptionWebSocketStream, list[OverlappingSpeechEvent]
]:
    ws = _FakeGatewayWS()
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    detector = AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=session,
    )
    events: list[OverlappingSpeechEvent] = []
    detector.on("overlapping_speech", lambda ev: events.append(ev))
    stream = detector.stream(conn_options=CONN_OPTIONS)
    return ws, stream, events


async def _wait_until(cond, timeout: float = 5.0, message: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(message or "condition not met in time")
        await asyncio.sleep(0.01)


async def _settle(cond, duration: float = 0.15, message: str = "") -> None:
    """Assert that ``cond`` stays true for ``duration`` seconds."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        assert cond(), message
        await asyncio.sleep(0.01)


class TestOverlapStateRetention:
    @pytest.mark.asyncio
    async def test_second_speech_segment_keeps_open_overlap(self) -> None:
        """A later segment of the same turn must not stop user audio from
        reaching the gateway, and the end-of-overlap verdict must still surface."""
        ws, stream, events = _make_harness()
        try:
            stream.push_frame(_AgentSpeechStartedSentinel())
            stream.push_frame(_make_audio_frame())
            stream.push_frame(
                _OverlapSpeechStartedSentinel(speech_duration=0.1, started_at=time.time())
            )
            stream.push_frame(_make_audio_frame())
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(ws.sent_audio) >= 2, message="audio never reached the gateway"
            )

            # second segment of the same turn (tool reply, queued say()):
            # no _AgentSpeechEndedSentinel precedes it
            sent_before = len(ws.sent_audio)
            stream.push_frame(_AgentSpeechStartedSentinel())
            stream.push_frame(_make_audio_frame())
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(ws.sent_audio) >= sent_before + 2,
                message="user audio stopped reaching the gateway after the second segment",
            )

            stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time()))
            await _wait_until(
                lambda: len(events) >= 1,
                message="no OverlappingSpeechEvent after the overlap ended",
            )
            assert events[0].num_requests > 0
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_bargein_verdict_after_second_speech_segment(self) -> None:
        """Restoring the gate is only useful if the recovered audio can still
        produce an ``is_interruption`` verdict."""
        ws, stream, events = _make_harness()
        try:
            stream.push_frame(_AgentSpeechStartedSentinel())
            stream.push_frame(
                _OverlapSpeechStartedSentinel(speech_duration=0.1, started_at=time.time())
            )
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(ws.sent_audio) >= 1, message="audio never reached the gateway"
            )

            stream.push_frame(_AgentSpeechStartedSentinel())
            ws.bargein_next = True
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(events) >= 1,
                message="bargein verdict never surfaced after the second segment",
            )
            assert events[0].is_interruption is True
            assert events[0].num_requests > 0
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_new_agent_turn_still_resets_overlap_state(self) -> None:
        """The guard must be narrow: a genuine new turn (end sentinel seen) wipes
        the overlap, and audio is not forwarded until a fresh overlap is announced."""
        ws, stream, events = _make_harness()
        try:
            stream.push_frame(_AgentSpeechStartedSentinel())
            stream.push_frame(
                _OverlapSpeechStartedSentinel(speech_duration=0.1, started_at=time.time())
            )
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(ws.sent_audio) >= 1, message="audio never reached the gateway"
            )

            # the turn ends, then a new one begins: the old overlap is gone, so
            # audio must stay gated until a fresh VAD onset opens a new overlap
            stream.push_frame(_AgentSpeechEndedSentinel())
            stream.push_frame(_AgentSpeechStartedSentinel())
            sent_before = len(ws.sent_audio)
            stream.push_frame(_make_audio_frame())
            stream.push_frame(_make_audio_frame())
            await _settle(
                lambda: len(ws.sent_audio) == sent_before,
                message="audio was forwarded without an open overlap",
            )

            stream.push_frame(
                _OverlapSpeechStartedSentinel(speech_duration=0.1, started_at=time.time())
            )
            stream.push_frame(_make_audio_frame())
            await _wait_until(
                lambda: len(ws.sent_audio) > sent_before,
                message="audio never resumed for the new turn's overlap",
            )

            stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time()))
            await _wait_until(
                lambda: len(events) >= 1,
                message="no OverlappingSpeechEvent for the new turn's overlap",
            )
            assert events[-1].num_requests > 0
        finally:
            await stream.aclose()
