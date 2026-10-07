"""``stt_node`` must measure ``start_time_offset`` from the pipeline's own anchor.

Issue #7651, second cause. A plugin reports word times relative to the start of
the audio it has received, plus whatever ``start_time_offset`` the node set. The
recognition loop turns that back into wall clock as
``_stt_pipeline.input_started_at + end_time``, and that anchor is stamped on the
first frame to reach the pipeline.

So the offset has to be measured from the same anchor. The default node used to
fall back to the recording or session start when the pipeline had no anchor yet,
which is exactly the case for a pipeline created after the session started -- on
every handoff to an agent that overrides ``stt_node``. The time since the session
start was then counted twice: once in the offset and again when the new
pipeline's own anchor was stamped. Every timestamp on such a stream landed in the
future, got clamped to ``now``, and reported the arrival time of the event.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, Agent, LanguageCode, NotGivenOr
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.agent import ModelSettings

pytestmark = pytest.mark.unit

SESSION_AGE = 30.0
"""How long the session has been running when the new STT stream is created."""


class _OffsetRecordingStream(RecognizeStream):
    """Records the offset the node sets, then ends so the node returns."""

    def __init__(self, *, stt: _OffsetRecordingSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._recorder: _OffsetRecordingSTT = stt

    async def _run(self) -> None:
        self._recorder.recorded_offset = self.start_time_offset
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        text="hello",
                        language=LanguageCode("en"),
                        confidence=0.9,
                        start_time=self.start_time_offset,
                        end_time=self.start_time_offset + 1.0,
                    )
                ],
            )
        )


class _OffsetRecordingSTT(STT):
    def __init__(self) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))
        self.recorded_offset: float | None = None

    async def _recognize_impl(self, buffer, *, language, conn_options) -> SpeechEvent:
        raise NotImplementedError("streaming only")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _OffsetRecordingStream:
        return _OffsetRecordingStream(stt=self, conn_options=conn_options)


async def _no_audio() -> AsyncIterator[rtc.AudioFrame]:
    """The node starts its stream before any frame arrives; that is the case here."""
    if False:  # pragma: no cover - makes this an async generator
        yield  # type: ignore[unreachable]


def _agent_with(*, pipeline_anchor: float | None) -> tuple[Agent, _OffsetRecordingSTT]:
    now = time.time()
    stt_impl = _OffsetRecordingSTT()

    activity = MagicMock()
    activity.stt = stt_impl
    activity.session.conn_options.stt_conn_options = DEFAULT_API_CONNECT_OPTIONS
    activity._audio_recognition._input_started_at = pipeline_anchor
    # Both fallbacks the old code preferred over "no anchor yet" are set, and far
    # enough back that using either is unmistakable in the assertion.
    activity.session._recorder_io.recording_started_at = now - SESSION_AGE
    activity.session._started_at = now - SESSION_AGE

    agent = Agent(instructions="test")
    agent._activity = activity
    return agent, stt_impl


async def _run_node(agent: Agent, audio: AsyncIterable[rtc.AudioFrame]) -> None:
    gen = Agent.default.stt_node(agent, audio, ModelSettings())
    try:
        await asyncio.wait_for(gen.__anext__(), timeout=5)
    finally:
        await gen.aclose()


async def test_new_pipeline_offsets_from_its_own_first_frame() -> None:
    """No anchor yet means the first frame is about to arrive, so the offset is 0.

    This is the handoff case. The old code used the session start here and the
    offset came out at ~SESSION_AGE, which the recognition loop then added to the
    new pipeline's anchor as well.
    """
    agent, stt_impl = _agent_with(pipeline_anchor=None)

    await _run_node(agent, _no_audio())

    assert stt_impl.recorded_offset == pytest.approx(0.0, abs=0.5)
    assert stt_impl.recorded_offset is not None
    assert stt_impl.recorded_offset < SESSION_AGE / 2


async def test_reused_pipeline_offsets_from_the_existing_anchor() -> None:
    """A reused pipeline still needs the gap between its audio start and now.

    This is the default-``stt_node`` case, where the pipeline outlives the stream,
    and it is the reason the offset exists at all.
    """
    gap = 7.5
    agent, stt_impl = _agent_with(pipeline_anchor=time.time() - gap)

    await _run_node(agent, _no_audio())

    assert stt_impl.recorded_offset == pytest.approx(gap, abs=0.5)
