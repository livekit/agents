from __future__ import annotations

import asyncio
from typing import cast

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import aio

from .utils import stt as timing

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _endpoint(end_time: float = 0.0, *, wall_time: float | None = None) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.END_OF_SPEECH,
        speech_end_time=wall_time,
        alternatives=[stt.SpeechData(language="en", text="", end_time=end_time)],
    )


def _check(events, **kwargs):
    return timing.assert_speech_end_time(
        events, audio_started_at=1000.0, expected_end=44.4, **kwargs
    )


@pytest.mark.parametrize("end_time", [44.0, 44.4, 44.8])
def test_relative_endpoint_uses_the_audio_timeline(end_time):
    assert _check([_endpoint(end_time)]) == pytest.approx(end_time)


def test_absolute_endpoint_takes_precedence_over_alternatives():
    assert _check([_endpoint(99.0, wall_time=1044.4)]) == pytest.approx(44.4)


def test_latest_endpoint_is_checked_instead_of_an_earlier_turn():
    assert _check([_endpoint(20.0), _endpoint(44.4)]) == pytest.approx(44.4)
    with pytest.raises(AssertionError, match="no speech-end timestamp"):
        _check([_endpoint(44.4), _endpoint()])


def test_an_earlier_endpoint_cannot_hide_a_later_open_turn():
    with pytest.raises(AssertionError, match="open speech turn"):
        _check([_endpoint(44.4), stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)])


@pytest.mark.parametrize("untimed_reason", [None, "Provider does not supply timestamps"])
def test_missing_endpoint_is_never_exempted(untimed_reason):
    with pytest.raises(AssertionError, match="No END_OF_SPEECH"):
        _check([], untimed_reason=untimed_reason)


def test_final_transcript_cannot_hide_an_untimed_endpoint():
    final = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text="hello", end_time=44.4)],
    )
    with pytest.raises(AssertionError, match="no speech-end timestamp"):
        _check([final, _endpoint()])


def test_explicit_exception_allows_only_missing_timing():
    assert _check([_endpoint()], untimed_reason="Provider has no timing") is None
    with pytest.raises(AssertionError, match="EOS reports"):
        _check([_endpoint(46.6)], untimed_reason="Provider has no timing")


@pytest.mark.parametrize("end_time", [20.0, 45.36, 46.6, float("inf")])
def test_wrong_speech_boundaries_fail(end_time):
    with pytest.raises(AssertionError, match="EOS reports"):
        _check([_endpoint(end_time)])


def test_nonfinite_absolute_timestamp_does_not_fall_back_to_valid_alternative():
    with pytest.raises(AssertionError, match="EOS reports"):
        _check([_endpoint(44.4, wall_time=float("nan"))])


class _Stream:
    def __init__(
        self, *, endpoint_first: bool, finish_early: bool = False, close_on_input: bool = True
    ):
        self.events = aio.Chan[stt.SpeechEvent]()
        self.endpoint_first = endpoint_first
        self.finish_early = finish_early
        self.close_on_input = close_on_input
        self.pushed = 0
        self.input_closed = asyncio.Event()

    def _turn(self):
        self.events.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        final = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text="hello")],
        )
        endpoint = _endpoint(44.4)
        for event in [endpoint, final] if self.endpoint_first else [final, endpoint]:
            self.events.send_nowait(event)

    def push_frame(self, frame):
        self.pushed += 1
        if self.finish_early and self.pushed == 1:
            self._turn()

    def end_input(self):
        self.input_closed.set()
        if not self.finish_early:
            self._turn()
        if self.close_on_input:
            self.events.close()

    def __aiter__(self):
        return self.events.__aiter__()


@pytest.mark.parametrize("endpoint_first", [False, True])
@pytest.mark.parametrize("finish_early", [False, True])
async def test_collector_drains_both_final_and_endpoint(endpoint_first, finish_early):
    stream = _Stream(endpoint_first=endpoint_first, finish_early=finish_early)
    frames = [rtc.AudioFrame.create(16000, 1, 160)] * 3
    async with asyncio.timeout(1):
        events, _ = await timing.collect_stream_events(cast(stt.RecognizeStream, stream), frames)
    assert stream.input_closed.is_set() and stream.pushed == len(frames)
    assert len(events) == 3
    assert {event.type for event in events} == {
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    }


@pytest.mark.parametrize("new_turn", [False, True])
async def test_collector_keeps_finals_and_endpoints_that_arrive_after_an_earlier_eos(new_turn):
    stream = _Stream(endpoint_first=False, finish_early=True, close_on_input=False)

    async def late_events():
        await stream.input_closed.wait()
        await asyncio.sleep(0.01)
        if new_turn:
            stream.events.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        stream.events.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="last segment")],
            )
        )
        if new_turn:
            stream.events.send_nowait(_endpoint(46.6))
        stream.events.close()

    producer = asyncio.create_task(late_events())
    try:
        async with asyncio.timeout(1):
            events, _ = await timing.collect_stream_events(
                cast(stt.RecognizeStream, stream), [rtc.AudioFrame.create(16000, 1, 160)]
            )
        finals = [event for event in events if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [event.alternatives[0].text for event in finals] == ["hello", "last segment"]
        if new_turn:
            with pytest.raises(AssertionError, match="46.600s"):
                _check(events)
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


async def test_collector_bounds_the_drain_for_a_stream_that_stays_open():
    stream = _Stream(endpoint_first=False, close_on_input=False)
    async with asyncio.timeout(1):
        events, _ = await timing.collect_stream_events(
            cast(stt.RecognizeStream, stream), [], drain_timeout=0.01
        )
    assert len(events) == 3


async def test_collector_propagates_errors_during_the_drain():
    class FailingStream(_Stream):
        async def __aiter__(self):
            await self.input_closed.wait()
            yield _endpoint(44.4)
            await asyncio.sleep(0)
            raise RuntimeError("late provider failure")

    stream = FailingStream(endpoint_first=False)
    with pytest.raises(RuntimeError, match="late provider failure"):
        await timing.collect_stream_events(cast(stt.RecognizeStream, stream), [])


async def test_pacing_does_not_accumulate_scheduler_or_push_overhead(monkeypatch):
    now = 0.0
    loop = asyncio.get_running_loop()
    sleep = asyncio.sleep
    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(timing.time, "time", lambda: 1000.0 + now)

    async def delayed_sleep(delay):
        nonlocal now
        now += delay + 0.003
        await sleep(0)

    monkeypatch.setattr(timing.asyncio, "sleep", delayed_sleep)
    stream = _Stream(endpoint_first=False)
    pushes = []
    push_frame = stream.push_frame

    def slow_push(frame):
        nonlocal now
        pushes.append(now)
        now += 0.005
        push_frame(frame)

    stream.push_frame = slow_push
    frames = [rtc.AudioFrame.create(16000, 1, 320)] * 3
    _, origin = await timing.collect_stream_events(cast(stt.RecognizeStream, stream), frames)
    assert origin == 1000.0
    assert pushes == pytest.approx([0.023, 0.043, 0.063])
