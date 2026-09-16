"""Transcript attribution and failure handling in AMDRacingSTT, without a session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.amd._transcription import AMDRacingSTT

from .fake_stt import DrainingStream, DrainingSTT

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.fixture
async def racing_stt() -> AsyncIterator[tuple[AMDRacingSTT, DrainingStream]]:
    model = DrainingSTT()
    async with aclosing(AMDRacingSTT(model, APIConnectOptions(max_retry=0))) as racing:
        racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        yield racing, model.streams[0]
    await model.aclose()


async def wait_for_transcript(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["session", "amd"])
async def test_first_transcript_selects_source_for_the_run(
    racing_stt: tuple[AMDRacingSTT, DrainingStream], winner: str
) -> None:
    racing, stream = racing_stt
    if winner == "session":
        racing.push_transcript("session transcript")
    stream.send_fake_transcript("amd transcript")
    await asyncio.sleep(0.001)
    turn = racing.end_turn("session transcript")
    assert turn.source == winner
    assert turn.transcript == f"{winner} transcript"

    if winner == "amd":
        racing.push_transcript("session is faster this time")
        stream.send_fake_transcript("amd next turn")
    await asyncio.sleep(0.001)
    second = racing.end_turn("session next turn")
    assert second.source == winner
    assert second.transcript == f"{winner} next turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("started", ["no_audio", "reader_pending", "reader_running"])
async def test_session_winner_stops_amd_stream_and_prevents_more_audio(started: str) -> None:
    model = DrainingSTT()
    model.aclose = Mock(wraps=model.aclose)
    frame = rtc.AudioFrame.create(16000, 1, 320)
    async with aclosing(AMDRacingSTT(model, APIConnectOptions(max_retry=0))) as racing:
        if started != "no_audio":
            racing.push_audio(frame)
            stream = model.streams[0]
            stream.push_frame = Mock(wraps=stream.push_frame)
            if started == "reader_running":
                await wait_for_transcript(lambda: bool(stream.frames))

        racing.push_transcript("Hello from session.")
        racing.push_audio(frame)

        if started != "no_audio":
            await wait_for_transcript(lambda: stream._task.done() and stream._metrics_task.done())
            stream.push_frame.assert_not_called()
            assert len(model.streams) == 1
        else:
            assert model.streams == []

        assert racing.end_turn("").transcript == "Hello from session."
        assert racing.end_turn("Next turn.").source == "session"
        racing.push_audio(frame)
        assert len(model.streams) == (0 if started == "no_audio" else 1)
    model.aclose.assert_not_called()
    await model.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "started", ["no_audio", "reader_pending", "reader_running", "amd_selected"]
)
async def test_close_stops_stream_and_prevents_more_audio(started: str) -> None:
    model = DrainingSTT()
    model.aclose = AsyncMock(wraps=model.aclose)
    frame = rtc.AudioFrame.create(16000, 1, 320)
    async with aclosing(AMDRacingSTT(model, APIConnectOptions(max_retry=0))) as racing:
        if started != "no_audio":
            racing.push_audio(frame)
            stream = model.streams[0]
            stream.aclose = AsyncMock(wraps=stream.aclose)
            stream.push_frame = Mock(wraps=stream.push_frame)
            if started in {"reader_running", "amd_selected"}:
                await wait_for_transcript(lambda: bool(stream.frames))
            if started == "amd_selected":
                racing.push_transcript("Hello from AMD.", source="amd")

        await racing.aclose()
        racing.push_audio(frame)
        await racing.aclose()
        racing.push_audio(frame)

        if started != "no_audio":
            stream.aclose.assert_awaited_once()
            stream.push_frame.assert_not_called()
            assert stream._task.done()
            assert stream._metrics_task.done()
        assert len(model.streams) == (0 if started == "no_audio" else 1)
    model.aclose.assert_not_awaited()
    await model.aclose()


@pytest.mark.asyncio
async def test_turns_commit_immediately_without_flushing_stt(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.flush = Mock(wraps=stream.flush)
    empty = racing.end_turn("")
    assert empty.transcript == ""
    assert empty.source is None

    stream.send_fake_transcript("Hello.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Hello."
    )
    turn = racing.end_turn("Hello from session.")
    assert turn.transcript == "Hello."
    assert turn.source == "amd"
    stream.flush.assert_not_called()
    assert not stream.input_ended.is_set()


@pytest.mark.asyncio
async def test_late_final_after_empty_turns_belongs_to_the_next_turn(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    earlier = [racing.end_turn("") for _ in range(3)]
    stream.send_fake_transcript("Hello.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Hello."
    )
    following = racing.end_turn("")
    assert following.transcript == "Hello."
    assert following.source == "amd"
    assert all(turn.transcript == "" and turn.source is None for turn in earlier)


@pytest.mark.asyncio
async def test_final_segments_accumulate_and_ignore_interim_transcript(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("partial", is_final=False)
    stream.send_fake_transcript("")
    stream.send_fake_transcript("Please leave")
    stream.send_fake_transcript("a message.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Please leave a message."
    )
    turn = racing.end_turn("Leave a message.")
    assert turn.transcript == "Please leave a message."
    assert turn.source == "amd"


@pytest.mark.asyncio
async def test_final_after_commit_belongs_to_next_turn(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("Hello.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Hello."
    )
    first = racing.end_turn("Hello.")
    racing.push_transcript("Yes.")
    stream.send_fake_transcript("Late words.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Late words."
    )
    second = racing.end_turn("Yes.")
    assert first.transcript == "Hello."
    assert second.transcript == "Late words."
    assert second.source == "amd"


@pytest.mark.asyncio
async def test_empty_selected_source_commits_without_switching_to_session(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("Please leave")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Please leave"
    )
    first = racing.end_turn("Please leave a message.")
    assert first.transcript == "Please leave"
    second = racing.end_turn("at the tone.")
    assert second.transcript == ""

    stream.send_fake_transcript("a message at the tone.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "a message at the tone."
    )
    third = racing.end_turn("Goodbye.")
    assert third.transcript == "a message at the tone."
    assert third.source == "amd"
    assert second.transcript == ""


@pytest.mark.asyncio
async def test_selected_amd_failure_uses_session_transcript_only_for_the_open_turn(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("Hello.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Hello."
    )
    first = racing.end_turn("Hello from session.")
    second = racing.end_turn("Session turn two.")
    racing.push_transcript("Session turn three.")

    stream.error = RuntimeError("connection lost")
    racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not racing.amd_stt_active)
    third = racing.end_turn("")
    assert first.transcript == "Hello."
    assert first.source == "amd"
    assert second.transcript == ""
    assert third.transcript == "Session turn three."
    assert third.source == "session"


@pytest.mark.asyncio
async def test_push_failure_uses_session_transcript_without_splicing_sources(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("Please leave")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Please leave"
    )
    racing.push_transcript("Please leave a message.")
    stream.push_frame = Mock(side_effect=RuntimeError("failed to push"))
    racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    turn = racing.end_turn("Please leave a message.")
    assert turn.transcript == "Please leave a message."
    assert turn.source == "session"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["push", "read"])
async def test_failure_uses_session_even_when_its_transcript_is_empty(
    racing_stt: tuple[AMDRacingSTT, DrainingStream], failure: str
) -> None:
    racing, stream = racing_stt
    stream.send_fake_transcript("Please leave a message.")
    await wait_for_transcript(
        lambda: racing._current.snapshot(racing._source).transcript == "Please leave a message."
    )
    if failure == "push":
        stream.push_frame = Mock(side_effect=RuntimeError("failed to push"))
    else:
        stream.error = RuntimeError("connection lost")
    racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not racing.amd_stt_active)
    turn = racing.end_turn("")
    assert turn.transcript == ""
    assert turn.source == "session"

    following = racing.end_turn("Session transcript.")
    assert following.transcript == "Session transcript."
    assert following.source == "session"


@pytest.mark.asyncio
async def test_failure_before_any_transcript_falls_back_to_session(
    racing_stt: tuple[AMDRacingSTT, DrainingStream],
) -> None:
    racing, stream = racing_stt
    stream.error = RuntimeError("connection lost")
    racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not racing.amd_stt_active)
    empty = racing.end_turn("")
    assert empty.transcript == ""
    assert empty.source == "session"

    turn = racing.end_turn("Hello.")
    assert turn.transcript == "Hello."
    assert turn.source == "session"


@pytest.mark.asyncio
async def test_stream_open_failure_keeps_session_transcript() -> None:
    model = DrainingSTT()
    model.stream = Mock(side_effect=RuntimeError("failed to open"))
    async with aclosing(AMDRacingSTT(model, APIConnectOptions(max_retry=0))) as racing:
        racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        racing.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        model.stream.assert_called_once()
        turn = racing.end_turn("Hello.")
        assert turn.transcript == "Hello."
        assert turn.source == "session"
    await model.aclose()
