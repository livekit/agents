"""Transcript attribution and failure handling in AMDTranscriber, without a session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.amd._chat_context import AMDTranscript
from livekit.agents.voice.amd._stt import AMDTranscriber

from .fake_stt import DrainingStream, DrainingSTT

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.fixture
async def transcriber() -> AsyncIterator[tuple[AMDTranscriber, DrainingStream]]:
    model = DrainingSTT()
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as transcriber:
        transcriber.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        yield transcriber, model.streams[0]
    await model.aclose()


async def wait_for_transcript(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


async def amd_heard(transcriber: AMDTranscriber, stream: DrainingStream, text: str) -> None:
    before = transcriber._current.get("amd")
    stream.send_fake_transcript(text)
    await wait_for_transcript(lambda: transcriber._current.get("amd") != before)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_first", [True, False])
async def test_each_turn_keeps_both_transcripts_whichever_arrives_first(
    transcriber: tuple[AMDTranscriber, DrainingStream], session_first: bool
) -> None:
    amd, stream = transcriber
    if session_first:
        amd.push_transcript("session transcript")
    await amd_heard(amd, stream, "amd transcript")
    first = amd.end_turn("session transcript")
    assert first == AMDTranscript(session="session transcript", amd="amd transcript")

    await amd_heard(amd, stream, "amd next turn")
    second = amd.end_turn("session next turn")
    assert second == AMDTranscript(session="session next turn", amd="amd next turn")
    assert amd.amd_stt_active


@pytest.mark.asyncio
async def test_a_late_amd_transcript_leaves_the_session_transcript(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Hi, you've reached Sam.")
    first = amd.end_turn("Hi, you've reached Sam.")
    second = amd.end_turn("Leave your name and number.")
    assert first.amd == "Hi, you've reached Sam."
    assert second == AMDTranscript(session="Leave your name and number.", amd="")
    assert second.transcript == "Leave your name and number."

    # the late words belong to the next turn
    await amd_heard(amd, stream, "Leave your name and number.")
    assert amd.end_turn("").amd == "Leave your name and number."


@pytest.mark.asyncio
async def test_a_final_within_the_wait_belongs_to_the_turn_just_ended(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    ended = amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    assert ended == AMDTranscript(session="Sam Lee speaking.", amd="")
    late = amd.late_final
    assert late is not None
    waiting = asyncio.create_task(late.wait())
    await asyncio.sleep(0.1)
    stream.send_fake_transcript("Sam Lee speaking.")
    assert await waiting == "Sam Lee speaking."
    # the next turn starts clean
    await amd_heard(amd, stream, "Yes, that works.")
    assert amd.end_turn("Yes, that works.").amd == "Yes, that works."


@pytest.mark.asyncio
async def test_a_final_after_the_wait_belongs_to_the_next_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    late = amd.late_final
    assert late is not None
    assert await late.wait() is None
    await amd_heard(amd, stream, "Sam Lee speaking.")
    assert amd.end_turn("Yes.").amd == "Sam Lee speaking."


@pytest.mark.asyncio
async def test_a_final_before_the_wait_starts_still_belongs_to_the_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    late = amd.late_final
    assert late is not None
    stream.send_fake_transcript("Sam Lee speaking.")
    await wait_for_transcript(lambda: amd.late_final is None)
    assert await late.wait() == "Sam Lee speaking."
    assert amd.end_turn("").amd == ""


@pytest.mark.asyncio
async def test_without_session_stt_a_final_after_the_wait_belongs_to_the_next_turn() -> None:
    model = DrainingSTT()
    async with aclosing(
        AMDTranscriber(model, APIConnectOptions(max_retry=0), session_stt=False)
    ) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.end_turn("Sam Lee speaking.", late_wait=0.3)
        late = amd.late_final
        assert late is not None
        assert await late.wait() is None
        await amd_heard(amd, model.streams[0], "Sam Lee speaking.")
        assert amd.end_turn("").amd == "Sam Lee speaking."
    await model.aclose()


@pytest.mark.asyncio
async def test_a_turn_with_its_amd_final_needs_no_wait(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Sam Lee speaking.")
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    assert amd.late_final is None


def test_the_classifier_sees_both_transcripts_only_with_an_amd_stt() -> None:
    assert AMDTranscript(session="Hi.").for_classifier() == "Hi."
    assert AMDTranscript(session=None, amd="Hi.").for_classifier() == "Hi."
    assert (
        AMDTranscript(session="Hi.", amd="Hi.").for_classifier()
        == '{"transcript_1": "Hi.", "transcript_2": "Hi."}'
    )
    assert (
        AMDTranscript(session="Hi.", amd="").for_classifier()
        == '{"transcript_1": "Hi.", "transcript_2": ""}'
    )
    assert AMDTranscript(session="", amd="Hi.").transcript == "Hi."


async def test_without_session_stt_amd_uses_its_own_transcript_until_it_fails() -> None:
    model = DrainingSTT()
    async with aclosing(
        AMDTranscriber(model, APIConnectOptions(max_retry=0), session_stt=False)
    ) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.push_transcript("Session transcript.")
        assert amd.end_turn("Session transcript.") == AMDTranscript(session=None, amd="")

        await amd_heard(amd, model.streams[0], "AMD transcript.")
        assert amd.end_turn("Session transcript.") == AMDTranscript(
            session=None, amd="AMD transcript."
        )

        amd.push_transcript("Fallback transcript.")
        amd.fail(RuntimeError("connection lost"))
        assert amd.end_turn("") == AMDTranscript(session="Fallback transcript.", amd=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", ["no_audio", "reader_pending", "reader_running", "amd_heard"])
async def test_close_stops_stream_and_prevents_more_audio(started: str) -> None:
    model = DrainingSTT()
    model.aclose = AsyncMock(wraps=model.aclose)
    frame = rtc.AudioFrame.create(16000, 1, 320)
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as amd:
        if started != "no_audio":
            amd.push_audio(frame)
            stream = model.streams[0]
            stream.aclose = AsyncMock(wraps=stream.aclose)
            stream.push_frame = Mock(wraps=stream.push_frame)
            if started in {"reader_running", "amd_heard"}:
                await wait_for_transcript(lambda: bool(stream.frames))
            if started == "amd_heard":
                amd.push_transcript("Hello from AMD.", source="amd")

        await amd.aclose()
        amd.push_audio(frame)
        await amd.aclose()
        amd.push_audio(frame)

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
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    stream.flush = Mock(wraps=stream.flush)
    assert amd.end_turn("") == AMDTranscript(session="", amd="")

    await amd_heard(amd, stream, "Hello.")
    assert amd.end_turn("Hello from session.") == AMDTranscript(
        session="Hello from session.", amd="Hello."
    )
    stream.flush.assert_not_called()
    assert not stream.input_ended.is_set()


@pytest.mark.asyncio
async def test_late_final_after_empty_turns_belongs_to_the_next_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    earlier = [amd.end_turn("") for _ in range(3)]
    await amd_heard(amd, stream, "Hello.")
    assert amd.end_turn("").amd == "Hello."
    assert all(turn.transcript == "" for turn in earlier)


@pytest.mark.asyncio
async def test_final_segments_accumulate_and_ignore_interim_transcript(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    stream.send_fake_transcript("partial", is_final=False)
    stream.send_fake_transcript("")
    stream.send_fake_transcript("Please leave")
    stream.send_fake_transcript("a message.")
    await wait_for_transcript(lambda: amd._current.get("amd") == "Please leave a message.")
    assert amd.end_turn("Leave a message.") == AMDTranscript(
        session="Leave a message.", amd="Please leave a message."
    )


@pytest.mark.asyncio
async def test_amd_failure_leaves_the_session_transcript_for_the_open_turn_on(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Hello.")
    first = amd.end_turn("Hello from session.")
    amd.push_transcript("Session turn two.")

    stream.error = RuntimeError("connection lost")
    amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not amd.amd_stt_active)
    second = amd.end_turn("")
    assert first == AMDTranscript(session="Hello from session.", amd="Hello.")
    assert second == AMDTranscript(session="Session turn two.", amd=None)
    assert amd.end_turn("Session turn three.").amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["push", "read"])
async def test_failure_drops_the_amd_transcript_even_when_the_session_is_empty(
    transcriber: tuple[AMDTranscriber, DrainingStream], failure: str
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Please leave a message.")
    if failure == "push":
        stream.push_frame = Mock(side_effect=RuntimeError("failed to push"))
    else:
        stream.error = RuntimeError("connection lost")
    amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not amd.amd_stt_active)
    assert amd.end_turn("") == AMDTranscript(session="", amd=None)
    assert amd.end_turn("Session transcript.") == AMDTranscript(
        session="Session transcript.", amd=None
    )


@pytest.mark.asyncio
async def test_stream_open_failure_keeps_session_transcript() -> None:
    model = DrainingSTT()
    model.stream = Mock(side_effect=RuntimeError("failed to open"))
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        model.stream.assert_called_once()
        assert amd.end_turn("Hello.") == AMDTranscript(session="Hello.", amd=None)
    await model.aclose()
