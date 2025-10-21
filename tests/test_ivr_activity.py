import asyncio
from typing import Any

import pytest

from livekit.agents import Agent
from livekit.agents.voice.ivr.ivr_activity import TfidfLoopDetector

from .fake_session import FakeActions, create_session, run_session

SESSION_TIMEOUT = 10.0


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


@pytest.mark.asyncio
async def test_tfidf_no_loop_on_unique_user_speech() -> None:
    """Does not emit loop events when all user speeches are unique."""
    speed = 10.0
    actions = FakeActions()
    silence_duration = 0.1
    audio_duration = 1.5

    transcripts = [
        "Welcome to automated phone system",
        "Type 1 for sales",
        "Type 2 for support",
        "Type 3 for billing",
        "Type 4 for technical support",
        "Type 5 for account management",
        "Type 6 for billing",
        "Type 7 for user management",
    ]

    t = 0.5
    for transcript in transcripts:
        actions.add_user_speech(t, t + audio_duration, transcript)
        t += audio_duration + silence_duration

    session = create_session(actions, speed_factor=speed)
    detector = TfidfLoopDetector(session=session)

    loop_detected_count = 0

    def on_loop_detected(_: Any) -> None:
        nonlocal loop_detected_count
        loop_detected_count += 1

    detector.on("loop_detected", on_loop_detected)

    await asyncio.wait_for(detector.start(), timeout=10.0)
    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)
    assert loop_detected_count == 0

    await detector.aclose()


@pytest.mark.asyncio
async def test_tfidf_detects_loop_on_repeated_user_speech() -> None:
    """Emits loop events when user speeches repeat across turns."""
    speed = 10.0
    actions = FakeActions()
    repeat_count = 2
    silence_duration = 0.1
    audio_duration = 1.5

    transcripts_to_repeat = [
        "Welcome to automated phone system",
        "Type 1 for sales",
        "Type 2 for support",
        "Type 3 for billing",
        "Type 4 for technical support",
    ]

    t = 0.5
    for _ in range(repeat_count):
        for transcript in transcripts_to_repeat:
            actions.add_user_speech(t, t + audio_duration, transcript)
            t += audio_duration + silence_duration

    session = create_session(actions, speed_factor=speed)
    detector = TfidfLoopDetector(session=session)

    loop_detected_count = 0

    def on_loop_detected(_: Any) -> None:
        nonlocal loop_detected_count
        loop_detected_count += 1

    detector.on("loop_detected", on_loop_detected)

    await asyncio.wait_for(detector.start(), timeout=10.0)
    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)
    assert loop_detected_count == 3  # loop detected from the second "Type 2 for ..."

    await detector.aclose()


@pytest.mark.asyncio
async def test_tfidf_detects_loop_after_interleaved_unique_and_repeated() -> None:
    """Resets on unique speech then detects loops again when repeats resume."""
    speed = 10.0
    actions = FakeActions()
    repeat_count = 2
    silence_duration = 0.1
    audio_duration = 1.5

    transcripts_to_repeat = [
        "Welcome to automated phone system",
        "Type 1 for sales",
        "Type 2 for support",
        "Type 3 for billing",
    ]

    t = 0.5
    for _ in range(repeat_count):
        for transcript in transcripts_to_repeat:
            actions.add_user_speech(t, t + audio_duration, transcript)
            t += audio_duration + silence_duration

    # consecutive count of similar chunks will reset to 0 here
    actions.add_user_speech(t, t + audio_duration, "This is a non-repeated user speech")
    t += audio_duration + silence_duration

    # re-increment consecutive count of similar chunks here
    for transcript in transcripts_to_repeat:
        actions.add_user_speech(t, t + audio_duration, transcript)
        t += audio_duration + silence_duration

    session = create_session(actions, speed_factor=speed)
    detector = TfidfLoopDetector(session=session)

    loop_detected_count = 0

    def on_loop_detected(_: Any) -> None:
        nonlocal loop_detected_count
        loop_detected_count += 1

    detector.on("loop_detected", on_loop_detected)

    await asyncio.wait_for(detector.start(), timeout=10.0)
    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)
    assert loop_detected_count == 4

    await detector.aclose()


@pytest.mark.asyncio
async def test_tfidf_detects_loop_with_minor_text_variations() -> None:
    """Treats minor textual variations as similar enough to trigger a loop."""
    speed = 10.0
    actions = FakeActions()
    silence_duration = 0.1
    audio_duration = 1.5

    transcripts = [
        "Welcome to automated phone system",
        "Type 1 for sales, type 2 for support, type 3 for billing",
        "Again, type 1 for sales, type 2 for support, type 3 for billing",
        "And again, type 1 for sales, type 2 for support, type 3 for billing",
        "Repeat, type 1 for sales, type 2 for support, and type 3 for billing",
    ]

    t = 0.5
    for transcript in transcripts:
        actions.add_user_speech(t, t + audio_duration, transcript)
        t += audio_duration + silence_duration

    session = create_session(actions, speed_factor=speed)
    detector = TfidfLoopDetector(session=session)

    loop_detected_count = 0

    def on_loop_detected(_: Any) -> None:
        nonlocal loop_detected_count
        loop_detected_count += 1

    detector.on("loop_detected", on_loop_detected)

    await asyncio.wait_for(detector.start(), timeout=10.0)
    await asyncio.wait_for(run_session(session, MyAgent()), timeout=SESSION_TIMEOUT)
    assert loop_detected_count == 1

    await detector.aclose()
