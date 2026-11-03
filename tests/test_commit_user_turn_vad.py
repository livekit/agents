from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import AgentSession, stt, utils, vad

from .fake_vad import FakeVAD
from .test_agent_stt_node import _FlushableSTT, _TestAgent

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000


def _speech_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(_SAMPLE_RATE, 1, 160)


def _vad_event(event_type: vad.VADEventType, **kwargs: float) -> vad.VADEvent:
    defaults: dict[str, float] = {
        "samples_index": 0,
        "timestamp": 0.0,
        "speech_duration": 0.0,
        "silence_duration": 0.0,
    }
    defaults.update(kwargs)
    return vad.VADEvent(type=event_type, **defaults)


def _final_event(text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text=text)],
    )


def _interim_event(text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text=text)],
    )


async def _start_session(*, manual_flush: bool, use_vad: bool):
    stt_impl = _FlushableSTT(manual_flush=manual_flush)
    session = AgentSession(
        stt=stt_impl,
        vad=FakeVAD() if use_vad else None,
        turn_handling={"turn_detection": "manual"},
        session_close_transcript_timeout=0.0,
    )
    await session.start(_TestAgent())
    stream = await asyncio.wait_for(stt_impl.streams.get(), 5)
    recognition = session._activity._audio_recognition
    return session, recognition, stream


async def _drain(stream, count: int) -> None:
    for _ in range(count):
        await asyncio.wait_for(stream.inputs.get(), 5)


async def test_vad_end_before_final_reuses_cached_transcript_without_silence() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_vad_event(_vad_event(vad.VADEventType.START_OF_SPEECH))
        await recognition._on_vad_event(_vad_event(vad.VADEventType.END_OF_SPEECH))
        await recognition._on_stt_event(_final_event("hello"))
        recognition._last_final_transcript_time -= 0.6
        recognition._last_speaking_time -= 0.6

        silence = utils.audio.silence_frame(0.01, _SAMPLE_RATE)
        for _ in range(70):
            recognition._push_audio(silence)
        await _drain(stream, 70)
        assert stream.inputs.qsize() == 0

        loop = asyncio.get_running_loop()
        start = loop.time()
        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=10.0, stt_flush_duration=0.2, skip_reply=True
        )
        transcript = await asyncio.wait_for(commit, 1.5)
        elapsed = loop.time() - start

        assert transcript == "hello"
        assert elapsed < 1.5
        assert stream.inputs.qsize() == 0
    finally:
        await session.aclose()


async def test_vad_end_before_final_without_trailing_silence() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_vad_event(_vad_event(vad.VADEventType.START_OF_SPEECH))
        await recognition._on_vad_event(_vad_event(vad.VADEventType.END_OF_SPEECH))
        await recognition._on_stt_event(_final_event("hello"))
        recognition._last_final_transcript_time -= 0.6
        recognition._last_speaking_time -= 0.6

        loop = asyncio.get_running_loop()
        start = loop.time()
        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=10.0, stt_flush_duration=0.2, skip_reply=True
        )
        transcript = await asyncio.wait_for(commit, 1.5)
        elapsed = loop.time() - start

        assert transcript == "hello"
        assert elapsed < 1.5
    finally:
        await session.aclose()


async def test_vad_end_after_final_flushes_and_waits_for_new_final() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_stt_event(_final_event("hello"))
        recognition._last_final_transcript_time -= 1.0

        await recognition._on_vad_event(_vad_event(vad.VADEventType.START_OF_SPEECH))
        await recognition._on_vad_event(_vad_event(vad.VADEventType.END_OF_SPEECH))

        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=5.0, stt_flush_duration=0.2, skip_reply=True
        )

        flushed_silence = await asyncio.wait_for(stream.inputs.get(), 5)
        assert isinstance(flushed_silence, rtc.AudioFrame)
        assert not commit.done()

        await recognition._on_stt_event(_final_event("again"))
        transcript = await asyncio.wait_for(commit, 5)

        assert transcript == "hello again"
    finally:
        await session.aclose()


async def test_no_final_yet_resolves_shortly_after_final_arrives() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        loop = asyncio.get_running_loop()
        start = loop.time()
        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=5.0, stt_flush_duration=0.2, skip_reply=True
        )

        async def _inject_final_later() -> None:
            await asyncio.sleep(0.3)
            await recognition._on_stt_event(_final_event("hello"))

        injector = asyncio.create_task(_inject_final_later())

        try:
            transcript = await asyncio.wait_for(commit, 0.5)
        finally:
            await injector
        elapsed = loop.time() - start

        assert transcript == "hello"
        assert elapsed < 0.5
    finally:
        await session.aclose()


async def test_pending_interim_waits_for_timeout_and_appends() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_stt_event(_final_event("hello"))
        recognition._last_final_transcript_time -= 1.0
        await recognition._on_stt_event(_interim_event("wor"))

        loop = asyncio.get_running_loop()
        start = loop.time()
        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=0.2, stt_flush_duration=0.05, skip_reply=True
        )
        transcript = await asyncio.wait_for(commit, 5)
        elapsed = loop.time() - start

        assert transcript == "hello wor"
        assert elapsed >= 0.2
    finally:
        await session.aclose()


async def test_vad_none_keeps_legacy_behaviour_and_waits_for_timeout() -> None:
    session, recognition, stream = await _start_session(manual_flush=False, use_vad=False)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_stt_event(_final_event("hello"))
        recognition._last_final_transcript_time -= 1.0

        loop = asyncio.get_running_loop()
        start = loop.time()
        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=0.2, stt_flush_duration=0.05, skip_reply=True
        )

        flushed_silence = await asyncio.wait_for(stream.inputs.get(), 5)
        assert isinstance(flushed_silence, rtc.AudioFrame)

        transcript = await asyncio.wait_for(commit, 5)
        elapsed = loop.time() - start

        assert transcript == "hello"
        assert elapsed >= 0.2
    finally:
        await session.aclose()


async def test_manual_flush_stt_still_flushes_with_complete_cached_final() -> None:
    session, recognition, stream = await _start_session(manual_flush=True, use_vad=True)
    try:
        recognition._push_audio(_speech_frame())
        await _drain(stream, 1)

        await recognition._on_vad_event(_vad_event(vad.VADEventType.START_OF_SPEECH))
        await recognition._on_vad_event(_vad_event(vad.VADEventType.END_OF_SPEECH))
        await recognition._on_stt_event(_final_event("hello"))

        commit = recognition._commit_user_turn(
            audio_detached=True, transcript_timeout=5.0, stt_flush_duration=0.2, skip_reply=True
        )

        assert isinstance(
            await asyncio.wait_for(stream.inputs.get(), 5), stt.RecognizeStream._FlushSentinel
        )
        assert not commit.done()

        await recognition._on_stt_event(_final_event("hello again"))
        transcript = await asyncio.wait_for(commit, 5)

        assert transcript == "hello hello again"
    finally:
        await session.aclose()
