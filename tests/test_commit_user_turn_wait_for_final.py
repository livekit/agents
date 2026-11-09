from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, Agent, AgentSession, stt, vad
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit

_SPEECH = rtc.AudioFrame(
    data=b"\x01\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
)
_ENDPOINTING_DELAY = 0.3


def _final(text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text=text)],
    )


def _interim(text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en", text=text)],
    )


def _vad_event(type: vad.VADEventType, frames: list[rtc.AudioFrame] | None = None) -> vad.VADEvent:
    return vad.VADEvent(
        type=type,
        samples_index=0,
        timestamp=0,
        speech_duration=0,
        silence_duration=0,
        frames=frames or [],
    )


@dataclass
class _Script:
    on_audio: str | None = None
    """Final emitted for the first audio frame."""
    on_end: str | None = None
    """Final emitted once the input ends, like a provider that finalizes on end of input."""
    end_delay: float = 0.0
    hang: bool = False
    """Never finish after the input ends, like a provider that stopped answering."""


class _ScriptedStream(stt.RecognizeStream):
    def __init__(self, stt_impl: stt.STT, conn_options: APIConnectOptions, script: _Script):
        super().__init__(stt=stt_impl, conn_options=conn_options)
        self._script = script
        self.flushed = asyncio.Event()

    async def _run(self) -> None:
        on_audio = self._script.on_audio
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                self.flushed.set()
            elif on_audio is not None:
                self._event_ch.send_nowait(_final(on_audio))
                on_audio = None
        await asyncio.sleep(self._script.end_delay)
        if self._script.on_end is not None:
            self._event_ch.send_nowait(_final(self._script.on_end))
        if self._script.hang:
            await asyncio.Event().wait()


class _ScriptedSTT(FakeSTT):
    def __init__(self, *scripts: _Script, manual_flush: bool) -> None:
        super().__init__()
        self.capabilities.manual_flush = manual_flush
        self._scripts = iter(scripts)
        self.streams: asyncio.Queue[_ScriptedStream] = asyncio.Queue()

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        stream = _ScriptedStream(self, conn_options, next(self._scripts, _Script()))
        self.streams.put_nowait(stream)
        return stream


class _SilenceGatedVAD(vad.VAD):
    """Ends a segment only after enough silence, never when its input ends (like Silero)."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _SilenceGatedVADStream(self)


class _SilenceGatedVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        speech: list[rtc.AudioFrame] = []
        silence = 0.0
        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                speech, silence = [], 0.0
            elif any(frame.data):
                if not speech:
                    self._event_ch.send_nowait(_vad_event(vad.VADEventType.START_OF_SPEECH))
                speech.append(frame)
                silence = 0.0
            elif speech:
                silence += frame.duration
                if silence >= 0.5:
                    self._event_ch.send_nowait(_vad_event(vad.VADEventType.END_OF_SPEECH, speech))
                    speech = []


async def _start(
    stt_impl: stt.STT, vad_impl: vad.VAD | None = None
) -> tuple[AgentSession, asyncio.Queue[_EndOfTurnInfo]]:
    session = AgentSession(
        stt=stt_impl,
        vad=vad_impl or FakeVAD(),
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": _ENDPOINTING_DELAY},
        },
        session_close_transcript_timeout=0.0,
    )
    await session.start(Agent(instructions="test"))
    activity = session._activity
    assert activity is not None

    delivered: asyncio.Queue[_EndOfTurnInfo] = asyncio.Queue()
    on_end_of_turn = activity.on_end_of_turn

    def _record(info: _EndOfTurnInfo) -> bool:
        delivered.put_nowait(info)
        return on_end_of_turn(info)

    activity.on_end_of_turn = _record  # type: ignore[method-assign]
    return session, delivered


async def _delivered_once(delivered: asyncio.Queue[_EndOfTurnInfo]) -> tuple[str, bool]:
    info = await asyncio.wait_for(delivered.get(), 5)
    await asyncio.sleep(_ENDPOINTING_DELAY * 2)
    assert delivered.empty()
    return info.new_transcript, info.skip_reply


@pytest.mark.parametrize("final_arrives", [True, False])
async def test_flushed_commit_never_promotes_an_interim(final_arrives: bool) -> None:
    stt_impl = _ScriptedSTT(manual_flush=True)
    session, delivered = await _start(stt_impl)
    try:
        stream = await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(_interim("hel"))

        commit = session.commit_user_turn(
            transcript_timeout=0.1, skip_reply=True, wait_for_final=True
        )
        await asyncio.wait_for(stream.flushed.wait(), 5)
        if final_arrives:
            await recognition._on_stt_event(_final("hello"))

        assert await asyncio.wait_for(commit, 5) == ("hello" if final_arrives else "")
        if final_arrives:
            assert await _delivered_once(delivered) == ("hello", True)
        assert recognition._audio_interim_transcript == ""
    finally:
        await session.aclose()


async def test_unflushable_commit_drains_the_stream_before_the_next_turn() -> None:
    # the next stream finalizes immediately, which must not join the committed turn
    stt_impl = _ScriptedSTT(
        _Script(on_end="hello there"), _Script(on_audio="next turn"), manual_flush=False
    )
    session, delivered = await _start(stt_impl)
    try:
        first = await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        recognition._push_audio(_SPEECH)

        commit = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        assert await asyncio.wait_for(commit, 5) == "hello there"
        second = await asyncio.wait_for(stt_impl.streams.get(), 5)
        assert second is not first
        recognition._push_audio(_SPEECH)

        info = await asyncio.wait_for(delivered.get(), 5)
        assert (info.new_transcript, info.skip_reply) == ("hello there", True)
        # the next stream is consumed once the committed turn is delivered
        info = await asyncio.wait_for(delivered.get(), 5)
        assert (info.new_transcript, info.skip_reply) == ("next turn", False)
    finally:
        await session.aclose()


async def test_vad_segmented_stt_finalizes_an_open_segment() -> None:
    # a non-streaming STT runs behind a StreamAdapter, which recognizes VAD segments
    stt_impl = FakeSTT(fake_transcript="hello")
    stt_impl.capabilities.streaming = False
    session, delivered = await _start(stt_impl, _SilenceGatedVAD())
    try:
        recognition = session._activity._audio_recognition
        for _ in range(5):
            recognition._push_audio(_SPEECH)

        commit = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        assert await asyncio.wait_for(commit, 5) == "hello"
        assert await _delivered_once(delivered) == ("hello", True)
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    "event", [vad.VADEventType.START_OF_SPEECH, vad.VADEventType.END_OF_SPEECH]
)
async def test_automatic_turn_detection_cannot_take_over_the_commit(
    event: vad.VADEventType,
) -> None:
    stt_impl = _ScriptedSTT(manual_flush=True)
    session, delivered = await _start(stt_impl)
    try:
        stream = await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition

        commit = session.commit_user_turn(
            transcript_timeout=1.0, skip_reply=True, wait_for_final=True
        )
        await asyncio.wait_for(stream.flushed.wait(), 5)
        await recognition._on_stt_event(_final("hello"))
        assert await asyncio.wait_for(commit, 5) == "hello"

        # the committed turn is still in its endpointing delay
        await recognition._on_vad_event(_vad_event(event))

        assert await _delivered_once(delivered) == ("hello", True)
    finally:
        await session.aclose()


async def test_pending_automatic_turn_does_not_split_the_commit() -> None:
    stt_impl = _ScriptedSTT(
        _Script(on_audio="part one", on_end="part two", end_delay=_ENDPOINTING_DELAY * 2),
        manual_flush=False,
    )
    session, delivered = await _start(stt_impl)
    try:
        await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        recognition._push_audio(_SPEECH)
        await asyncio.wait_for(recognition._final_transcript_received.wait(), 5)
        # an automatic end of turn is now waiting out its endpointing delay
        await recognition._on_vad_event(_vad_event(vad.VADEventType.END_OF_SPEECH))

        commit = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        assert await asyncio.wait_for(commit, 5) == "part one part two"
        assert await _delivered_once(delivered) == ("part one part two", True)
    finally:
        await session.aclose()


async def test_a_newer_commit_keeps_the_drained_tail() -> None:
    stt_impl = _ScriptedSTT(_Script(on_end="tail", end_delay=0.3), manual_flush=False)
    session, delivered = await _start(stt_impl)
    try:
        await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        recognition._push_audio(_SPEECH)

        superseded = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        await asyncio.sleep(0.1)
        commit = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        assert await asyncio.wait_for(commit, 5) == "tail"
        assert superseded.cancelled()
        assert await _delivered_once(delivered) == ("tail", True)
    finally:
        await session.aclose()


async def test_clearing_during_the_commit_releases_the_drained_stream() -> None:
    stt_impl = _ScriptedSTT(_Script(on_audio="hello", hang=True), manual_flush=False)
    session, delivered = await _start(stt_impl)
    try:
        await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        drained = recognition._stt_pipeline
        recognition._push_audio(_SPEECH)
        await asyncio.wait_for(recognition._final_transcript_received.wait(), 5)

        commit = session.commit_user_turn(
            transcript_timeout=0.1, skip_reply=True, wait_for_final=True
        )
        assert await asyncio.wait_for(commit, 5) == "hello"
        assert recognition._final_commit_pending

        session.clear_user_turn()
        await asyncio.wait([drained._pump_task], timeout=1)
        assert drained._pump_task.done()
        assert delivered.empty()
    finally:
        await session.aclose()


async def test_clearing_a_chain_of_commits_returns_instead_of_raising() -> None:
    stt_impl = _ScriptedSTT(_Script(hang=True), _Script(hang=True), manual_flush=False)
    session, delivered = await _start(stt_impl)
    try:
        await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        recognition._push_audio(_SPEECH)

        superseded = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        await asyncio.sleep(0.05)
        commit = session.commit_user_turn(
            transcript_timeout=2.0, skip_reply=True, wait_for_final=True
        )
        await asyncio.sleep(0.05)
        session.clear_user_turn()

        assert await asyncio.wait_for(commit, 5) == ""
        assert superseded.cancelled()
        assert delivered.empty()
    finally:
        await session.aclose()
