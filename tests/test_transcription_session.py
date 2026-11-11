from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, APIConnectionError, LanguageCode, NotGivenOr
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, TimedString
from livekit.agents.utils.audio import AudioBuffer
from livekit.agents.utils.audio_gate import AdaptiveNoiseGate
from livekit.agents.voice.transcription_session import TranscriptionSession, TranscriptSegment
from livekit.agents.voice.transcription_session.activity import ActivityTimeline, TrackActivity
from livekit.agents.voice.transcription_session.attribution import _Attributor
from livekit.agents.voice.transcription_session.mixer import _Mixer, _TrackInput

pytestmark = pytest.mark.unit

_SR = 16000
_BLOCK = 0.02
_QUIET = 0.002
_LOUD = 0.3


def _tone(duration: float, level: float, *, sample_rate: int = _SR) -> list[rtc.AudioFrame]:
    """20 ms frames of a 220 Hz tone at ``level`` RMS."""
    n = int(sample_rate * _BLOCK)
    t = np.arange(int(sample_rate * duration)) / sample_rate
    samples = (np.sin(2 * np.pi * 220 * t) * level * np.sqrt(2) * 32767).astype(np.int16)
    return [
        rtc.AudioFrame(
            data=samples[i : i + n].tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=n,
        )
        for i in range(0, samples.size - n + 1, n)
    ]


def _track(*parts: tuple[float, float], sample_rate: int = _SR) -> list[rtc.AudioFrame]:
    return [f for duration, level in parts for f in _tone(duration, level, sample_rate=sample_rate)]


def _word(text: str, start: float, end: float, speaker_id: str | None = None) -> TimedString:
    return TimedString(text, start_time=start, end_time=end, speaker_id=speaker_id)


def _final(text: str, words: list[TimedString] | None = None, **kwargs: object) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language=LanguageCode("en"), text=text, words=words, **kwargs)],  # type: ignore[arg-type]
    )


def _timeline(blocks: list[dict[str, tuple[float, bool]]]) -> ActivityTimeline:
    timeline = ActivityTimeline(block_duration=_BLOCK, retention=60.0)
    for block in blocks:
        timeline.append({k: TrackActivity(rms, open) for k, (rms, open) in block.items()})
    return timeline


class _ScriptedSTT(STT):
    """Emits each event once the session's audio reaches ``at`` seconds.

    Session time is a stream's ``start_time_offset`` plus the audio it has been sent, as a real
    plugin places its timestamps. The script is shared by every stream the session opens.
    """

    def __init__(
        self,
        script: list[tuple[float, SpeechEvent]],
        *,
        late: list[tuple[float, SpeechEvent]] | None = None,
        fail_at: list[float] | None = None,
    ) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self.script = sorted(script, key=lambda item: item[0])
        self.late = late or []
        """Events the first stream returns this many seconds after its input ends."""
        self.speech_start = False
        """Each stream reports speech starting on its first frame, before anything else."""
        self.fail_at = fail_at or []
        """Session times at which the stream hearing them fails for good."""
        self.pushed: list[rtc.AudioFrame] = []
        self.offsets: list[float] = []
        """Each stream's ``start_time_offset``, once its first frame was read."""

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return _ScriptedStream(stt=self, conn_options=conn_options)


class _ScriptedStream(RecognizeStream):
    async def _run(self) -> None:
        stt = self._stt
        assert isinstance(stt, _ScriptedSTT)
        pending = stt.script
        heard = 0.0
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                if not heard:
                    stt.offsets.append(self.start_time_offset)
                    if stt.speech_start:
                        self._event_ch.send_nowait(
                            SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                        )
                stt.pushed.append(data)
                heard += data.duration
            # an event past what this stream heard belongs to a later stream
            while pending and pending[0][0] <= self.start_time_offset + heard + 1e-9:
                self._event_ch.send_nowait(pending.pop(0)[1])
            if stt.fail_at and self.start_time_offset + heard >= stt.fail_at[0]:
                stt.fail_at = stt.fail_at[1:]
                raise APIConnectionError("scripted failure", retryable=False)
        late, stt.late = stt.late, []
        for delay, ev in late:
            await asyncio.sleep(delay)
            self._event_ch.send_nowait(ev)


async def _paced(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
    """The frames in real time, as a room delivers them."""
    started = time.monotonic()
    for i, frame in enumerate(frames):
        if (wait := started + i * _BLOCK - time.monotonic()) > 0:
            await asyncio.sleep(wait)
        yield frame


async def _run_session(
    stt: _ScriptedSTT, tracks: dict[str, list[rtc.AudioFrame]]
) -> list[TranscriptSegment]:
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()
    # frame by frame across participants, in real time as a room delivers them
    started = time.monotonic()
    for i in range(max(len(frames) for frames in tracks.values())):
        if (wait := started + i * _BLOCK - time.monotonic()) > 0:
            await asyncio.sleep(wait)
        for identity, frames in tracks.items():
            if i < len(frames):
                session.push_audio(identity, frames[i])
    await session.aclose()
    return segments


# -- timeline -------------------------------------------------------------------------------


def test_timeline_scores_only_open_blocks() -> None:
    timeline = _timeline(
        [
            {"a": (0.3, True), "b": (0.01, False)},
            {"a": (0.3, True), "b": (0.2, True)},
            {"a": (0.01, False), "b": (0.2, True)},
        ]
    )
    assert timeline.scores(0.0, 0.02) == {"a": 0.3}
    assert timeline.scores(0.0, 0.06) == pytest.approx({"a": 0.6, "b": 0.4})
    # a span shorter than a block still covers the block it starts in
    assert timeline.scores(0.045, 0.045) == {"b": 0.2}


def test_timeline_knows_which_track_a_span_was_spoken_on() -> None:
    timeline = ActivityTimeline(block_duration=_BLOCK, retention=60.0)
    for sid in ["TR_old"] * 10 + ["TR_new"] * 10:
        timeline.append({"a": TrackActivity(0.3, True, track_sid=sid)})
    assert timeline.track_sid("a", 0.0, 0.2) == "TR_old"
    assert timeline.track_sid("a", 0.2, 0.4) == "TR_new"
    assert timeline.track_sid("b", 0.0, 0.4) is None


def test_timeline_forgets_past_retention() -> None:
    timeline = ActivityTimeline(block_duration=_BLOCK, retention=0.1)
    for _ in range(10):
        timeline.append({"a": TrackActivity(0.3, True)})
    assert timeline.end_index == 10
    assert timeline.scores(0.0, 0.1) == {}
    assert timeline.scores(0.1, 0.2) == pytest.approx({"a": 1.5})


# -- attribution ----------------------------------------------------------------------------


def _turns() -> ActivityTimeline:
    # a speaks for the first second, b for the second
    return _timeline(
        [{"a": (0.3, True), "b": (0.0, False)}] * 50 + [{"a": (0.0, False), "b": (0.3, True)}] * 50
    )


def test_single_speaker_keeps_the_whole_transcript() -> None:
    attributor = _Attributor(_turns(), overlap_ratio=0.5)
    alt = _final("hello there.", start_time=0.1, end_time=0.9).alternatives[0]
    [run] = attributor.attribute(alt, fallback=(0.0, 2.0))
    assert (run.identity, run.text, run.method, run.overlap) == (
        "a",
        "hello there.",
        "single",
        False,
    )


def test_words_split_a_turn_change_inside_one_utterance() -> None:
    attributor = _Attributor(_turns(), overlap_ratio=0.5)
    words = [
        _word("hi", 0.2, 0.4),
        _word("bob.", 0.5, 0.8),
        _word("hey", 1.1, 1.3),
        _word("alice!", 1.4, 1.8),
    ]
    alt = _final("Hi Bob. Hey Alice!", words).alternatives[0]
    runs = attributor.attribute(alt, fallback=(0.0, 2.0))
    assert [(r.identity, r.text, r.method) for r in runs] == [
        ("a", "Hi Bob.", "word"),
        ("b", "Hey Alice!", "word"),
    ]
    assert runs[1].start_time == pytest.approx(1.1)


def test_word_nobody_was_audible_for_joins_its_neighbour() -> None:
    timeline = _timeline([{"a": (0.3, True)}] * 25 + [{}] * 50 + [{"b": (0.3, True)}] * 25)
    attributor = _Attributor(timeline, overlap_ratio=0.5)
    words = [_word("one", 0.1, 0.3), _word("two", 0.9, 1.0), _word("three", 1.6, 1.9)]
    runs = attributor.attribute(_final("one two three", words).alternatives[0], fallback=(0, 2))
    assert [(r.identity, r.text) for r in runs] == [("a", "one two"), ("b", "three")]


def test_utterance_without_word_timings_goes_to_the_louder_speaker() -> None:
    timeline = _timeline([{"a": (0.3, True), "b": (0.2, True)}] * 50)
    attributor = _Attributor(timeline, overlap_ratio=0.5)
    alt = _final("talking over each other", start_time=0.0, end_time=1.0).alternatives[0]
    [run] = attributor.attribute(alt, fallback=(0.0, 1.0))
    assert (run.identity, run.method, run.overlap) == ("a", "utterance", True)


def test_no_timings_uses_the_fallback_span() -> None:
    attributor = _Attributor(_turns(), overlap_ratio=0.5)
    [run] = attributor.attribute(_final("late words").alternatives[0], fallback=(1.0, 2.0))
    assert (run.identity, run.method) == ("b", "single")


def test_nobody_audible_is_unattributed() -> None:
    attributor = _Attributor(_timeline([{}] * 50), overlap_ratio=0.5)
    alt = _final("noise", start_time=0.0, end_time=1.0).alternatives[0]
    [run] = attributor.attribute(alt, fallback=(0.0, 1.0))
    assert (run.identity, run.method) == (None, "none")


def test_diarization_resolves_overlap_once_learned() -> None:
    # 0-1s a alone, 1-2s b alone, 2-3s both equally loud
    timeline = _timeline(
        [{"a": (0.3, True)}] * 50
        + [{"b": (0.3, True)}] * 50
        + [{"a": (0.3, True), "b": (0.3, True)}] * 50
    )
    attributor = _Attributor(timeline, overlap_ratio=0.5)
    learn = [_word(f"a{i}", 0.1 + i * 0.2, 0.2 + i * 0.2, "S0") for i in range(4)] + [
        _word(f"b{i}", 1.1 + i * 0.2, 1.2 + i * 0.2, "S1") for i in range(4)
    ]
    attributor.attribute(_final(" ".join(learn), learn).alternatives[0], fallback=(0, 2))

    overlapped = [_word("mine", 2.1, 2.3, "S1"), _word("no", 2.5, 2.7, "S0")]
    runs = attributor.attribute(_final("mine no", overlapped).alternatives[0], fallback=(2, 3))
    assert [(r.identity, r.text, r.method) for r in runs] == [
        ("b", "mine", "diarization"),
        ("a", "no", "diarization"),
    ]


def test_diarization_never_overrides_a_clear_energy_margin() -> None:
    attributor = _Attributor(_turns(), overlap_ratio=0.5)
    # the diarizer calls everything S0, which is learned as a; b's clear turn stays b
    learn = [_word(f"a{i}", 0.1 + i * 0.2, 0.2 + i * 0.2, "S0") for i in range(4)]
    attributor.attribute(_final(" ".join(learn), learn).alternatives[0], fallback=(0, 1))
    words = [_word("a", 0.5, 0.7, "S0"), _word("b", 1.5, 1.7, "S0")]
    runs = attributor.attribute(_final("a b", words).alternatives[0], fallback=(0, 2))
    assert [r.identity for r in runs] == ["a", "b"]


# -- mixer ----------------------------------------------------------------------------------


def _mixer() -> tuple[_Mixer, ActivityTimeline]:
    timeline = ActivityTimeline(block_duration=_BLOCK, retention=60.0)
    mixer = _Mixer(
        sample_rate=_SR,
        block_size=int(_SR * _BLOCK),
        timeline=timeline,
        max_backlog=5,
        backlog_window=50,
    )
    return mixer, timeline


def _input(identity: str) -> _TrackInput:
    return _TrackInput(
        identity,
        track_sid=None,
        gate=AdaptiveNoiseGate(),
        sample_rate=_SR,
        block_size=int(_SR * _BLOCK),
        max_blocks=1000,
    )


def _rms(frame: rtc.AudioFrame) -> float:
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768
    return float(np.sqrt(np.mean(samples**2)))


def test_mixer_records_speech_but_not_room_tone() -> None:
    mixer, timeline = _mixer()
    inp = _input("a")
    mixer.add(inp)
    for frame in _track((1.0, 0.05), (0.2, 0.3)):
        inp.push(frame)

    out = [mixer.mix() for _ in range(60)]
    # mixed at full level throughout; steady room tone never opens the gate, speech does
    assert _rms(out[40]) == pytest.approx(0.05, rel=0.05)
    assert _rms(out[55]) == pytest.approx(0.3, rel=0.05)
    assert timeline.end_index == 60
    assert not timeline.scores(0.0, 1.0)
    assert "a" in timeline.scores(1.0, 1.02)


def test_mixer_sums_participants_on_one_clock() -> None:
    mixer, timeline = _mixer()
    a, b = _input("a"), _input("b")
    mixer.add(a)
    mixer.add(b)
    for frame in _track((0.6, _QUIET), (0.4, _LOUD)):
        a.push(frame)
    for frame in _track((0.6, _QUIET), (0.4, _LOUD)):
        b.push(frame)
    out = [mixer.mix() for _ in range(50)]
    assert _rms(out[-1]) == pytest.approx(2 * _LOUD, rel=0.05)
    assert timeline.scores(0.8, 1.0).keys() == {"a", "b"}


def test_underrun_is_silence_and_leaves_the_clock_alone() -> None:
    mixer, timeline = _mixer()
    a, b = _input("a"), _input("b")
    mixer.add(a)
    mixer.add(b)
    for frame in _track((0.6, _QUIET), (0.4, _LOUD)):
        a.push(frame)
    for _ in range(50):
        mixer.mix()
    assert timeline.end_index == 50
    assert timeline.scores(0.8, 1.0).keys() == {"a"}


def test_mixer_drops_a_backlog_that_never_drains() -> None:
    mixer, _ = _mixer()
    a, b = _input("a"), _input("b")
    mixer.add(a)
    mixer.add(b)
    # b came in 12 blocks late and keeps sending in real time: it never catches up by itself
    for frame in _track((0.24, _QUIET)):
        b.push(frame)
    block = _tone(_BLOCK, _QUIET)[0]
    for _ in range(49):
        a.push(block)
        b.push(block)
        mixer.mix()
    assert b.queued == 12
    a.push(block)
    b.push(block)
    mixer.mix()
    assert b.queued == 4


def test_mixer_keeps_long_frames_that_drain() -> None:
    # 200 ms at a time queues 10 blocks at once, but all of it is mixed
    mixer, timeline = _mixer()
    a, b = _input("a"), _input("b")
    mixer.add(a)
    mixer.add(b)
    block = _tone(_BLOCK, _QUIET)[0]
    for i in range(100):
        if i % 10 == 0:
            for frame in _tone(0.2, _QUIET):
                b.push(frame)
        a.push(block)
        mixer.mix()
    assert b.queued == 0
    assert all("b" in activity for activity in timeline._blocks)


def test_track_input_resamples_and_downmixes() -> None:
    inp = _input("a")
    stereo = rtc.AudioFrame(
        data=np.full(480 * 2 * 10, 1000, dtype=np.int16).tobytes(),
        sample_rate=48000,
        num_channels=2,
        samples_per_channel=480 * 10,
    )
    inp.push(stereo)
    blocks = []
    while (block := inp.pop()) is not None:
        blocks.append(block)
    assert all(block.size == 320 for block in blocks)
    assert 3 <= len(blocks) <= 5  # 100 ms at 16 kHz, less what the resampler holds back


# -- session --------------------------------------------------------------------------------


async def test_session_attributes_each_participant() -> None:
    # 0.6 s of room tone each, then a talks for 0.6 s, then b does
    words = [
        _word("good", 0.70, 0.90),
        _word("morning", 0.95, 1.20),
        _word("hi", 1.30, 1.50),
        _word("there", 1.55, 1.80),
    ]
    stt = _ScriptedSTT([(1.9, _final("Good morning. Hi there.", words))])
    segments = await _run_session(
        stt,
        {
            "alice": _track((0.6, _QUIET), (0.6, _LOUD), (0.7, _QUIET)),
            "bob": _track((0.6, _QUIET), (0.6, _QUIET), (0.7, _LOUD)),
        },
    )
    assert [(s.participant_identity, s.text, s.final) for s in segments] == [
        ("alice", "Good morning.", True),
        ("bob", "Hi there.", True),
    ]
    assert segments[0].id != segments[1].id


async def test_session_attributes_a_participant_who_joins_mid_sentence() -> None:
    # no quiet stretch first: the gate must not learn the speech as its floor
    stt = _ScriptedSTT([(1.1, _final("already talking", start_time=0.1, end_time=0.9))])
    segments = await _run_session(stt, {"alice": _track((1.0, _LOUD), (0.2, _QUIET))})
    assert [(s.participant_identity, s.text) for s in segments] == [("alice", "already talking")]


async def test_session_closes_an_interim_whose_speaker_changes() -> None:
    interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[
            SpeechData(language=LanguageCode("en"), text="hello", words=[_word("hello", 0.7, 0.9)])
        ],
    )
    # the final places the same word in bob's turn
    final = _final("hello", [_word("hello", 1.3, 1.5)])
    stt = _ScriptedSTT([(1.6, interim), (1.8, final)])
    segments = await _run_session(
        stt,
        {
            "alice": _track((0.6, _QUIET), (0.6, _LOUD), (0.7, _QUIET)),
            "bob": _track((0.6, _QUIET), (0.6, _QUIET), (0.7, _LOUD)),
        },
    )
    assert [(s.participant_identity, s.text, s.final) for s in segments] == [
        ("alice", "hello", False),
        ("alice", "", True),
        ("bob", "hello", True),
    ]
    assert segments[0].id == segments[1].id != segments[2].id


async def test_session_closes_interim_segments_a_final_drops() -> None:
    interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[
            SpeechData(
                language=LanguageCode("en"),
                text="hello hi",
                words=[_word("hello", 0.7, 0.9), _word("hi", 1.3, 1.5)],
            )
        ],
    )
    final = _final("hello", [_word("hello", 0.7, 0.9)])
    stt = _ScriptedSTT([(1.6, interim), (1.8, final)])
    segments = await _run_session(
        stt,
        {
            "alice": _track((0.6, _QUIET), (0.6, _LOUD), (0.7, _QUIET)),
            "bob": _track((0.6, _QUIET), (0.6, _QUIET), (0.7, _LOUD)),
        },
    )
    interims = [s for s in segments if not s.final]
    assert [(s.participant_identity, s.text) for s in interims] == [
        ("alice", "hello"),
        ("bob", "hi"),
    ]
    finals = {s.id: s for s in segments if s.final}
    assert finals[interims[0].id].text == "hello"
    assert finals[interims[1].id].text == ""


async def test_session_leaves_unheard_speech_unattributed() -> None:
    stt = _ScriptedSTT([(0.5, _final("phantom", start_time=0.1, end_time=0.4))])
    [segment] = await _run_session(stt, {"alice": _track((1.0, _QUIET))})
    assert (segment.participant_identity, segment.attribution) == (None, "none")


async def test_session_mixes_everything_it_was_sent_before_closing() -> None:
    stt = _ScriptedSTT([])
    await _run_session(stt, {"alice": _track((1.0, _QUIET)), "bob": _track((0.5, _QUIET))})
    assert sum(f.duration for f in stt.pushed) == pytest.approx(1.0, abs=0.021)


async def test_session_follows_an_input_stream_until_it_ends() -> None:
    stt = _ScriptedSTT([(1.3, _final("from a stream", start_time=0.7, end_time=1.2))])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()

    track = _track((0.6, _QUIET), (0.6, _LOUD), (0.2, _QUIET))
    session.add_input("carol", _paced(track), track_sid="TR_carol")
    while "carol" in session.participants:
        await asyncio.sleep(0.01)
    await session.aclose()

    assert [(s.participant_identity, s.track_sid) for s in segments] == [("carol", "TR_carol")]
    assert session.history == segments


async def test_session_reopens_the_stt_on_the_mix_clock_after_the_room_empties() -> None:
    stt = _ScriptedSTT([(1.3, _final("back again", start_time=1.1, end_time=1.3))])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()

    session.add_input("alice", _paced(_track((0.6, _QUIET), (0.2, _LOUD))))
    while session.participants:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # the room stays empty a while; the wall clock runs on

    session.add_input("bob", _paced(_track((0.2, _QUIET), (0.4, _LOUD))))
    while session.participants:
        await asyncio.sleep(0.01)
    await session.aclose()

    # the second stream picks up where the mix's clock stopped, not where the wall clock is
    assert len(stt.offsets) == 2
    assert stt.offsets[0] == pytest.approx(0.0, abs=0.025)
    assert stt.offsets[1] == pytest.approx(0.8, abs=0.025)
    assert [(s.participant_identity, s.text) for s in segments] == [("bob", "back again")]


async def test_session_sends_a_burst_in_real_time_so_the_stream_dates_it_right() -> None:
    # a second of audio arrives at once and the session closes straight away
    stt = _ScriptedSTT([(1.0, _final("caught up", start_time=0.65, end_time=0.95))])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()
    for frame in _track((0.6, _QUIET), (0.4, _LOUD)):
        session.push_audio("alice", frame)
    started = time.monotonic()
    await session.aclose()

    assert time.monotonic() - started >= 0.9
    assert stt.offsets == [pytest.approx(0.0, abs=0.025)]
    assert [(s.participant_identity, s.text) for s in segments] == [("alice", "caught up")]


async def test_late_untimed_final_stays_within_its_own_stream() -> None:
    # alice's stream ends when she leaves; its untimed final arrives while bob is talking
    stt = _ScriptedSTT([], late=[(0.5, _final("from alice"))])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()

    session.add_input("alice", _paced(_track((0.6, _QUIET), (0.4, 0.1))))
    while session.participants:
        await asyncio.sleep(0.01)
    session.add_input("bob", _paced(_track((0.2, _QUIET), (0.8, 0.6))))
    while session.participants:
        await asyncio.sleep(0.01)
    await session.aclose()

    assert [(s.participant_identity, s.text) for s in segments] == [("alice", "from alice")]


async def test_session_recovers_from_a_failed_stt_stream() -> None:
    interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[
            SpeechData(language=LanguageCode("en"), text="hello", words=[_word("hello", 0.7, 0.9)])
        ],
    )
    final = _final("again", [_word("again", 1.5, 1.7)])
    stt = _ScriptedSTT([(1.0, interim), (1.8, final)], fail_at=[1.1])
    segments = await _run_session(
        stt, {"alice": _track((0.6, _QUIET), (1.4, _LOUD), (0.2, _QUIET))}
    )

    # the failed stream's open interim stands, and a new stream carries on at once
    assert len(stt.offsets) == 2
    assert sum(f.duration for f in stt.pushed) == pytest.approx(2.2, abs=0.005)
    assert [(s.participant_identity, s.text, s.final) for s in segments] == [
        ("alice", "hello", False),
        ("alice", "hello", True),
        ("alice", "again", True),
    ]


async def test_an_stt_that_keeps_failing_is_reopened_less_often() -> None:
    # every stream fails on its first block
    stt = _ScriptedSTT([], fail_at=[i * 0.02 + 0.01 for i in range(100)])
    stt.speech_start = True  # which isn't a sign the STT works
    await _run_session(stt, {"alice": _track((1.0, _QUIET))})
    # once at once, then a second later: nowhere near a stream per block
    assert len(stt.offsets) <= 3


def test_utterance_speaker_label_is_learned_when_words_have_none() -> None:
    timeline = _timeline([{"a": (0.3, True)}] * 50 + [{"a": (0.3, True), "b": (0.3, True)}] * 50)
    attributor = _Attributor(timeline, overlap_ratio=0.5)
    for i in range(3):
        words = [_word("hi", 0.1 + i * 0.3, 0.3 + i * 0.3)]
        attributor.attribute(_final("hi", words, speaker_id="S0").alternatives[0], fallback=(0, 1))
    [run] = attributor.attribute(
        _final("mine", start_time=1.2, end_time=1.8, speaker_id="S0").alternatives[0],
        fallback=(1, 2),
    )
    assert (run.identity, run.method) == ("a", "diarization")


async def test_diarization_labels_are_learned_per_stream() -> None:
    learn = [_word(f"a{i}", 0.45 + i * 0.15, 0.55 + i * 0.15, "S0") for i in range(4)]
    # the next connection's S0 is someone else; bob and carol are equally loud
    overlapped = _final("hi", [_word("hi", 1.5, 1.7, "S0")])
    stt = _ScriptedSTT([(1.15, _final(" ".join(map(str, learn)), learn)), (1.9, overlapped)])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    segments: list[TranscriptSegment] = []
    session.on("transcript_segment", segments.append)
    await session.start()

    session.add_input("alice", _paced(_track((0.4, _QUIET), (0.8, _LOUD))))
    while session.participants:
        await asyncio.sleep(0.01)
    session.add_input("bob", _paced(_track((0.2, _QUIET), (0.6, _LOUD))))
    session.add_input("carol", _paced(_track((0.2, _QUIET), (0.6, _LOUD))))
    while session.participants:
        await asyncio.sleep(0.01)
    await session.aclose()

    assert len(stt.offsets) == 2
    [hi] = [s for s in segments if s.text == "hi"]
    assert hi.participant_identity in {"bob", "carol"} and hi.attribution == "word"


async def test_session_mixes_what_a_departing_participant_already_sent() -> None:
    stt = _ScriptedSTT([])
    session = TranscriptionSession(stt=stt, publish_transcriptions=False)
    await session.start()

    async def live(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
        for frame in frames:
            yield frame
        await asyncio.Event().wait()  # a track stays open until it is unsubscribed

    session.add_input("alice", live(_track((0.5, _LOUD))))
    await asyncio.sleep(0.05)
    session._on_participant_disconnected(SimpleNamespace(identity="alice"))  # type: ignore[arg-type]
    while session.participants:
        await asyncio.sleep(0.01)
    await session.aclose()

    assert sum(f.duration for f in stt.pushed) == pytest.approx(0.5, abs=0.021)


class _FakeWriter:
    def __init__(
        self, sent: list[tuple[str, dict[str, str], str]], topic: str, attrs: dict[str, str]
    ) -> None:
        self._sent, self._topic, self._attrs = sent, topic, attrs

    async def write(self, text: str) -> None:
        self._sent.append((self._topic, self._attrs, text))

    async def aclose(self) -> None:
        pass


class _FakeLocalParticipant:
    identity = "agent"

    def __init__(self) -> None:
        self.streams: list[tuple[str, dict[str, str], str]] = []
        self.legacy: list[rtc.Transcription] = []

    async def stream_text(
        self, *, topic: str, attributes: dict[str, str], **_: object
    ) -> _FakeWriter:
        return _FakeWriter(self.streams, topic, attributes)

    async def publish_transcription(self, transcription: rtc.Transcription) -> None:
        self.legacy.append(transcription)


class _FakeRemote:
    def __init__(self, client_protocol: int) -> None:
        self.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
        self.attributes: dict[str, str] = {}
        self._info = type("Info", (), {"client_protocol": client_protocol})()


class _FakeRoom:
    def __init__(self, client_protocol: int) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.remote_participants = {"viewer": _FakeRemote(client_protocol)}

    def isconnected(self) -> bool:
        return True


@pytest.mark.parametrize("client_protocol, legacy", [(0, True), (10, False)])
async def test_publish_adds_legacy_packets_only_for_old_clients(
    client_protocol: int, legacy: bool
) -> None:
    from livekit.agents.utils import aio

    session = TranscriptionSession(stt=_ScriptedSTT([]))
    room = _FakeRoom(client_protocol)
    ch = aio.Chan[TranscriptSegment]()
    segment = TranscriptSegment(
        id="SG_1",
        participant_identity="alice",
        track_sid="TR_a",
        text="hello",
        final=True,
        start_time=0.0,
        end_time=1.0,
        language=LanguageCode("en"),
        words=None,
        confidence=1.0,
        overlap=False,
        attribution="single",
    )
    ch.send_nowait(segment)
    ch.close()
    await session._publish_task(room, ch)  # type: ignore[arg-type]

    [(topic, attrs, text)] = room.local_participant.streams
    assert (topic, text) == ("lk.transcription", "hello")
    assert attrs["lk.transcribed_track_id"] == "TR_a"
    assert attrs["lk.segment_id"] == "SG_1"
    assert len(room.local_participant.legacy) == (1 if legacy else 0)
    if legacy:
        packet = room.local_participant.legacy[0]
        assert (packet.participant_identity, packet.track_sid) == ("alice", "TR_a")


def test_non_streaming_stt_is_rejected() -> None:
    stt = _ScriptedSTT([])
    stt._capabilities = STTCapabilities(streaming=False, interim_results=False)
    with pytest.raises(ValueError):
        TranscriptionSession(stt=stt)
