"""Diagnostic rig for transcript/audio divergence under barge-in + pause/resume.

Reproduces the production wiring of a recorded session:

    RecorderAudioOutput -> _SyncedAudioOutput (TranscriptSynchronizer) -> FakeAudioOutput(can_pause)

TTS audio is amplitude-tagged per LLM reply so both the "live" playout (bottom
sink) and the recorder-reconstructed audio (what agent insights plays back) can
be attributed to the reply whose text they carry.

The repro tests mirror staging sim run SR_4UpFo6cNC7wA (project p_4m98r6497vn,
room RM_fqrSJAUMWfV4, job AJ_hzD7X2EpqsQc): the chat context commits the agent
reply "I still need the security code for the card before I can finalize this."
with a ~5s speaking window, while the audio actually heard in that window is a
STALE earlier reply ("Got it, so that's Monday, July fourteenth to Thursday,
July seventeenth.") and the committed reply's own audio never follows.

Mechanism of the original bug (all timing realistic):

  1. The caller barges into reply A -> the false-interruption handler pauses the
     shared audio output mid-reply (``agent_activity._interrupt_by_audio_activity``).
  2. The caller's STT FINAL lands later than ``false_interruption_timeout`` (2s
     default) after VAD end-of-speech - routine for real providers under load.
  3. The false-interruption timer fired first and RESUMED the stale reply: the
     tail of reply A played out loud even though the caller already finished
     their real next turn. Each interim transcript re-paused and re-armed the
     timer, so the stale tail kept bleeding out in bursts until the final
     transcript arrived.
  4. The final landed -> reply B was generated and committed to the chat
     context. On the wall-clock timeline the caller heard stale reply-A audio
     exactly where the transcript records their turn and reply B.

The fix keeps the paused reply paused once the barge-in shows evidence of real
speech (VAD segment longer than
``interruption.false_interruption_max_speech_duration``, or any non-empty
transcript), bounding the pause with an interrupt-watchdog instead of ever
resuming stale audio over the user's words.

The scenario tests assert only on the public surface: the committed chat
context (``agent.chat_ctx``), session events, and the audio frames that
reached the sink / the recorder. A separate whitebox section pins the
pause-ownership invariant of the shared audio output.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.tts import tts as _tts_mod
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.recorder_io import RecorderIO
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_session import FakeActions
from .fake_stt import FakeSTT
from .fake_tts import FakeSynthesizeStream, FakeTTS, FakeTTSResponse
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


# ---------------------------------------------------------------------------
# amplitude-tagged TTS
# ---------------------------------------------------------------------------


class TaggedSynthesizeStream(FakeSynthesizeStream):
    async def _run(self, output_emitter: _tts_mod.AudioEmitter) -> None:
        self._attempt += 1
        assert isinstance(self._tts, TaggedFakeTTS)

        output_emitter.initialize(
            request_id=utils.shortuuid("fake_tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )

        input_text = ""
        async for data in self._input_ch:
            if isinstance(data, str):
                input_text += data
                continue
            elif isinstance(data, FakeSynthesizeStream._FlushSentinel) and not input_text:
                continue

            start_time = time.perf_counter()
            self._mark_started()
            if not (resp := self._tts.fake_response_map.get(input_text)):
                resp = FakeTTSResponse(input=input_text, audio_duration=0.0, ttfb=0.0, duration=0.0)

            amp = self._tts.amp_for(input_text)
            input_text = ""
            if resp.audio_duration == 0.0:
                continue

            if resp.ttfb > 0.0:
                await asyncio.sleep(resp.ttfb - (time.perf_counter() - start_time))

            output_emitter.start_segment(segment_id=utils.shortuuid("fake_segment_"))

            sample = struct.pack("<h", amp)
            pushed_samples = 0
            max_samples = (
                int(self._tts.sample_rate * resp.audio_duration + 0.5) * self._tts.num_channels
            )
            while pushed_samples < max_samples:
                num_samples = min(self._tts.sample_rate // 100, max_samples - pushed_samples)
                output_emitter.push(sample * num_samples)
                pushed_samples += num_samples
                await asyncio.sleep(0)

            delay = resp.duration - (time.perf_counter() - start_time)
            if delay > 0.0:
                await asyncio.sleep(delay)

            output_emitter.flush()


class TaggedFakeTTS(FakeTTS):
    """FakeTTS emitting a distinct, constant int16 amplitude per response text."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._amp_map: dict[str, int] = {}
        for i, text in enumerate(self._fake_response_map):
            self._amp_map[text] = (i + 1) * 1000

    def amp_for(self, text: str) -> int:
        if text not in self._amp_map:
            self._amp_map[text] = (len(self._amp_map) + 1) * 1000
        return self._amp_map[text]

    def label_for_amp(self, amp: int) -> str:
        for text, a in self._amp_map.items():
            if a == amp:
                return text
        return f"<unknown amp {amp}>"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TaggedSynthesizeStream:
        stream = TaggedSynthesizeStream(tts=self, conn_options=conn_options)
        self._stream_ch.send_nowait(stream)
        return stream


def _amps_of(frames: list[rtc.AudioFrame]) -> dict[int, float]:
    """Return {amplitude: seconds} for non-silent samples in the frames."""
    out: dict[int, float] = {}
    for f in frames:
        if f.samples_per_channel == 0:
            continue
        data = memoryview(f.data).cast("B").cast("h")
        # constant-amplitude frames: inspect first non-zero sample
        amp = 0
        for s in data[:: max(1, len(data) // 8)]:
            if s != 0:
                amp = s
                break
        out[amp] = out.get(amp, 0.0) + f.duration
    return out


# ---------------------------------------------------------------------------
# capture-only recorder
# ---------------------------------------------------------------------------


@dataclass
class RecordedWrite:
    wall_time: float
    amps: dict[int, float]  # amplitude -> seconds of audio in this write


class CapturingRecorderIO(RecorderIO):
    """RecorderIO that skips the ogg encoder and keeps written chunks in memory."""

    def __init__(self, *, agent_session: AgentSession) -> None:
        super().__init__(agent_session=agent_session)
        self.writes: list[RecordedWrite] = []
        self._started = True  # `recording` gate

    def _write_cb(self, buf: list[rtc.AudioFrame]) -> None:  # type: ignore[override]
        # bypass input pairing/encoding; keep agent-channel content only
        self.writes.append(RecordedWrite(wall_time=time.time(), amps=_amps_of(buf)))

    async def aclose(self) -> None:  # type: ignore[override]
        self._started = False


# ---------------------------------------------------------------------------
# instrumented bottom sink
# ---------------------------------------------------------------------------


@dataclass
class LiveSegment:
    frames: list[rtc.AudioFrame] = field(default_factory=list)
    played: float | None = None
    interrupted: bool | None = None
    finished_at: float | None = None

    def played_amps(self, tts: TaggedFakeTTS) -> dict[str, float]:
        if self.played is None:
            return {}
        out: dict[str, float] = {}
        remaining = self.played
        for f in self.frames:
            if remaining <= 1e-9:
                break
            d = min(f.duration, remaining)
            for amp, dur in _amps_of([f]).items():
                label = tts.label_for_amp(amp)
                out[label] = out.get(label, 0.0) + min(dur, d)
            remaining -= d
        return out


class InstrumentedFakeAudioOutput(FakeAudioOutput):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.segments: list[LiveSegment] = []
        self._open_segment: LiveSegment | None = None
        self.events: list[tuple[float, str, Any]] = []

        def _on_finished(ev: Any) -> None:
            if self._open_segment is not None:
                self._open_segment.played = ev.playback_position
                self._open_segment.interrupted = ev.interrupted
                self._open_segment.finished_at = time.time()
                self._open_segment = None
            self.events.append(
                (time.time(), "playback_finished", (ev.playback_position, ev.interrupted))
            )

        self.on("playback_finished", _on_finished)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self._open_segment is None:
            self._open_segment = LiveSegment()
            self.segments.append(self._open_segment)
        self._open_segment.frames.append(frame)
        await super().capture_frame(frame)

    def pause(self) -> None:
        self.events.append((time.time(), "pause", None))
        super().pause()

    def resume(self) -> None:
        self.events.append((time.time(), "resume", None))
        super().resume()

    def clear_buffer(self) -> None:
        self.events.append((time.time(), "clear_buffer", None))
        super().clear_buffer()


# ---------------------------------------------------------------------------
# session rig
# ---------------------------------------------------------------------------


@dataclass
class Rig:
    session: AgentSession
    tts: TaggedFakeTTS
    stt: FakeSTT
    audio_input: FakeAudioInput
    bottom: InstrumentedFakeAudioOutput
    recorder: CapturingRecorderIO
    transcript_sync: TranscriptSynchronizer
    timeline: list[tuple[float, str, Any]] = field(default_factory=list)

    def log(self, kind: str, payload: Any) -> None:
        self.timeline.append((time.time(), kind, payload))


def create_rig(actions: FakeActions, *, speed: float = 1.0) -> Rig:
    user_speeches = actions.get_user_speeches(speed_factor=speed)
    llm_responses = actions.get_llm_responses(speed_factor=speed)
    tts_responses = actions.get_tts_responses(speed_factor=speed)

    stt = FakeSTT(fake_user_speeches=user_speeches)
    tts = TaggedFakeTTS(fake_responses=tts_responses)

    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5 / speed,
            min_speech_duration=0.05 / speed,
        ),
        stt=stt,
        llm=FakeLLM(fake_responses=llm_responses),
        tts=tts,
        turn_handling={
            "turn_detection": None,
            "endpointing": {"min_delay": 0.5 / speed, "max_delay": 6.0 / speed},
            "interruption": {
                "min_duration": 0.5 / speed,
                "false_interruption_timeout": 2.0 / speed,
            },
        },
        aec_warmup_duration=None,
    )

    audio_input = FakeAudioInput()
    bottom = InstrumentedFakeAudioOutput(can_pause=True)
    transcription_output = FakeTextOutput()

    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=bottom,
        next_in_chain_text=transcription_output,
        speed=speed,
    )

    session.input.audio = audio_input
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output

    recorder = CapturingRecorderIO(agent_session=session)
    session.input.audio = recorder.record_input(session.input.audio)
    session.output.audio = recorder.record_output(session.output.audio)

    rig = Rig(
        session=session,
        tts=tts,
        stt=stt,
        audio_input=audio_input,
        bottom=bottom,
        recorder=recorder,
        transcript_sync=transcript_sync,
    )

    session.on("agent_state_changed", lambda ev: rig.log("agent_state", ev.new_state))
    session.on(
        "speech_created",
        lambda ev: rig.log("speech_created", (ev.speech_handle.id, ev.source)),
    )
    session.on(
        "conversation_item_added",
        lambda ev: rig.log(
            "item_added",
            (
                ev.item.role if ev.item.type == "message" else ev.item.type,
                getattr(ev.item, "text_content", None),
                getattr(ev.item, "interrupted", None),
            ),
        ),
    )
    session.on(
        "agent_false_interruption",
        lambda ev: rig.log("false_interruption", ev.resumed),
    )
    return rig


async def run_rig(rig: Rig, agent: Agent, *, drain_delay: float = 6.0) -> float:
    await rig.session.start(agent)
    t0 = time.time()
    rig.audio_input.push(0.1)
    await rig.stt.fake_user_speeches_done
    await asyncio.sleep(drain_delay)
    with contextlib.suppress(RuntimeError):
        await rig.session.drain()
    await rig.session.aclose()
    await rig.transcript_sync.aclose()
    return t0


def dump(rig: Rig, agent: Agent, t0: float, title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")
    evs = sorted(
        rig.timeline + [(t, f"audio.{k}", p) for (t, k, p) in rig.bottom.events],
        key=lambda e: e[0],
    )
    for t, kind, payload in evs:
        print(f"  {t - t0:7.2f}s  {kind:<22} {payload}")

    print("\n  --- live playout (bottom sink) ---")
    for i, seg in enumerate(rig.bottom.segments):
        pushed = sum(f.duration for f in seg.frames)
        print(
            f"  seg{i}: pushed={pushed:.2f}s played={seg.played} interrupted={seg.interrupted} "
            f"finished_at={None if seg.finished_at is None else round(seg.finished_at - t0, 2)}"
        )
        for label, dur in seg.played_amps(rig.tts).items():
            print(f"        heard {dur:5.2f}s of: {label[:70]!r}")

    print("\n  --- recorder writes (insights audio) ---")
    for w in rig.recorder.writes:
        for amp, dur in w.amps.items():
            label = rig.tts.label_for_amp(amp) if amp else "<silence>"
            print(f"  at {w.wall_time - t0:6.2f}s: {dur:5.2f}s of {label[:70]!r}")

    print("\n  --- committed chat ---")
    for item in agent.chat_ctx.items:
        if item.type == "message" and item.role in ("user", "assistant"):
            print(
                f"  [{item.created_at - t0:6.2f}s] {item.role}: {item.text_content!r} "
                f"(interrupted={getattr(item, 'interrupted', None)})"
            )


# ---------------------------------------------------------------------------
# shared helpers / scenario constants
# ---------------------------------------------------------------------------

# kept for reuse by sibling behavioral tests (test_stale_resume_behavior.py)
GARDEN = "I do have garden view rooms available in our queen two bed category for two twenty."
PHONE = "I have that as one two three, five five five, zero one seven zero. Is that right?"
CONFIRM = "Great, your booking is confirmed for tomorrow."

# texts mirroring run SR_4UpFo6cNC7wA
REPLY_A = (
    "Got it, so that's Monday, July fourteenth to Thursday, July seventeenth. "
    "Do you have a smoking preference?"
)
USER_FINAL = "Great. That works for me. Please go ahead and finalize the booking."
REPLY_B = "I still need the security code for the card before I can finalize this."


class HotelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="hotel receptionist")


def _live_heard(rig: Rig) -> dict[str, float]:
    heard: dict[str, float] = {}
    for seg in rig.bottom.segments:
        for label, dur in seg.played_amps(rig.tts).items():
            heard[label] = heard.get(label, 0.0) + dur
    return heard


def _recorded(rig: Rig) -> dict[str, float]:
    rec: dict[str, float] = {}
    for w in rig.recorder.writes:
        for amp, dur in w.amps.items():
            if amp == 0:
                continue
            label = rig.tts.label_for_amp(amp)
            rec[label] = rec.get(label, 0.0) + dur
    return rec


def _committed_messages(agent: Agent) -> list[tuple[str, str]]:
    return [
        (item.role, item.text_content or "")
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role in ("user", "assistant")
    ]


def _speaking_started_at(rig: Rig, t0: float) -> float:
    speaking = [t for (t, kind, p) in rig.timeline if kind == "agent_state" and p == "speaking"]
    assert speaking, "agent never started speaking"
    return min(speaking) - t0


# ---------------------------------------------------------------------------
# scenario mirroring the recorded run
# ---------------------------------------------------------------------------

# scripted timeline (seconds, session-relative)
USER1_START, USER1_END = 0.5, 2.5
# the caller barges into reply A shortly after it starts (reply A starts ~3.6s)
BARGE_START, BARGE_END = 4.4, 6.4
# the caller's STT FINAL lands 8s after end-of-speech (interim at +4s) - later
# than false_interruption_timeout (2s), which is what arms the stale resume
SLOW_STT_DELAY = 8.0
FAST_STT_DELAY = 0.2

REPLY_A_AUDIO = 7.0  # reply A synthesizes ~7s of audio, most of it unplayed at the pause

# how much of reply A can legitimately play before the barge-in registers:
# bounded by BARGE_START + interruption min_duration (0.5) + one VAD hop of
# slack. Anything played beyond that bound is stale audio bleeding into (or
# past) the caller's REAL turn.
MIN_INTERRUPTION_DURATION = 0.5
VAD_SLACK = 0.3
ASSERT_MARGIN = 0.5


def _booking_call(stt_delay: float) -> FakeActions:
    actions = FakeActions()
    actions.add_user_speech(USER1_START, USER1_END, "the fourteenth through the seventeenth")
    actions.add_llm(REPLY_A, ttft=0.2, duration=0.4)
    actions.add_tts(REPLY_A_AUDIO, ttfb=0.2, duration=0.3)
    # the caller's real turn, barged into reply A
    actions.add_user_speech(BARGE_START, BARGE_END, USER_FINAL, stt_delay=stt_delay)
    actions.add_llm(REPLY_B, ttft=0.2, duration=0.4)
    actions.add_tts(4.5, ttfb=0.2, duration=0.3)
    return actions


async def _run_booking_call(stt_delay: float, *, drain_delay: float) -> tuple[Rig, Agent, float]:
    rig = create_rig(_booking_call(stt_delay))
    agent = HotelAgent()
    t0 = await run_rig(rig, agent, drain_delay=drain_delay)
    return rig, agent, t0


def _stale_reply_a_budget(rig: Rig, t0: float) -> float:
    """Seconds of reply A that could have legitimately played before the
    caller's real turn registered as a barge-in."""
    t_speaking = _speaking_started_at(rig, t0)
    return (BARGE_START + MIN_INTERRUPTION_DURATION + VAD_SLACK) - t_speaking


def _assert_run_committed_shape(agent: Agent) -> None:
    """The chat context looks exactly like the recorded run: the caller's final
    utterance is a committed user turn, answered by a committed reply B."""
    messages = _committed_messages(agent)
    user_texts = [text for role, text in messages if role == "user"]
    assert USER_FINAL in user_texts, f"caller's final turn not committed: {messages}"

    reply_b = [
        text for role, text in messages if role == "assistant" and REPLY_B.startswith(text.rstrip())
    ]
    assert reply_b, f"no reply committed for the caller's final turn: {messages}"


# ---------------------------------------------------------------------------
# regression tests - these failed before the stale-resume fix
# ---------------------------------------------------------------------------


async def test_caller_final_turn_window_free_of_stale_reply_audio() -> None:
    """What the CALLER hears must match what the transcript records.

    The transcript for the end of the call reads: caller says "...finalize the
    booking", agent answers with reply B. The audio in that window must
    therefore be reply B (or silence) - never a replay of reply A, which the
    caller had already cut off seconds earlier.

    Before the fix the false-interruption timer resumed the paused reply A
    while the caller's final transcript was still in flight, so several seconds
    of reply-A audio played DURING/AFTER the caller's real turn - the audio
    occupying reply B's window ("what's said") belonged to reply A, while the
    chat context recorded reply B ("what's added to the convo").
    """
    rig, agent, t0 = await _run_booking_call(SLOW_STT_DELAY, drain_delay=9.0)
    dump(rig, agent, t0, "slow stt final: caller turn window")

    _assert_run_committed_shape(agent)

    played_a = _live_heard(rig).get(REPLY_A, 0.0)
    budget = _stale_reply_a_budget(rig, t0)
    assert played_a <= budget + ASSERT_MARGIN, (
        f"stale audio divergence: {played_a:.2f}s of reply A ({REPLY_A[:45]!r}...) reached the "
        f"live sink, but only {budget:.2f}s could have played before the caller's real turn "
        f"({USER_FINAL[:35]!r}...) registered as a barge-in. The extra "
        f"{played_a - budget:.2f}s of stale reply-A audio played DURING/AFTER the caller's "
        f"final turn - occupying the window where the committed transcript shows "
        f"{REPLY_B[:45]!r}..."
    )


async def test_insights_recording_free_of_stale_reply_audio() -> None:
    """Same invariant applied to the recorder channel - the audio agent
    insights plays back for this call.

    Before the fix the recording contained the stale reply-A tail that the
    false-interruption timer resumed, so scrubbing to the final assistant item
    in insights played reply-A audio against reply B's transcript entry - the
    exact observation in run SR_4UpFo6cNC7wA.
    """
    rig, agent, t0 = await _run_booking_call(SLOW_STT_DELAY, drain_delay=9.0)
    dump(rig, agent, t0, "slow stt final: insights recording")

    _assert_run_committed_shape(agent)

    recorded_a = _recorded(rig).get(REPLY_A, 0.0)
    budget = _stale_reply_a_budget(rig, t0)
    assert recorded_a <= budget + ASSERT_MARGIN, (
        f"stale audio in the recording: {recorded_a:.2f}s of reply A ({REPLY_A[:45]!r}...) was "
        f"written to the recorder, but only {budget:.2f}s could have played before the caller's "
        f"real turn registered as a barge-in. Insights playback for the final assistant item "
        f"({REPLY_B[:45]!r}...) surfaces the extra {recorded_a - budget:.2f}s of reply-A audio."
    )


async def test_stale_reply_never_resumes_over_pending_user_turn() -> None:
    """The false-interruption resume must not fire while the caller's turn is
    merely waiting on a slow STT final.

    ``agent_false_interruption(resumed=True)`` is the session-level fingerprint
    of the stale replay: the timer treated a REAL turn (VAD saw 2s of speech,
    the final just hadn't landed yet) as a false interruption and resumed the
    outdated reply. Before the fix this fired - twice, because each interim
    transcript re-paused and re-armed the timer.
    """
    rig, agent, t0 = await _run_booking_call(SLOW_STT_DELAY, drain_delay=9.0)
    dump(rig, agent, t0, "slow stt final: resume events")

    _assert_run_committed_shape(agent)

    resumes = [p for (_, kind, p) in rig.timeline if kind == "false_interruption" and p]
    assert not resumes, (
        f"the false-interruption timer resumed the stale reply {len(resumes)} time(s) while the "
        f"caller's real turn ({USER_FINAL[:35]!r}...) was pending its slow STT final"
    )


# ---------------------------------------------------------------------------
# control - passes on HEAD, proving the assertions are calibrated
# ---------------------------------------------------------------------------


async def test_fast_stt_final_has_no_divergence() -> None:
    """Same call with a prompt STT final: the barge-in interrupts reply A for
    good, no stale audio plays, and reply B's audio actually reaches the sink.
    Proves the repro assertions fail only through the slow-final + resume-timer
    interaction, not through the scenario shape itself."""
    rig, agent, t0 = await _run_booking_call(FAST_STT_DELAY, drain_delay=8.0)
    dump(rig, agent, t0, "fast stt final: control")

    _assert_run_committed_shape(agent)

    heard = _live_heard(rig)
    played_a = heard.get(REPLY_A, 0.0)
    budget = _stale_reply_a_budget(rig, t0)
    assert played_a <= budget + ASSERT_MARGIN, (
        f"control broken: {played_a:.2f}s of reply A played with a fast STT final "
        f"(budget {budget:.2f}s)"
    )

    played_b = heard.get(REPLY_B, 0.0)
    assert played_b > 1.0, f"control broken: reply B audio never played ({played_b:.2f}s)"

    resumes = [p for (_, kind, p) in rig.timeline if kind == "false_interruption" and p]
    assert not resumes, "control broken: false-interruption resume fired with a fast STT final"


async def test_short_noise_blip_still_resumes() -> None:
    """Guardrail for the noise path the resume timer exists for: a brief
    VAD-only blip (no transcript, shorter than
    ``false_interruption_max_speech_duration``) must still pause and then
    resume the reply, playing it to completion."""
    actions = FakeActions()
    actions.add_user_speech(USER1_START, USER1_END, "the fourteenth through the seventeenth")
    actions.add_llm(REPLY_A, ttft=0.2, duration=0.4)
    actions.add_tts(REPLY_A_AUDIO, ttfb=0.2, duration=0.3)
    # 0.7s VAD-only noise blip during playout - no STT events at all
    actions.add_user_speech(4.4, 5.1, "")

    rig = create_rig(actions)
    agent = HotelAgent()
    t0 = await run_rig(rig, agent, drain_delay=12.0)
    dump(rig, agent, t0, "noise blip: resume control")

    resumes = [p for (_, kind, p) in rig.timeline if kind == "false_interruption" and p]
    assert resumes, "noise blip did not resume the paused reply"

    played_a = _live_heard(rig).get(REPLY_A, 0.0)
    assert played_a > REPLY_A_AUDIO - ASSERT_MARGIN, (
        f"reply A did not play to completion after the noise blip ({played_a:.2f}s of "
        f"{REPLY_A_AUDIO:.2f}s)"
    )

    messages = _committed_messages(agent)
    reply_a = [text for role, text in messages if role == "assistant" and text.startswith("Got it")]
    assert reply_a, f"reply A not committed after the noise blip: {messages}"


# ---------------------------------------------------------------------------
# pause ownership invariants (whitebox)
#
# The shared audio output must never stay paused without an owning
# _PausedSpeechInfo: every path that drops the pause bookkeeping has to resume
# the output, otherwise later replies push audio into a paused sink (text
# committed, nothing ever played) and the leftover buffered audio surfaces
# against a different turn.
# ---------------------------------------------------------------------------


async def _idle_rig() -> tuple[Rig, HotelAgent]:
    actions = FakeActions()
    # a single user speech far in the future keeps the fakes alive but inert
    actions.add_user_speech(60.0, 60.5, "")
    rig = create_rig(actions)
    agent = HotelAgent()
    await rig.session.start(agent)
    rig.audio_input.push(0.1)
    await asyncio.sleep(0.1)
    return rig, agent


async def _teardown_rig(rig: Rig) -> None:
    with contextlib.suppress(Exception):
        await rig.session.aclose()
    with contextlib.suppress(Exception):
        await rig.transcript_sync.aclose()


def _bottom_paused(rig: Rig) -> bool:
    return rig.bottom._paused_at is not None


async def test_false_interruption_timer_abandon_resumes_output() -> None:
    """Timer fires after a NEW speech became current: the pause is abandoned,
    but the shared output must be resumed, otherwise the new speech plays
    nothing."""
    from livekit.agents.voice.speech_handle import SpeechHandle

    rig, _agent = await _idle_rig()
    try:
        activity = rig.session._activity
        assert activity is not None

        paused_handle = SpeechHandle.create()
        new_handle = SpeechHandle.create()

        activity._update_paused_speech(paused_handle, timeout=0.01)
        rig.session.output.audio.pause()
        assert _bottom_paused(rig)

        activity._current_speech = new_handle  # a new speech took over
        activity._start_false_interruption_timer(0.01)
        await asyncio.sleep(0.1)

        assert activity._paused_speech is None
        assert not _bottom_paused(rig), (
            "audio output left paused with no owner after the pause was abandoned"
        )
    finally:
        await _teardown_rig(rig)


async def test_false_interruption_timer_done_speech_resumes_output() -> None:
    """Timer fires after the paused speech already completed (e.g. its audio
    tail drained during the pause): nothing to resume on its behalf, but the
    output must not stay paused."""
    from livekit.agents.voice.speech_handle import SpeechHandle

    rig, _agent = await _idle_rig()
    try:
        activity = rig.session._activity
        assert activity is not None

        paused_handle = SpeechHandle.create()
        activity._update_paused_speech(paused_handle, timeout=0.01)
        rig.session.output.audio.pause()
        assert _bottom_paused(rig)

        paused_handle._mark_done()  # completed while paused
        activity._current_speech = None
        activity._start_false_interruption_timer(0.01)
        await asyncio.sleep(0.1)

        assert activity._paused_speech is None
        assert not _bottom_paused(rig), "audio output left paused after the paused speech completed"
    finally:
        await _teardown_rig(rig)


async def test_cancel_speech_pause_repairs_ownerless_pause() -> None:
    """A new user turn (final transcript / turn-completed path) must never
    start against a paused output, even if the pause bookkeeping was already
    cleared."""
    rig, _agent = await _idle_rig()
    try:
        activity = rig.session._activity
        assert activity is not None

        rig.session.output.audio.pause()  # ownerless pause (leaked earlier)
        assert activity._paused_speech is None
        assert _bottom_paused(rig)

        await activity._cancel_speech_pause()

        assert not _bottom_paused(rig), "new user turn started against a paused audio output"
    finally:
        await _teardown_rig(rig)
