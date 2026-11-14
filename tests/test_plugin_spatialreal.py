"""SpatialReal avatar plugin: segment lifecycle, pause/resume and provider recovery.

Hermetic: the spatialreal SDK session and the framework audio buffer are stubbed, and the
plugin's state machine is driven directly, so no network or LiveKit room is needed.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
from spatialreal import PlaybackSignal

from livekit import rtc
from livekit.agents.voice import io as voice_io
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput
from livekit.plugins.spatialreal import avatar
from livekit.plugins.spatialreal.avatar import AvatarSession
from livekit.plugins.spatialreal.resumable_queue_io import ResumableQueueAudioOutput

pytestmark = pytest.mark.unit


async def _stop_idle_end_timer(session: AvatarSession) -> None:
    """Cancel the missing-end-marker timer a test left armed by sending frames."""
    task = session._active_segment_idle_end_task
    session._cancel_active_segment_idle_end()
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


# --- Segment lifecycle ------------------------------------------------------------------


class FakeAvatarkitSession:
    def __init__(self, req_id: str = "req-1") -> None:
        self.req_id = req_id
        self.interrupt_calls = 0
        self.sent: list[tuple[int, bool]] = []

    async def send_audio(self, audio: bytes, end: bool = False) -> str:
        self.sent.append((len(audio), end))
        return self.req_id

    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        return self.req_id

    async def pause(self) -> str:
        self.pause_calls = getattr(self, "pause_calls", 0) + 1
        return self.req_id

    async def resume(self) -> str:
        self.resume_calls = getattr(self, "resume_calls", 0) + 1
        return self.req_id

    async def close(self) -> None:
        pass


class FakeBuffer:
    """Records the notify_* calls the plugin makes toward the framework."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_playback_started(self) -> None:
        self.events.append("started")

    def notify_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        self.events.append(("finished", interrupted))


def make_session() -> tuple[AvatarSession, FakeAvatarkitSession, FakeBuffer]:
    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    fake = FakeAvatarkitSession()
    buffer = FakeBuffer()
    session._avatarkit_session = fake  # type: ignore[assignment]
    session._audio_buffer = buffer  # type: ignore[assignment]
    return session, fake, buffer


def make_frame(samples: int = 240, sample_rate: int = 24000) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
    )


def provider_end_signal(req_id: str) -> PlaybackSignal:
    return PlaybackSignal(req_id=req_id, end=True)


async def test_early_provider_completion_is_preserved() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    # provider completion arrives BEFORE the local AudioSegmentEnd
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started"], buffer.events

    assert await session._finalize_active_segment(source="segment_end")
    assert buffer.events == ["started", ("finished", False)], buffer.events
    assert not session._segments and not session._pending_segment_ids
    assert (0, True) in fake.sent  # end=true was sent to the provider
    print("PASS early provider completion preserved (provider_end_deferred)")


async def test_duplicate_interrupts_coalesced() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    await session._handle_interrupt()
    await session._handle_interrupt()  # duplicate clear_buffer

    assert fake.interrupt_calls == 1, fake.interrupt_calls
    # started is synthesized before finished even though playback was never observed
    assert buffer.events == ["started", ("finished", True)], buffer.events
    print("PASS duplicate interrupts coalesced, started precedes finished")


async def test_started_precedes_finished_without_signals() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    assert await session._finalize_active_segment(source="segment_end")
    assert buffer.events == [], buffer.events  # waiting for completion

    assert session._complete_segment(
        req_id=fake.req_id, interrupted=False, reason="timeout_no_playback_signal"
    )
    assert buffer.events == ["started", ("finished", False)], buffer.events
    print("PASS no-signal fallback still emits started before finished")


async def test_remote_track_marks_playback_started() -> None:
    session, fake, buffer = make_session()
    playback_events = []
    session.on("playback_started", playback_events.append)

    await session._send_audio_frame(make_frame())

    silent = make_frame()
    session._on_avatar_audio_frame(silent)
    assert buffer.events == [], buffer.events

    noise_floor = make_frame()
    noise_floor.data[0] = 40  # below AVATAR_AUDIO_ACTIVITY_THRESHOLD
    session._on_avatar_audio_frame(noise_floor)
    assert buffer.events == [], buffer.events

    speech = make_frame()
    speech.data[0] = 1200
    session._on_avatar_audio_frame(speech)
    assert buffer.events == ["started"], buffer.events
    assert len(playback_events) == 1
    assert playback_events[0].request_id == fake.req_id
    assert playback_events[0].source == "livekit_avatar_audio_track"
    assert playback_events[0].observed_at > 0

    segment = session._segments[fake.req_id]
    assert segment.playback_start_source == "livekit_avatar_audio_track"

    session._complete_segment(req_id=fake.req_id, interrupted=False, reason="test")
    print("PASS playback start detected from remote track with energy threshold")


async def test_active_speaker_secondary_signal() -> None:
    session, fake, buffer = make_session()
    playback_events = []
    session.on("playback_started", playback_events.append)
    session._represented_participant_identity = "agent-1"

    await session._send_audio_frame(make_frame())

    stranger = SimpleNamespace(identity="someone-else", attributes={})
    session._on_active_speakers_changed([stranger])
    assert buffer.events == [], buffer.events

    session._avatar_is_speaking = False
    proxy_avatar = SimpleNamespace(
        identity="egress-worker", attributes={"lk.publish_on_behalf": "agent-1"}
    )
    session._on_active_speakers_changed([proxy_avatar])
    assert buffer.events == ["started"], buffer.events
    assert len(playback_events) == 1
    assert playback_events[0].source == "livekit_active_speaker"

    segment = session._segments[fake.req_id]
    assert segment.playback_start_source == "livekit_active_speaker"

    session._complete_segment(req_id=fake.req_id, interrupted=False, reason="test")
    print("PASS active-speaker signal matched via lk.publish_on_behalf")


class _Leaf(voice_io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="leaf",
            next_in_chain=None,
            sample_rate=24000,
            capabilities=voice_io.AudioOutputCapabilities(pause=False),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


class _Wrapper(voice_io.AudioOutput):
    def __init__(self, next_in_chain: voice_io.AudioOutput) -> None:
        super().__init__(
            label="wrapper",
            next_in_chain=next_in_chain,
            sample_rate=24000,
            capabilities=voice_io.AudioOutputCapabilities(pause=False),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


async def test_audio_tail_attach_and_restore() -> None:
    session, _, _ = make_session()

    leaf = _Leaf()
    wrapper = _Wrapper(leaf)  # auto-wraps leaf behind an _AudioSinkProxy
    output = voice_io.AgentOutput(
        video_changed=lambda: None,
        audio_changed=lambda: None,
        transcription_changed=lambda: None,
    )
    output.audio = wrapper

    queue_buffer = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    session._audio_buffer = queue_buffer
    session._agent_session = SimpleNamespace(output=output)  # type: ignore[assignment]
    session._original_audio_output = output.audio
    session._original_audio_tail = AvatarSession._resolve_audio_tail(output.audio)
    assert session._original_audio_tail is leaf

    output.replace_audio_tail(queue_buffer)
    session._audio_output_attached = True
    assert output.audio is wrapper  # wrappers stay in place
    assert AvatarSession._resolve_audio_tail(output.audio) is queue_buffer

    session._detach_audio_output()
    assert output.audio is wrapper
    assert AvatarSession._resolve_audio_tail(output.audio) is leaf
    print("PASS tail swap keeps wrappers and restores the original tail on close")


async def test_audio_tail_restore_without_wrappers() -> None:
    session, _, _ = make_session()

    leaf = _Leaf()
    output = voice_io.AgentOutput(
        video_changed=lambda: None,
        audio_changed=lambda: None,
        transcription_changed=lambda: None,
    )
    output.audio = leaf

    queue_buffer = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    session._audio_buffer = queue_buffer
    session._agent_session = SimpleNamespace(output=output)  # type: ignore[assignment]
    session._original_audio_output = output.audio
    session._original_audio_tail = AvatarSession._resolve_audio_tail(output.audio)

    output.replace_audio_tail(queue_buffer)
    session._audio_output_attached = True

    session._detach_audio_output()
    assert output.audio is leaf
    print("PASS whole-chain fallback restore works when no wrappers exist")


async def test_duplicate_provider_end_ignored() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    assert await session._finalize_active_segment(source="segment_end")
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started", ("finished", False)], buffer.events

    # egress ALR retransmission can re-deliver end=true for the same req_id
    session._on_playback_signal(provider_end_signal(fake.req_id))
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started", ("finished", False)], buffer.events
    assert not session._early_provider_started_ids and not session._early_provider_completed_ids
    print("PASS duplicate provider end=true events are ignored")


async def test_native_pause_defers_duration_completion() -> None:
    """Regression (live E2E against the spatialreal.dev test backend): on the native
    server-side path the duration-derived completion ignored the time the server held
    playback, so the segment finished ~pause-duration before the avatar stopped talking.
    A native pause must hold completion, and resume must push the expected end back."""
    import time

    session, fake, buffer = make_session()
    session._server_playback_control = True

    await session._send_audio_frame(make_frame())
    req = session._active_req_id
    assert await session._finalize_active_segment(source="segment_end")
    segment = session._segments[req]
    # 10s of audio whose playback started 4s ago
    segment.attempt_duration = 10.0
    segment.playback_started = True
    segment.playback_started_at = time.time() - 4.0
    before = session._compute_completion_timeout(segment)

    session._on_pause()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)

    # a completion timer that fires while the server holds playback must not finish the segment
    await session._wait_for_segment_completion_timeout(req, 0)
    assert buffer.events == [], buffer.events

    await asyncio.sleep(0.3)  # server holds playback
    session._on_resume()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)

    # the paused time does not count toward playback: remaining time is unchanged by the pause
    after = session._compute_completion_timeout(segment)
    assert after > before - 0.1, (before, after)
    assert buffer.events == [], buffer.events

    session._complete_segment(req_id=req, interrupted=False, reason="test")
    print("PASS native pause holds and defers duration-derived completion")


async def test_native_resume_without_provider_releases_hold() -> None:
    """A resume that lands while the provider session is unavailable (reconnect window)
    must still release the native hold; otherwise the duration fallback never completes
    the segment and the framework's speech handle is stranded."""
    import time

    session, fake, buffer = make_session()
    session._server_playback_control = True

    await session._send_audio_frame(make_frame())
    req = session._active_req_id
    assert await session._finalize_active_segment(source="segment_end")
    segment = session._segments[req]
    segment.attempt_duration = 10.0
    segment.playback_started = True
    segment.playback_started_at = time.time() - 4.0

    session._on_pause()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)
    assert segment.native_paused_at is not None

    session._avatarkit_session = None  # provider reconnect in progress
    session._on_resume()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)
    assert segment.native_paused_at is None

    await session._wait_for_segment_completion_timeout(req, 0)
    assert buffer.events == [("finished", False)], buffer.events
    print("PASS native resume during provider reconnect releases the hold")


async def test_livekit_egress_token_ttl_outlives_session() -> None:
    """The egress LiveKit room-join token must outlive a long conversation.

    Regression for the 1h-TTL bug: after the token expired mid-session, an
    egress rejoin (triggered by a provider-WS reconnect) hit SFU 401 and the
    avatar dropped out permanently. The token TTL is decoupled from the
    SpatialReal session TTL and defaults to 24h.
    """
    import base64
    import json
    from datetime import timedelta

    from livekit.plugins.spatialreal.avatar import DEFAULT_LIVEKIT_TOKEN_TTL

    assert DEFAULT_LIVEKIT_TOKEN_TTL >= timedelta(hours=12)

    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    assert session._livekit_token_ttl == DEFAULT_LIVEKIT_TOKEN_TTL

    # mint a token the same way start() does and decode its exp/nbf claims
    from livekit import api

    jwt = (
        api.AccessToken(api_key="devkey", api_secret="secret" * 6)
        .with_identity("spatialreal-avatar")
        .with_ttl(session._livekit_token_ttl)
        .to_jwt()
    )
    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload))
    assert claims["exp"] - claims["nbf"] >= 12 * 3600, claims

    override = AvatarSession(
        api_key="k", app_id="a", avatar_id="av", livekit_token_ttl_seconds=7200
    )
    assert override._livekit_token_ttl == timedelta(seconds=7200)
    print("PASS egress LiveKit token TTL is long (>=12h default) and configurable")


async def test_livekit_egress_token_lets_the_egress_publish_data() -> None:
    """The egress also publishes data, so its room-join token must allow it."""
    import base64
    import json

    from livekit import api
    from livekit.plugins.spatialreal.avatar import _egress_video_grants

    grants = _egress_video_grants("room-1")
    assert grants.room_join is True and grants.room == "room-1"
    assert grants.can_publish_data is True
    assert grants.can_subscribe is False

    jwt = (
        api.AccessToken(api_key="devkey", api_secret="secret" * 6)
        .with_identity("spatialreal-avatar")
        .with_grants(grants)
        .to_jwt()
    )
    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    video = json.loads(base64.urlsafe_b64decode(payload))["video"]
    assert video["canPublishData"] is True, video
    assert video["room"] == "room-1", video
    print("PASS egress LiveKit token lets the egress publish data")


async def test_native_pause_resume_keeps_audio_flowing() -> None:
    """With server playback_control, pause/resume drive the SDK directly and do
    NOT interrupt, retain, or divert frames — the segment stays active."""
    session, fake, buffer = make_session()
    session._server_playback_control = True

    await session._send_audio_frame(make_frame())
    active = session._active_req_id
    assert active is not None

    session._on_pause()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)

    # native pause: SDK.pause() called, NOT interrupt; no frame diversion
    assert getattr(fake, "pause_calls", 0) == 1
    assert fake.interrupt_calls == 0
    assert session._pause_requested is False
    assert session._paused_segment is None
    assert session._active_req_id == active  # segment untouched

    # audio keeps flowing to the same request while paused
    await session._send_audio_frame(make_frame())
    assert session._active_req_id == active

    session._on_resume()
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)
    assert getattr(fake, "resume_calls", 0) == 1
    assert buffer.events == []  # never completed; still playing

    session._complete_segment(req_id=active, interrupted=False, reason="test")
    print("PASS native pause/resume drives SDK without interrupt/retain/divert")


async def test_native_pause_after_confirmed_interruption_leaves_provider_alone() -> None:
    """A confirmed interruption ends the segment. When the framework still pauses and
    resumes that speech afterwards (it can, when its own bookkeeping of the interruption
    lags), there is no audio in flight to hold: the provider's last request is the
    interrupted one and must be neither paused nor resumed."""

    async def drain() -> None:
        await asyncio.sleep(0)
        await asyncio.gather(*list(session._background_tasks), return_exceptions=True)

    session, fake, buffer = make_session()
    session._server_playback_control = True

    await session._send_audio_frame(make_frame())
    session._on_pause()  # the user starts talking: hold the playing segment
    await drain()
    assert getattr(fake, "pause_calls", 0) == 1

    session._on_clear_buffer()  # confirmed interruption
    await drain()
    assert not session._segments and session._active_req_id is None, session._segments

    session._on_pause()
    await drain()
    session._on_resume()
    await drain()

    assert getattr(fake, "pause_calls", 0) == 1, "paused the interrupted request again"
    assert getattr(fake, "resume_calls", 0) == 0, "resumed the interrupted request"
    print("PASS native pause/resume after a confirmed interruption leaves the provider alone")


async def _drain(session: AvatarSession) -> None:
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)


async def test_native_pause_before_audio_holds_the_first_frames() -> None:
    """The user can interrupt while the reply is still in llm/tts. The server has no
    audio to hold yet, so the first frames are held here and only sent on resume."""
    session, fake, buffer = make_session()
    session._server_playback_control = True

    session._on_pause()
    await _drain(session)
    await session._send_audio_frame(make_frame())
    await session._send_audio_frame(make_frame())
    assert fake.sent == [], "frames played during the interruption hold"
    assert getattr(fake, "pause_calls", 0) == 0

    session._on_resume()
    await _drain(session)
    assert len(fake.sent) == 2, fake.sent
    assert getattr(fake, "resume_calls", 0) == 0, "resumed a request the server never had"
    assert session._paused_segment is None
    await _stop_idle_end_timer(session)
    print("PASS native pause before the first frame holds the frames until resume")


async def test_native_pause_before_audio_then_interruption_drops_the_frames() -> None:
    session, fake, buffer = make_session()
    session._server_playback_control = True

    session._on_pause()
    await _drain(session)
    await session._send_audio_frame(make_frame())
    session._on_clear_buffer()  # confirmed interruption
    await _drain(session)
    session._on_resume()
    await _drain(session)

    assert fake.sent == [], fake.sent
    assert session._paused_segment is None and not session._segments
    print("PASS native pause before the first frame, then interruption, sends nothing")


async def test_provider_error_finishes_segments_as_interrupted() -> None:
    """Recovery after a provider error replaces the connection, which loses the audio
    it was given. Like a close, that speech ends as interrupted and a server-side hold
    is released, instead of the segment later timing out as fully played."""
    session, fake, buffer = make_session()
    session._server_playback_control = True
    session._provider_session_generation = 1
    recoveries: list[str] = []
    session._schedule_provider_recovery = lambda *, reason: recoveries.append(reason)  # type: ignore[method-assign]

    await session._send_audio_frame(make_frame())
    session._on_pause()
    await _drain(session)
    segment = session._segments[session._active_req_id]
    assert segment.native_paused_at is not None

    session._on_provider_error(1, RuntimeError("boom"))

    assert segment.native_paused_at is None, "native hold was not released"
    assert not session._segments and session._active_req_id is None
    assert ("finished", True) in buffer.events, buffer.events
    assert len(recoveries) == 1
    print("PASS provider error finishes in-flight segments as interrupted")


async def test_pause_timeout_finalizes_segment_interrupted() -> None:
    """A server-forced pause_timeout interrupt must finalize the segment so the
    framework's speech handle isn't stranded."""
    from spatialreal import InterruptReason, PlaybackState, PlaybackStateEvent

    session, fake, buffer = make_session()
    session._server_playback_control = True

    await session._send_audio_frame(make_frame())
    req = session._active_req_id

    session._on_playback_state(
        PlaybackStateEvent(
            req_id=req,
            state=PlaybackState.INTERRUPTED,
            played_ms=1500,
            reason=InterruptReason.PAUSE_TIMEOUT,
        )
    )
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)

    assert buffer.events and buffer.events[-1] == ("finished", True)
    assert not session._segments
    print("PASS pause_timeout finalizes the segment as interrupted")


async def test_non_pause_timeout_playback_state_is_noop() -> None:
    """explicit/preempted interrupts flow through clear_buffer, not this path."""
    from spatialreal import InterruptReason, PlaybackState, PlaybackStateEvent

    session, fake, buffer = make_session()
    session._server_playback_control = True
    await session._send_audio_frame(make_frame())

    session._on_playback_state(
        PlaybackStateEvent(
            req_id=session._active_req_id,
            state=PlaybackState.INTERRUPTED,
            reason=InterruptReason.EXPLICIT,
        )
    )
    await asyncio.sleep(0)
    await asyncio.gather(*list(session._background_tasks), return_exceptions=True)
    # not finalized here — the framework's own clear_buffer owns explicit interrupts
    assert buffer.events == []
    session._complete_segment(req_id=session._active_req_id, interrupted=True, reason="test")
    print("PASS non-pause_timeout playback-state is a no-op on this path")


async def test_session_ttl_outlives_the_call_within_the_server_cap() -> None:
    """The server closes the SpatialReal session when it expires and rejects an
    expiry more than 24h out."""
    from datetime import timedelta

    from livekit.plugins.spatialreal.avatar import DEFAULT_SESSION_TTL

    assert timedelta(hours=12) <= DEFAULT_SESSION_TTL < timedelta(hours=24)


# --- Pause, resume and interruption -----------------------------------------------------


class _ProviderSpy:
    def __init__(self) -> None:
        self.generation = 1
        self.interrupt_calls = 0
        self.sent: list[tuple[str, bytes, bool]] = []

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        req_id = f"request-{self.generation}"
        self.sent.append((req_id, audio, end))
        return req_id

    async def interrupt(self) -> str:
        req_id = f"request-{self.generation}"
        self.interrupt_calls += 1
        self.generation += 1
        return req_id

    async def init(self) -> None:
        return None

    async def start(self) -> str:
        return "connection-test"

    async def close(self) -> None:
        return None


class _ReusedRequestIdProvider(_ProviderSpy):
    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        req_id = "request-1"
        self.sent.append((req_id, audio, end))
        return req_id


class _FailingInterruptProvider(_ProviderSpy):
    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        raise RuntimeError("interrupt unavailable")


class _FailingResumeProvider(_ProviderSpy):
    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        raise RuntimeError("send unavailable")


class _FailFirstSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_calls = 0

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        if self.send_calls == 1:
            raise RuntimeError("transient send failure")
        return await super().send_audio(audio=audio, end=end)


class _BlockingInterruptProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_started = asyncio.Event()
        self.allow_interrupt = asyncio.Event()

    async def interrupt(self) -> str:
        self.interrupt_started.set()
        await self.allow_interrupt.wait()
        return await super().interrupt()


class _BlockingFailSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.allow_failure = asyncio.Event()
        self.fail_next_send = True

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        if self.fail_next_send:
            self.fail_next_send = False
            self.send_started.set()
            await self.allow_failure.wait()
            raise RuntimeError("send unavailable")
        return await super().send_audio(audio=audio, end=end)


class _BlockingSuccessfulSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.allow_send = asyncio.Event()
        self.send_calls = 0

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        if self.send_calls == 1:
            self.send_started.set()
            await self.allow_send.wait()
        return await super().send_audio(audio=audio, end=end)


class _AudioBufferSpy:
    def __init__(self) -> None:
        self.playback_started = 0
        self.playback_finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.playback_started += 1

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
    ) -> None:
        self.playback_finished.append((playback_position, interrupted))


class _StreamingAudioBufferSpy(_AudioBufferSpy):
    def __init__(self, items: list[rtc.AudioFrame | AudioSegmentEnd]) -> None:
        super().__init__()
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


class _AudioStreamCloseSpy:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _session() -> avatar.AvatarSession:
    session = avatar.AvatarSession(api_key="test", app_id="test", avatar_id="test")
    session._livekit_egress = object()
    session._resolved_sample_rate = 24000
    return session


def _buffered_frame(value: int, duration: float = 0.25) -> avatar._BufferedAudioFrame:
    sample_rate = 8
    samples = round(sample_rate * duration)
    return avatar._BufferedAudioFrame(
        audio=bytes([value, 0]) * samples,
        duration=duration,
        sample_rate=sample_rate,
        num_channels=1,
    )


def _rtc_frame(value: int = 9) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([value, 0]) * 4,
        sample_rate=8,
        num_channels=1,
        samples_per_channel=4,
    )


def _playing_segment() -> avatar._SegmentState:
    frames = deque(_buffered_frame(value) for value in (1, 2, 3, 4))
    return avatar._SegmentState(
        req_id="request-1",
        pushed_duration=1.0,
        playback_started=True,
        playback_started_at=100.0,
        playback_start_source="test",
        audio_frames=frames,
        buffered_duration=1.0,
        attempt_duration=1.0,
        input_finalized=True,
    )


def test_resumable_queue_advertises_pause_and_coalesces_lifecycle_events() -> None:
    async def run_test() -> None:
        output = ResumableQueueAudioOutput(sample_rate=24000, wait_playback_start=True)
        events: list[str] = []
        output.on("pause", lambda: events.append("pause"))
        output.on("resume", lambda: events.append("resume"))

        output.pause()
        output.pause()
        output.resume()
        output.resume()

        assert output.can_pause is True
        assert events == ["pause", "resume"]

    asyncio.run(run_test())


def test_false_interruption_replays_only_unplayed_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is segment
        assert segment.playback_position_offset == pytest.approx(0.5)
        assert audio.playback_finished == []

        session._pause_requested = False
        await session._handle_resume()

        replay_audio = b"".join(audio_bytes for _, audio_bytes, end in provider.sent if not end)
        assert replay_audio == _buffered_frame(3).audio + _buffered_frame(4).audio
        assert provider.sent[-1] == ("request-2", b"", True)
        assert session._paused_segment is None
        assert list(session._pending_segment_ids) == ["request-2"]
        assert audio.playback_finished == []

    asyncio.run(run_test())


def test_confirmed_interruption_discards_paused_tail_without_second_provider_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        session._discard_requested = True
        session._pause_requested = False
        await session._handle_interrupt()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(pytest.approx(0.5), True)]

    asyncio.run(run_test())


def test_confirmed_interruption_before_provider_request_completes_framework_playout() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._pause_requested = True

        frame = rtc.AudioFrame(
            data=bytes([9, 0]) * 4,
            sample_rate=8,
            num_channels=1,
            samples_per_channel=4,
        )
        await session._send_audio_frame(frame)
        assert session._paused_segment is not None
        assert session._segments == {}

        session._discard_requested = True
        session._pause_requested = False
        await session._handle_interrupt()

        assert provider.interrupt_calls == 0
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(0.0, True)]

    asyncio.run(run_test())


def test_session_close_completes_segment_paused_before_provider_request() -> None:
    session = _session()
    audio = _AudioBufferSpy()
    segment = session._new_local_segment()
    session._audio_buffer = audio
    session._paused_segment = segment
    session._append_audio_frame(segment, _buffered_frame(1), sent_to_provider=False)

    session._complete_all_segments(interrupted=True, reason="session_close")

    assert session._paused_segment is None
    assert session._segments == {}
    assert audio.playback_finished == [(0.0, True)]


def test_partial_frame_resume_preserves_sample_alignment() -> None:
    frame = _buffered_frame(7, duration=0.5)
    segment = avatar._SegmentState(
        req_id="request-1",
        pushed_duration=0.5,
        audio_frames=deque([frame]),
        buffered_duration=0.5,
    )

    remaining = avatar.AvatarSession._remaining_audio_frames(segment, 0.25)

    assert len(remaining) == 1
    assert remaining[0].duration == pytest.approx(0.25)
    assert len(remaining[0].audio) % 2 == 0
    assert remaining[0].audio == bytes([7, 0]) * 2


def test_frames_arriving_while_paused_are_included_in_resumed_request() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        session._pause_requested = True
        session._paused_segment = avatar._SegmentState(req_id="paused")

        frame = rtc.AudioFrame(
            data=bytes([9, 0]) * 4,
            sample_rate=8,
            num_channels=1,
            samples_per_channel=4,
        )
        await session._send_audio_frame(frame)
        session._paused_segment.input_finalized = True

        session._pause_requested = False
        await session._handle_resume()

        assert provider.sent == [
            ("request-1", bytes([9, 0]) * 4, False),
            ("request-1", b"", True),
        ]

    asyncio.run(run_test())


def test_resume_buffer_is_bounded_to_configured_duration() -> None:
    session = _session()
    session._resume_buffer_max_seconds = 0.5
    segment = avatar._SegmentState(req_id="request-1")

    for value in (1, 2, 3, 4):
        session._append_audio_frame(
            segment,
            _buffered_frame(value),
            sent_to_provider=True,
        )

    assert segment.pushed_duration == pytest.approx(1.0)
    assert segment.buffered_duration == pytest.approx(0.5)
    assert segment.buffer_start_position == pytest.approx(0.5)
    assert [frame.audio for frame in segment.audio_frames] == [
        _buffered_frame(3).audio,
        _buffered_frame(4).audio,
    ]


def test_resume_before_pause_lock_avoids_unneeded_provider_interrupt() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        segment = _playing_segment()
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)

        await session._provider_io_lock.acquire()
        session._pause_requested = True
        pause_task = asyncio.create_task(session._handle_pause())
        await asyncio.sleep(0)
        session._pause_requested = False
        resume_task = asyncio.create_task(session._handle_resume())
        session._provider_io_lock.release()
        await asyncio.gather(pause_task, resume_task)

        assert provider.interrupt_calls == 0
        assert session._paused_segment is None

    asyncio.run(run_test())


def test_new_pause_supersedes_queued_resume() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._paused_segment = segment
        session._segments[segment.req_id] = segment

        await session._provider_io_lock.acquire()
        session._pause_requested = False
        resume_task = asyncio.create_task(session._handle_resume())
        await asyncio.sleep(0)
        session._pause_requested = True
        session._provider_io_lock.release()
        await resume_task

        assert provider.sent == []
        assert session._paused_segment is segment

    asyncio.run(run_test())


def test_pause_provider_failure_keeps_original_segment_authoritative() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailingInterruptProvider()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        await session._handle_pause()

        assert provider.interrupt_calls == 1
        assert session._pause_requested is False
        assert session._paused_segment is None
        assert session._segments[segment.req_id] is segment

    asyncio.run(run_test())


def test_resume_provider_failure_completes_speech_instead_of_stranding_it() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailingResumeProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        segment.playback_position_offset = 0.5
        segment.playback_started = False
        segment.playback_started_at = None
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._paused_segment = segment
        session._segments[segment.req_id] = segment

        await session._handle_resume()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(pytest.approx(0.5), True)]

    asyncio.run(run_test())


def test_pause_before_observed_playback_does_not_skip_unheard_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _playing_segment()
    segment.playback_started = False
    segment.playback_started_at = None
    segment.first_frame_at = 100.0
    monkeypatch.setattr(avatar.time, "time", lambda: 105.0)

    assert avatar.AvatarSession._estimate_interrupted_playback_position(segment) == 0.0


def test_reused_provider_request_id_is_reactivated_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        assert "request-1" in session._recently_completed_req_ids

        session._pause_requested = False
        await session._handle_resume()

        assert "request-1" not in session._recently_completed_req_ids
        assert session._segments["request-1"] is segment
        assert segment.provider_events_trusted is False

    asyncio.run(run_test())


def test_stale_completion_for_reused_request_id_cannot_finish_resumed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        session._pause_requested = False
        await session._handle_resume()
        session._on_playback_signal(PlaybackSignal(req_id="request-1", end=True))

        assert session._segments["request-1"] is segment
        assert audio.playback_finished == []

    asyncio.run(run_test())


def test_transient_provider_send_failure_does_not_kill_audio_forwarder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailFirstSendProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _StreamingAudioBufferSpy(
            [
                _rtc_frame(1),
                _rtc_frame(2),
                AudioSegmentEnd(),
                _rtc_frame(3),
            ]
        )
        session._avatarkit_session = provider
        session._audio_buffer = audio

        await session._run_main_task()

        assert provider.send_calls == 1
        assert provider.interrupt_calls == 1
        assert session._active_req_id == "request-1"
        assert audio.playback_finished == [(0.0, True)]
        assert replacement.sent == [("request-1", bytes([3, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_after_provider_recovery_releases_next_response() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        session._drop_frames_until_segment_end = True

        session._on_clear_buffer()
        await session._send_audio_frame(_rtc_frame(4))

        assert session._drop_frames_until_segment_end is False
        assert provider.sent == [("request-1", bytes([4, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_provider_recovery_does_not_discard_next_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingInterruptProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio

        recovery = asyncio.create_task(
            session._recover_provider_forwarding_error(
                item=_rtc_frame(1),
                error=RuntimeError("send unavailable"),
            )
        )
        await provider.interrupt_started.wait()

        session._on_clear_buffer()
        provider.allow_interrupt.set()
        await recovery
        await session._send_audio_frame(_rtc_frame(2))

        assert session._provider_recovery_in_flight is False
        assert session._clear_during_provider_recovery is False
        assert session._drop_frames_until_segment_end is False
        assert provider.interrupt_calls == 1
        assert replacement.sent == [("request-1", bytes([2, 0]) * 4, False)]
        assert audio.playback_finished == [(0.0, True)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_failing_provider_send_does_not_discard_next_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingFailSendProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio

        async def send_and_recover() -> None:
            frame = _rtc_frame(1)
            try:
                await session._send_audio_frame(frame)
            except Exception as error:
                await session._recover_provider_forwarding_error(item=frame, error=error)

        failed_send = asyncio.create_task(send_and_recover())
        await provider.send_started.wait()

        session._on_clear_buffer()
        provider.allow_failure.set()
        await failed_send
        await session._send_audio_frame(_rtc_frame(2))

        assert session._provider_send_in_flight is False
        assert session._clear_during_provider_send is False
        assert session._drop_frames_until_segment_end is False
        assert provider.interrupt_calls == 1
        assert replacement.sent == [("request-1", bytes([2, 0]) * 4, False)]
        assert audio.playback_finished == [(0.0, True)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_queued_frame_from_before_clear_is_not_forwarded() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider

        await session._provider_io_lock.acquire()
        stale_send = asyncio.create_task(session._send_audio_frame(_rtc_frame(1)))
        await asyncio.sleep(0)
        session._on_clear_buffer()
        session._provider_io_lock.release()

        await stale_send
        if session._background_tasks:
            await asyncio.gather(*list(session._background_tasks))
        await session._send_audio_frame(_rtc_frame(2))

        assert provider.sent == [("request-1", bytes([2, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_resume_stops_before_remaining_tail_is_sent() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingSuccessfulSendProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        segment.playback_position_offset = 0.0
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._paused_segment = segment

        resume = asyncio.create_task(session._handle_resume())
        await provider.send_started.wait()
        session._on_clear_buffer()
        provider.allow_send.set()
        await resume

        assert provider.send_calls == 1
        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(1.0, True)]

    asyncio.run(run_test())


def test_session_close_drains_lifecycle_tasks_before_final_cleanup() -> None:
    async def run_test() -> None:
        session = _session()

        async def resurrect_paused_state_on_cancel() -> None:
            try:
                await asyncio.sleep(60)
            finally:
                session._paused_segment = session._new_local_segment()

        task = session._spawn_background_task(
            resurrect_paused_state_on_cancel(),
            name="test_resurrect_paused_state",
        )

        await session.aclose()

        assert task.done()
        assert session._paused_segment is None
        assert session._segments == {}

    asyncio.run(run_test())


def test_closing_session_rejects_new_lifecycle_tasks() -> None:
    async def run_test() -> None:
        session = _session()
        started = False

        async def late_lifecycle_work() -> None:
            nonlocal started
            started = True

        session._closing = True
        task = session._spawn_background_task(
            late_lifecycle_work(),
            name="test_late_lifecycle_work",
        )
        await asyncio.sleep(0)

        assert task is None
        assert started is False
        assert session._background_tasks == set()

    asyncio.run(run_test())


def test_session_close_awaits_avatar_audio_stream_close() -> None:
    async def run_test() -> None:
        session = _session()
        stream = _AudioStreamCloseSpy()
        session._avatar_audio_stream = stream

        await session.aclose()

        assert stream.closed is True
        assert session._avatar_audio_stream is None

    asyncio.run(run_test())


def test_stale_completion_for_reused_request_id_cannot_finish_normal_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._recently_completed_req_ids.append("request-1")

        await session._send_audio_frame(_rtc_frame())
        segment = session._segments["request-1"]
        session._on_playback_signal(PlaybackSignal(req_id="request-1", end=True))

        assert segment.provider_events_trusted is False
        assert session._segments["request-1"] is segment
        assert audio.playback_finished == []

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_segment_rollover_preserves_retained_audio_state() -> None:
    target = avatar._SegmentState(req_id="request-2")
    source = _playing_segment()
    source.buffer_start_position = 0.25
    source.playback_position_offset = 0.5

    avatar.AvatarSession._merge_segment_state(target, source)

    assert target.pushed_duration == pytest.approx(1.0)
    assert target.buffered_duration == pytest.approx(1.0)
    assert target.buffer_start_position == pytest.approx(0.25)
    assert target.playback_position_offset == pytest.approx(0.5)
    assert list(target.audio_frames) == list(source.audio_frames)


# --- Provider connection recovery -------------------------------------------------------


class _Provider:
    def __init__(
        self,
        *,
        on_error=None,
        on_close=None,
        start_error: Exception | None = None,
        start_gate: asyncio.Event | None = None,
    ) -> None:
        self.on_error = on_error
        self.on_close = on_close
        self.start_error = start_error
        self.start_gate = start_gate
        self.init_calls = 0
        self.start_calls = 0
        self.close_calls = 0
        self.interrupt_calls = 0
        self.send_calls = 0

    async def init(self) -> None:
        self.init_calls += 1

    async def start(self) -> str:
        self.start_calls += 1
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        return f"connection-{self.start_calls}"

    async def close(self) -> None:
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close()

    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        raise ValueError("interrupt: websocket connection is not established")

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        return f"request-{self.send_calls}"


class _AudioBuffer:
    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.started += 1

    def notify_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
    ) -> None:
        self.finished.append((playback_position, interrupted))


def _recovery_session() -> avatar.AvatarSession:
    session = avatar.AvatarSession(api_key="test", app_id="test", avatar_id="test")
    session._livekit_egress = SimpleNamespace()
    session._resolved_sample_rate = 24000
    session._audio_buffer = _AudioBuffer()
    return session


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=240,
    )


async def test_dead_websocket_reconnects_with_fresh_provider_and_confirms_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _recovery_session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    states: list[str] = []
    session.on(
        "provider_connection_state_changed",
        lambda event: states.append(event.state),
    )

    replacements: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
        )
        replacements.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)

    await session._recover_provider_forwarding_error(
        item=_frame(),
        error=ValueError("WebSocket connection is not established"),
    )

    assert original.close_calls == 1
    assert len(replacements) == 1
    assert replacements[0].init_calls == 1
    assert replacements[0].start_calls == 1
    assert session._avatarkit_session is replacements[0]
    assert states == ["recovering", "connected"]

    await session._send_audio_frame(_frame())
    segment = next(iter(session._segments.values()))
    session._mark_playback_started(segment, source="livekit_avatar_audio_track")

    assert states == ["recovering", "connected", "recovered"]
    assert session._provider_terminal_failure is False
    await _stop_idle_end_timer(session)


async def test_provider_recovery_retries_with_bounded_fresh_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _recovery_session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_error=(RuntimeError("first reconnect failed") if not created else None),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0, 0.0))

    assert await session._ensure_provider_connection(reason="provider_closed") is True
    assert len(created) == 2
    assert created[0].start_calls == 1
    assert created[1].start_calls == 1
    assert session._avatarkit_session is created[1]


async def test_provider_recovery_times_out_hung_start_and_uses_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _recovery_session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_gate=(asyncio.Event() if not created else None),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(avatar, "PROVIDER_CONNECT_TIMEOUT_SECONDS", 0.01)

    assert await session._ensure_provider_connection(reason="provider_closed") is True
    assert len(created) == 2
    assert created[0].close_calls == 1
    assert created[1].start_calls == 1
    assert session._avatarkit_session is created[1]


async def test_provider_recovery_exhaustion_emits_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _recovery_session()
    session._avatarkit_session = _Provider()
    session._provider_generation = 1
    session._provider_session_generation = 1
    states: list[str] = []
    session.on(
        "provider_connection_state_changed",
        lambda event: states.append(event.state),
    )

    def factory(**kwargs):
        return _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_error=RuntimeError("provider unavailable"),
        )

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0, 0.0))

    assert await session._ensure_provider_connection(reason="provider_closed") is False
    assert states == ["recovering", "failed"]
    assert session._provider_terminal_failure is True


async def test_unexpected_provider_close_schedules_single_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _recovery_session()
    session._avatarkit_session = _Provider()
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0,))

    session._on_provider_close(1)
    first_task = session._provider_reconnect_task
    session._on_provider_close(1)

    assert first_task is not None
    assert session._provider_reconnect_task is first_task
    assert await first_task is True
    assert len(created) == 1


# --- Review fixes -------------------------------------------------------------------------


class _ConnectedRoom:
    name = "room-1"

    def __init__(self) -> None:
        self.off_calls: list[str] = []

    def isconnected(self) -> bool:
        return True

    def off(self, event: str, *_: object) -> None:
        self.off_calls.append(event)


async def test_aclose_lets_the_base_remove_the_avatar_from_a_live_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base aclose() removes the avatar participant only while it can still see the room."""
    from livekit.agents.voice.avatar import _types as base_avatar

    removed: list[object] = []

    async def remove_participant(request: object) -> None:
        removed.append(request)

    job_ctx = SimpleNamespace(
        api=SimpleNamespace(room=SimpleNamespace(remove_participant=remove_participant))
    )
    monkeypatch.setattr(base_avatar, "get_job_context", lambda required=False: job_ctx)

    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    room = _ConnectedRoom()
    session._room = room  # type: ignore[assignment]

    await session.aclose()

    assert len(removed) == 1, "the avatar participant was left in the room"
    assert removed[0].identity == session.avatar_identity
    assert "connection_state_changed" in room.off_calls
    assert session._room is None


class _SuspendingAvatarkitSession(FakeAvatarkitSession):
    """A send that yields to the loop, like a real websocket write."""

    async def send_audio(self, audio: bytes, end: bool = False) -> str:
        await asyncio.sleep(0)
        return await super().send_audio(audio, end)


async def test_missing_end_marker_fallback_finalizes_the_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the segment end never arrives, the idle fallback sends end=True and queues the segment."""
    monkeypatch.setattr(avatar, "ACTIVE_SEGMENT_IDLE_END_SECONDS", 0.01)
    session, _, buffer = make_session()
    fake = _SuspendingAvatarkitSession()
    session._avatarkit_session = fake  # type: ignore[assignment]

    await session._send_audio_frame(make_frame())
    idle_end = session._active_segment_idle_end_task
    assert idle_end is not None
    await asyncio.gather(idle_end, return_exceptions=True)

    assert not idle_end.cancelled(), "the fallback cancelled itself"
    assert fake.sent[-1] == (0, True), fake.sent
    assert session._active_req_id is None
    assert fake.req_id in session._pending_segment_ids

    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started", ("finished", False)], buffer.events


class _BlockingSendProvider(_ProviderSpy):
    """Holds the second send open so a clear can land while it is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.sends = 0
        self.send_started = asyncio.Event()
        self.allow_send = asyncio.Event()

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.sends += 1
        if self.sends == 2:
            self.send_started.set()
            await self.allow_send.wait()
        return await super().send_audio(audio=audio, end=end)


class _QueuedAudioBuffer(_AudioBufferSpy):
    def __init__(self) -> None:
        super().__init__()
        self.items: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()

    def __aiter__(self) -> _QueuedAudioBuffer:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        return await self.items.get()


async def test_barge_in_during_a_send_interrupts_without_reconnecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user interruption that lands mid-send is a normal interruption, not a provider failure."""
    reconnects: list[object] = []
    monkeypatch.setattr(avatar, "new_avatar_session", lambda **kw: reconnects.append(kw))

    session = _session()
    provider = _BlockingSendProvider()
    audio = _QueuedAudioBuffer()
    session._avatarkit_session = provider
    session._audio_buffer = audio  # type: ignore[assignment]
    forwarder = asyncio.create_task(session._run_main_task())
    try:
        audio.items.put_nowait(_rtc_frame(1))
        audio.items.put_nowait(_rtc_frame(2))
        await provider.send_started.wait()

        session._on_clear_buffer()
        provider.allow_send.set()
        for _ in range(20):
            await asyncio.sleep(0)

        assert reconnects == [], "a barge-in reconnected the provider"
        assert session._avatarkit_session is provider
        assert provider.interrupt_calls == 1
        assert audio.playback_finished and audio.playback_finished[-1][1] is True
        assert not session._segments and session._active_req_id is None

        audio.items.put_nowait(_rtc_frame(3))
        for _ in range(20):
            await asyncio.sleep(0)
        assert provider.sent[-1] == ("request-2", bytes([3, 0]) * 4, False), provider.sent
    finally:
        await _stop_idle_end_timer(session)
        forwarder.cancel()
        await asyncio.gather(forwarder, return_exceptions=True)
        await asyncio.gather(*list(session._background_tasks), return_exceptions=True)


class _PositionBuffer(FakeBuffer):
    def notify_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        self.events.append(("finished", round(playback_position, 3), interrupted))


async def test_a_reply_split_across_requests_finishes_once() -> None:
    """A req_id rollover mid-reply must not report the reply finished when the first request ends."""
    session, fake, _ = make_session()
    buffer = _PositionBuffer()
    session._audio_buffer = buffer  # type: ignore[assignment]

    await session._send_audio_frame(make_frame())  # 10 ms on req-1
    fake.req_id = "req-2"
    await session._send_audio_frame(make_frame())  # rolls over to req-2
    await session._send_audio_frame(make_frame())

    session._on_playback_signal(provider_end_signal("req-1"))
    assert "finished" not in [e[0] for e in buffer.events if isinstance(e, tuple)], buffer.events

    assert await session._finalize_active_segment(source="segment_end")
    session._on_playback_signal(provider_end_signal("req-2"))
    finished = [e for e in buffer.events if isinstance(e, tuple)]
    assert finished == [("finished", 0.03, False)], buffer.events


async def test_an_interrupted_split_reply_finishes_once() -> None:
    session, fake, _ = make_session()
    buffer = _PositionBuffer()
    session._audio_buffer = buffer  # type: ignore[assignment]

    await session._send_audio_frame(make_frame())
    fake.req_id = "req-2"
    await session._send_audio_frame(make_frame())

    session._on_clear_buffer()
    await _drain(session)
    finished = [e for e in buffer.events if isinstance(e, tuple)]
    assert len(finished) == 1 and finished[0][2] is True, buffer.events


async def test_a_reply_restored_by_an_empty_resume_finishes_once_after_rollover() -> None:
    """The segment an empty resume puts back in play still belongs to the same reply."""
    import time

    session, fake, _ = make_session()
    buffer = _PositionBuffer()
    session._audio_buffer = buffer  # type: ignore[assignment]

    await session._send_audio_frame(make_frame())
    segment = session._segments[fake.req_id]
    session._mark_playback_started(segment, source="test")
    segment.playback_started_at = time.time() - 5.0  # everything sent has played

    session._on_pause()
    await _drain(session)
    session._on_resume()
    await _drain(session)
    assert session._segments.get("req-1") is segment, "the empty resume should restore the segment"

    fake.req_id = "req-2"
    await session._send_audio_frame(make_frame())
    session._on_playback_signal(provider_end_signal("req-1"))
    assert [e for e in buffer.events if isinstance(e, tuple)] == [], buffer.events

    assert await session._finalize_active_segment(source="segment_end")
    session._on_playback_signal(provider_end_signal("req-2"))
    # req-1 was interrupted by the pause, so its late end event is ignored and the
    # restored request completes on its duration fallback; the reply is still open
    assert [e for e in buffer.events if isinstance(e, tuple)] == [], buffer.events

    await session._wait_for_segment_completion_timeout("req-1", 0)
    finished = [e for e in buffer.events if isinstance(e, tuple)]
    assert finished == [("finished", 0.02, False)], buffer.events
    assert buffer.events.count("started") == 1, buffer.events
    await asyncio.sleep(0)  # let the cancelled completion timers unwind
