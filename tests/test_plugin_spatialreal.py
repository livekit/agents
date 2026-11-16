"""SpatialReal avatar plugin: reply lifecycle, interruption, pause/resume and recovery.

Hermetic: the SpatialReal SDK session and the framework audio buffer are stubbed, and the
plugin is driven through the same queue and callbacks the framework uses, so no network or
LiveKit room is needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from spatialreal import InterruptReason, PlaybackSignal, PlaybackState, PlaybackStateEvent

from livekit import api, rtc
from livekit.agents.voice import io as voice_io
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput
from livekit.plugins.spatialreal import avatar
from livekit.plugins.spatialreal.avatar import AvatarSession
from livekit.plugins.spatialreal.resumable_queue_io import ResumableQueueAudioOutput

pytestmark = pytest.mark.unit


# --- fakes --------------------------------------------------------------------------


class FakeProvider:
    """Mimics the SDK session: a fresh request id per reply, reset by end or interrupt."""

    def __init__(self, name: str = "p1", capabilities: tuple[str, ...] = ("playback_control",)):
        self.name = name
        self.capabilities = capabilities
        self.calls: list[tuple] = []
        self.closed = False
        self._seq = 0
        self._current: str | None = None
        self._last: str | None = None
        self.fail_sends = False
        self.send_gate: asyncio.Event | None = None

    async def send_audio(self, audio: bytes, end: bool = False) -> str:
        if self.send_gate is not None:
            gate, self.send_gate = self.send_gate, None
            await gate.wait()
        if self.fail_sends or self.closed:
            raise ValueError("WebSocket connection is not established")
        if self._current is None:
            self._seq += 1
            self._current = self._last = f"{self.name}-req-{self._seq}"
        req_id = self._current
        self.calls.append(("end" if end else "audio", req_id))
        if end:
            self._current = None
        return req_id

    async def interrupt(self) -> str:
        self.calls.append(("interrupt", self._last))
        self._current = None
        return self._last or ""

    async def pause(self) -> str:
        self.calls.append(("pause", self._last))
        return self._last or ""

    async def resume(self) -> str:
        self.calls.append(("resume", self._last))
        return self._last or ""

    async def init(self) -> None:
        pass

    async def start(self) -> str:
        return "connection"

    async def close(self) -> None:
        self.closed = True

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


class FakeBuffer:
    """The framework side: a queue the plugin reads, and the notifications it receives."""

    def __init__(self) -> None:
        self.items: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self.events: list[tuple] = []

    def __aiter__(self) -> FakeBuffer:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        return await self.items.get()

    def notify_playback_started(self) -> None:
        self.events.append(("started",))

    def notify_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        self.events.append(("finished", round(playback_position, 3), interrupted))

    _pending_playback_count = 0

    def finishes(self) -> list[bool]:
        return [event[2] for event in self.events if event[0] == "finished"]

    async def aclose(self) -> None:
        pass


def frame(duration: float = 0.1) -> rtc.AudioFrame:
    sample_rate = 1000
    samples = int(sample_rate * duration)
    return rtc.AudioFrame(
        data=b"\x01\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def settle(rounds: int = 30) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


class Harness:
    def __init__(self, provider: FakeProvider | None = None) -> None:
        self.session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
        self.provider = provider or FakeProvider()
        self.buffer = FakeBuffer()
        self.session._provider = self.provider  # type: ignore[assignment]
        self.session._generation = 1
        self.session._playback_control = "playback_control" in self.provider.capabilities
        self.session._audio_buffer = self.buffer  # type: ignore[assignment]
        self.forwarder = asyncio.create_task(self.session._forward_audio())

    async def reply(self, frames: int = 2, *, end: bool = True) -> None:
        for _ in range(frames):
            self.buffer.items.put_nowait(frame())
        if end:
            self.buffer.items.put_nowait(AudioSegmentEnd())
        await settle()

    def signal(self, req_id: str, *, end: bool = False, generation: int | None = None) -> None:
        self.session._on_playback_signal(
            PlaybackSignal(req_id=req_id, end=end),
            self.session._generation if generation is None else generation,
        )

    async def close(self) -> None:
        self.forwarder.cancel()
        await asyncio.gather(self.forwarder, return_exceptions=True)
        tasks = list(self.session._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
async def h():
    harness = Harness()
    yield harness
    await harness.close()


# --- a normal reply -----------------------------------------------------------------


async def test_a_reply_is_one_request_and_finishes_when_the_server_says_so(h: Harness) -> None:
    await h.reply(frames=3)
    assert h.provider.calls == [
        ("audio", "p1-req-1"),
        ("audio", "p1-req-1"),
        ("audio", "p1-req-1"),
        ("end", "p1-req-1"),
    ]
    assert h.buffer.events == []

    h.signal("p1-req-1")
    assert h.buffer.events == [("started",)]
    h.signal("p1-req-1", end=True)
    assert h.buffer.events == [("started",), ("finished", 0.3, False)]


async def test_repeated_or_unknown_server_events_are_ignored(h: Harness) -> None:
    await h.reply()
    h.signal("p1-req-1", end=True)
    h.signal("p1-req-1", end=True)  # the egress can repeat end=true
    h.signal("someone-else", end=True)
    h.signal("p1-req-1", end=True, generation=0)  # from a previous connection
    assert h.buffer.finishes() == [False]


async def test_consecutive_replies_get_their_own_requests(h: Harness) -> None:
    await h.reply()
    await h.reply()
    h.signal("p1-req-2", end=True)
    h.signal("p1-req-1", end=True)
    assert h.buffer.finishes() == [False, False]
    assert ("end", "p1-req-2") in h.provider.calls


async def test_a_reply_without_an_end_report_is_finished_after_it_should_have_played(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(avatar, "COMPLETION_MARGIN_SECONDS", 0.0)
    monkeypatch.setattr(avatar, "START_ALLOWANCE_SECONDS", 0.0)
    await h.reply(frames=1)
    h.signal("p1-req-1")
    await asyncio.sleep(0.15)
    assert h.buffer.events == [("started",), ("finished", 0.1, False)]


# --- interruption -------------------------------------------------------------------


async def test_an_interruption_stops_the_server_and_finishes_the_reply_once(h: Harness) -> None:
    await h.reply(end=False)
    h.signal("p1-req-1")
    h.session._on_clear_buffer()
    h.session._on_clear_buffer()  # LiveKit can clear twice
    await settle()

    assert h.provider.kinds().count("interrupt") == 1
    assert h.buffer.finishes() == [True]
    assert not h.session._replies

    await h.reply()  # the next reply is a new request
    assert h.provider.calls[-1] == ("end", "p1-req-2")


async def test_an_interruption_with_nothing_sent_does_not_touch_the_server(h: Harness) -> None:
    h.session._on_clear_buffer()
    await settle()
    assert h.provider.calls == []


async def test_an_interruption_during_a_send_lands_after_it_and_before_the_next_reply(
    h: Harness,
) -> None:
    gate = asyncio.Event()
    h.provider.send_gate = gate
    h.buffer.items.put_nowait(frame())
    await settle()  # the send is in flight

    h.session._on_clear_buffer()
    h.buffer.items.put_nowait(frame())  # first frame of the next reply
    gate.set()
    await settle()

    assert h.provider.calls == [
        ("audio", "p1-req-1"),
        ("interrupt", "p1-req-1"),
        ("audio", "p1-req-2"),
    ]
    assert h.buffer.finishes() == [True]
    assert h.session._provider is h.provider, "an interruption is not a provider failure"


# --- pause and resume ---------------------------------------------------------------


async def test_pause_and_resume_hold_the_reply_on_the_server(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(avatar.time, "time", lambda: clock[0])
    await h.reply(frames=10, end=False)
    h.signal("p1-req-1")

    clock[0] = 100.2
    h.session._on_pause()
    await settle()
    h.buffer.items.put_nowait(frame())  # audio keeps flowing while held
    await settle()
    clock[0] = 103.2
    h.session._on_resume()
    await settle()

    assert h.provider.kinds() == ["audio"] * 10 + ["pause", "audio", "resume"]
    clock[0] = 103.5
    h.session._on_clear_buffer()
    await settle()
    # 0.5s played: the 3s held does not count
    assert h.buffer.events[-1] == ("finished", 0.5, True)


async def test_a_pause_before_any_audio_holds_the_reply_from_its_first_frame(h: Harness) -> None:
    h.session._on_pause()
    await settle()
    assert h.provider.calls == []

    await h.reply(frames=2, end=False)
    assert h.provider.kinds() == ["audio", "pause", "audio"]
    assert h.provider.calls[1] == ("pause", "p1-req-1")


async def test_without_server_support_pause_is_not_offered() -> None:
    output = ResumableQueueAudioOutput(sample_rate=24000, can_pause=False)
    assert output.can_pause is False
    output = ResumableQueueAudioOutput(sample_rate=24000, can_pause=True)
    assert output.can_pause is True

    events: list[str] = []
    for name in ("pause", "resume", "clear_buffer"):
        output.on(name, lambda name=name: events.append(name))
    output.pause()
    output.pause()
    output.resume()
    output.resume()
    output.pause()
    output.clear_buffer()
    output.resume()  # a clear already ended the pause
    assert events == ["pause", "resume", "pause", "clear_buffer"]


async def test_a_server_pause_timeout_finishes_the_held_reply_and_drops_its_rest(
    h: Harness,
) -> None:
    await h.reply(end=False)
    h.session._on_pause()
    await settle()

    timeout = PlaybackStateEvent(
        req_id="p1-req-1", state=PlaybackState.INTERRUPTED, reason=InterruptReason.PAUSE_TIMEOUT
    )
    h.session._on_playback_state(timeout, 1)
    assert h.buffer.finishes() == [True]

    await h.reply(frames=2)  # the rest of the timed-out reply
    assert h.provider.kinds().count("audio") == 2
    await h.reply(frames=1)  # the next reply plays
    assert h.provider.calls[-1] == ("end", "p1-req-2")


async def test_a_stale_pause_timeout_does_not_cut_off_the_next_reply(h: Harness) -> None:
    await h.reply(end=False)
    h.session._on_clear_buffer()
    await settle()
    await h.reply(end=False)

    def timeout(req_id: str) -> PlaybackStateEvent:
        return PlaybackStateEvent(
            req_id=req_id, state=PlaybackState.INTERRUPTED, reason=InterruptReason.PAUSE_TIMEOUT
        )

    h.session._on_playback_state(timeout("p1-req-1"), 1)  # the interrupted reply
    h.session._on_playback_state(timeout("p1-req-2"), 0)  # a previous connection
    assert h.buffer.finishes() == [True]
    assert len(h.session._replies) == 1


# --- provider failure and recovery --------------------------------------------------


def use_replacements(monkeypatch: pytest.MonkeyPatch, *providers: FakeProvider) -> list:
    """Make reconnects create the given providers (an exception = a failed attempt)."""
    queue = list(providers)
    created: list = []

    def factory(**kwargs):
        provider = queue.pop(0) if queue else FakeProvider("spare")
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0, 0.0))
    return created


class FailingStart(FakeProvider):
    async def start(self) -> str:
        raise RuntimeError("provider unavailable")


def prepare_reconnect(h: Harness) -> None:
    h.session._livekit_egress = SimpleNamespace()  # type: ignore[assignment]
    h.session._resolved_sample_rate = 24000


async def test_a_lost_connection_finishes_the_reply_drops_its_rest_and_reconnects(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    replacement = FakeProvider("p2")
    use_replacements(monkeypatch, replacement)
    states: list[str] = []
    h.session.on("provider_connection_state_changed", lambda ev: states.append(ev.state))

    await h.reply(end=False)
    h.session._on_provider_lost(1, "provider_closed")
    await settle()
    assert h.buffer.finishes() == [True]
    assert h.session._provider is replacement
    assert h.provider.closed
    assert states == ["recovering", "recovered"]

    await h.reply(frames=2)  # the rest of the reply that was cut off
    assert replacement.calls == []
    await h.reply(frames=1)
    assert replacement.calls == [("audio", "p2-req-1"), ("end", "p2-req-1")]


async def test_a_reply_during_a_reconnect_waits_for_the_new_connection(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    gate = asyncio.Event()

    class SlowStart(FakeProvider):
        async def start(self) -> str:
            await gate.wait()
            return "connection"

    replacement = SlowStart("p2")
    use_replacements(monkeypatch, replacement)

    h.session._on_provider_lost(1, "provider_closed")
    await h.reply(frames=1)
    assert replacement.calls == [] and h.buffer.events == []

    gate.set()
    await settle()
    assert replacement.calls == [("audio", "p2-req-1"), ("end", "p2-req-1")]
    h.signal("p2-req-1", end=True, generation=h.session._generation)
    assert h.buffer.finishes() == [False]


async def test_after_recovery_fails_every_reply_still_finishes_once(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    use_replacements(monkeypatch, FailingStart("x"), FailingStart("y"), FailingStart("z"))
    states: list[str] = []
    h.session.on("provider_connection_state_changed", lambda ev: states.append(ev.state))

    h.session._on_provider_lost(1, "provider_closed")
    await h.reply(frames=2)
    await h.reply(frames=2)

    assert states == ["recovering", "failed"]
    assert h.buffer.finishes() == [True, True]
    assert h.buffer.events.count(("started",)) == 2
    assert not h.session._drop_until_segment_end


async def test_a_failed_send_finishes_the_reply_and_replaces_the_connection(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    replacement = FakeProvider("p2")
    use_replacements(monkeypatch, replacement)
    h.provider.fail_sends = True

    await h.reply(frames=2)
    assert h.buffer.finishes() == [True]
    assert h.session._provider is replacement

    await h.reply(frames=1)
    assert replacement.calls == [("audio", "p2-req-1"), ("end", "p2-req-1")]


async def test_a_late_event_from_the_lost_connection_is_ignored(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    use_replacements(monkeypatch, FakeProvider("p2"))
    await h.reply()
    h.session._on_provider_lost(1, "provider_error")
    h.session._on_provider_lost(1, "provider_closed")  # the same loss, reported twice
    await settle()
    h.signal("p1-req-1", end=True, generation=1)
    assert h.buffer.finishes() == [True]


async def test_a_frame_held_through_a_failed_reconnect_does_not_swallow_the_next_reply(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    gate = asyncio.Event()

    class GatedFailure(FakeProvider):
        async def start(self) -> str:
            await gate.wait()
            raise RuntimeError("provider unavailable")

    use_replacements(monkeypatch, GatedFailure("x"), FailingStart("y"), FailingStart("z"))
    h.session._on_provider_lost(1, "provider_closed")
    h.buffer.items.put_nowait(frame())  # reply A: the forwarder holds it, waiting
    await settle()
    h.session._on_clear_buffer()  # the user interrupts A meanwhile
    gate.set()  # and the reconnect fails
    await settle()

    await h.reply(frames=2)  # reply B
    assert h.buffer.finishes() == [True, True], "reply B was never finished"
    assert not h.session._drop_until_segment_end


async def test_a_stale_connection_reference_never_tears_down_its_replacement(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_reconnect(h)
    replacement = FakeProvider("p2")
    created = use_replacements(monkeypatch, replacement)

    await h.session._io_lock.acquire()  # e.g. a pause in flight on the old connection
    h.buffer.items.put_nowait(frame())
    await settle()  # the forwarder took p1 and waits for the lock
    h.session._on_provider_lost(1, "provider_closed")
    await settle()
    assert h.session._provider is replacement
    h.session._io_lock.release()
    await settle()

    # the held frame belongs to the reply the loss cut off, so it is dropped, and the
    # new connection is left alone
    assert replacement.calls == []
    assert not replacement.closed and created == [replacement]
    assert h.session._provider is replacement

    await h.reply(frames=0)  # the end of the cut-off reply
    await h.reply(frames=1)
    assert replacement.calls == [("audio", "p2-req-1"), ("end", "p2-req-1")]


async def test_a_reply_cleared_before_its_first_frame_was_read_still_finishes() -> None:
    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    session._provider = FakeProvider()  # type: ignore[assignment]
    session._generation = 1
    output = ResumableQueueAudioOutput(sample_rate=1000, wait_playback_start=True)
    output.on("clear_buffer", session._on_clear_buffer)
    session._audio_buffer = output
    forwarder = asyncio.create_task(session._forward_audio())
    await settle()
    try:
        await output.capture_frame(frame())  # queued; the forwarder has not run yet
        output.flush()
        output.clear_buffer()
        event = await asyncio.wait_for(output.wait_for_playout(), 1)
        assert event.interrupted
    finally:
        forwarder.cancel()
        await asyncio.gather(forwarder, *session._background_tasks, return_exceptions=True)


# --- session setup and teardown -----------------------------------------------------


class _Leaf(voice_io.AudioOutput):
    def __init__(self, next_in_chain: voice_io.AudioOutput | None = None) -> None:
        super().__init__(
            label="leaf",
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


def _output(audio: voice_io.AudioOutput) -> voice_io.AgentOutput:
    output = voice_io.AgentOutput(
        video_changed=lambda: None,
        audio_changed=lambda: None,
        transcription_changed=lambda: None,
    )
    output.audio = audio
    return output


@pytest.mark.parametrize("wrapped", [True, False])
async def test_close_restores_the_original_audio_output(wrapped: bool) -> None:
    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    leaf = _Leaf()
    top = _Leaf(next_in_chain=leaf) if wrapped else leaf
    output = _output(top)
    queue = QueueAudioOutput(sample_rate=24000)
    session._audio_buffer = queue  # type: ignore[assignment]
    session._agent_session = SimpleNamespace(output=output)  # type: ignore[assignment]
    session._original_audio_output = output.audio
    session._original_audio_tail = AvatarSession._resolve_audio_tail(output.audio)

    output.replace_audio_tail(queue)
    assert AvatarSession._resolve_audio_tail(output.audio) is queue
    session._detach_audio_output()
    assert output.audio is top
    assert AvatarSession._resolve_audio_tail(output.audio) is leaf


class _ConnectedRoom:
    name = "room-1"

    def __init__(self) -> None:
        self.off_calls: list[str] = []

    def isconnected(self) -> bool:
        return True

    def off(self, event: str, *_: object) -> None:
        self.off_calls.append(event)


async def test_close_lets_the_base_remove_the_avatar_from_a_live_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


async def test_close_finishes_what_is_still_playing(h: Harness) -> None:
    await h.reply()
    await h.session.aclose()
    assert h.buffer.finishes() == [True]
    assert h.provider.closed


def _claims(jwt: str) -> dict:
    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


async def test_the_egress_token_outlives_the_call_and_lets_the_egress_publish() -> None:
    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    assert session._livekit_token_ttl >= timedelta(hours=12)
    override = AvatarSession(
        api_key="k", app_id="a", avatar_id="av", livekit_token_ttl_seconds=7200
    )
    assert override._livekit_token_ttl == timedelta(seconds=7200)

    jwt = (
        api.AccessToken(api_key="devkey", api_secret="secret" * 6)
        .with_identity("spatialreal-avatar")
        .with_ttl(session._livekit_token_ttl)
        .with_grants(avatar._egress_video_grants("room-1"))
        .to_jwt()
    )
    claims = _claims(jwt)
    assert claims["exp"] - claims["nbf"] >= 12 * 3600
    assert claims["video"]["room"] == "room-1"
    assert claims["video"]["canPublishData"] is True
    assert claims["video"]["canSubscribe"] is False


async def test_the_session_outlives_the_call_within_the_server_cap() -> None:
    assert timedelta(hours=12) <= avatar.DEFAULT_SESSION_TTL < timedelta(hours=24)


async def test_credentials_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in ("SPATIALREAL_API_KEY", "SPATIALREAL_APP_ID", "SPATIALREAL_AVATAR_ID"):
        monkeypatch.delenv(env, raising=False)
    with pytest.raises(avatar.SpatialRealException, match="api_key"):
        AvatarSession(app_id="a", avatar_id="av")
    monkeypatch.setenv("SPATIALREAL_API_KEY", "k")
    monkeypatch.setenv("SPATIALREAL_APP_ID", "a")
    monkeypatch.setenv("SPATIALREAL_AVATAR_ID", "av")
    AvatarSession()
