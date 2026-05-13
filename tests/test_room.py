"""Integration tests for wait_for_participant, wait_for_track_publication, and wait_for_agent.

These tests require a running LiveKit server (bootstrapped via the livekit_server fixture).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import pytest

from livekit import api, rtc
from livekit.agents.utils.participant import (
    wait_for_agent,
    wait_for_participant,
    wait_for_track_publication,
)
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

from .lk_server import LK_API_KEY, LK_API_SECRET, LK_URL, livekit_server  # noqa: F401
from .utils.audio_test import AudioEnergyMonitor, SineToneSource
from .utils.livekit_test import (
    connect_room as _connect_e2e_room,
    make_room_name as _e2e_room_name,
    simulate_full_reconnect,
    simulate_resume,
    wait_for_event,
)

TIMEOUT = 5.0


def _make_token(
    identity: str,
    room: str,
    *,
    kind: str | None = None,
    agent: bool = False,
) -> str:
    token = (
        api.AccessToken(LK_API_KEY, LK_API_SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                agent=agent,
            )
        )
    )
    if kind is not None:
        token = token.with_kind(kind)
    return token.to_jwt()


@asynccontextmanager
async def connect_room(
    identity: str, room_name: str, *, kind: str | None = None, agent: bool = False
):
    room = rtc.Room()
    token = _make_token(identity, room_name, kind=kind, agent=agent)
    await room.connect(LK_URL, token)
    try:
        yield room
    finally:
        await room.disconnect()


def _room_name() -> str:
    return f"test-{uuid.uuid4()}"


async def _publish_audio_track(room: rtc.Room) -> None:
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track)


async def _test_wait_disconnect(wait_fn: Callable[[rtc.Room], Awaitable[object]]) -> None:
    """Connect a room, start waiting, disconnect, and assert RuntimeError."""
    name = _room_name()
    room = rtc.Room()
    token = _make_token("observer", name)
    await room.connect(LK_URL, token)

    task = asyncio.ensure_future(wait_fn(room))
    # ensure the task is running and has registered its event handlers
    await asyncio.sleep(0.5)
    await room.disconnect()

    with pytest.raises(RuntimeError, match="disconnected"):
        await asyncio.wait_for(task, timeout=TIMEOUT)


# -- wait_for_participant tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForParticipant:
    async def test_joins_after(self):
        """Room A waits, Room B joins afterwards -> returns B."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_participant(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("joiner", name):
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.identity == "joiner"

    async def test_already_joined(self):
        """Room B already present when A calls wait -> returns immediately."""
        name = _room_name()
        async with connect_room("first", name):
            await asyncio.sleep(0.5)
            async with connect_room("observer", name) as room_a:
                await asyncio.sleep(0.5)
                result = await asyncio.wait_for(wait_for_participant(room_a), timeout=TIMEOUT)
                assert result.identity == "first"

    async def test_with_identity(self):
        """Only returns participant matching specific identity."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_participant(room_a, identity="target"))
            await asyncio.sleep(0.3)

            # Join with wrong identity first
            async with connect_room("decoy", name):
                await asyncio.sleep(0.3)
                assert not task.done()

                # Now join with the target identity
                async with connect_room("target", name):
                    result = await asyncio.wait_for(task, timeout=TIMEOUT)
                    assert result.identity == "target"

    async def test_with_kind(self):
        """Standard participant doesn't match; agent participant does."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(
                wait_for_participant(
                    room_a,
                    kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                )
            )
            await asyncio.sleep(0.3)

            # Standard participant should not match
            async with connect_room("standard", name):
                await asyncio.sleep(0.3)
                assert not task.done()

                # Agent participant should match
                async with connect_room("agent-p", name, kind="agent", agent=True):
                    result = await asyncio.wait_for(task, timeout=TIMEOUT)
                    assert result.identity == "agent-p"

    async def test_disconnect(self):
        """Room disconnects while waiting -> RuntimeError raised."""
        await _test_wait_disconnect(wait_for_participant)


# -- wait_for_track_publication tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForTrackPublication:
    async def test_published_after(self):
        """Observer waits, publisher publishes audio track -> returns publication."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_track_publication(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("publisher", name) as room_b:
                await _publish_audio_track(room_b)
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.kind == rtc.TrackKind.KIND_AUDIO

    async def test_already_published(self):
        """Track already published before wait -> returns immediately."""
        name = _room_name()
        async with connect_room("publisher", name) as room_b:
            await _publish_audio_track(room_b)
            await asyncio.sleep(0.5)

            async with connect_room("observer", name) as room_a:
                await asyncio.sleep(0.5)
                result = await asyncio.wait_for(wait_for_track_publication(room_a), timeout=TIMEOUT)
                assert result.kind == rtc.TrackKind.KIND_AUDIO

    async def test_disconnect(self):
        """Room disconnects while waiting -> RuntimeError raised."""
        await _test_wait_disconnect(wait_for_track_publication)


# -- wait_for_agent tests --


@pytest.mark.usefixtures("livekit_server")
class TestWaitForAgent:
    async def test_agent_joins(self):
        """Agent-kind participant detected correctly."""
        name = _room_name()
        async with connect_room("observer", name) as room_a:
            task = asyncio.ensure_future(wait_for_agent(room_a))
            await asyncio.sleep(0.5)

            async with connect_room("my-agent", name, kind="agent", agent=True):
                result = await asyncio.wait_for(task, timeout=TIMEOUT)
                assert result.identity == "my-agent"
                assert result.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


# -- Reconnect E2E tests --
#
# exercise resume vs full-reconnect behavior end-to-end against a
# real LiveKit server.

_AGENT_IDENTITY = "agent"
_USER_IDENTITY = "user"
_TONE_NAME = "agent_tone"
# Threshold tuned to discriminate the steady sine wave from server-side
# silence/noise. The 440Hz tone at 0.5 amplitude lands well above 0.1.
_AUDIO_RMS_THRESHOLD = 0.05


@contextlib.asynccontextmanager
async def _agent_publishing_tone(agent_room: rtc.Room):
    """Publish a steady sine tone via the production `_ParticipantAudioOutput`
    helper, so the test exercises both the SDK and the agents-framework layer."""
    tone = SineToneSource(frequency=440.0, amplitude=0.5)
    output = _ParticipantAudioOutput(
        agent_room,
        sample_rate=tone._sample_rate,
        num_channels=1,
        track_publish_options=rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        track_name=_TONE_NAME,
    )
    # Replace the helper's internal AudioSource with our SineToneSource so the
    # published track carries the known signal. _ParticipantAudioOutput
    # constructs its own AudioSource in __init__; swap it before .start() so
    # the LocalAudioTrack is built against ours.
    await output._audio_source.aclose()
    output._audio_source = tone.source
    await output.start()
    await tone.start()
    try:
        yield output, tone
    finally:
        await tone.aclose()
        await output.aclose()


def _agent_audio_publications(user_room: rtc.Room) -> list[rtc.RemoteTrackPublication]:
    """All audio publications the user sees from the agent."""
    agent = user_room.remote_participants.get(_AGENT_IDENTITY)
    if agent is None:
        return []
    return [
        pub for pub in agent.track_publications.values() if pub.kind == rtc.TrackKind.KIND_AUDIO
    ]


async def _await_subscribed_audio(user_room: rtc.Room) -> rtc.RemoteAudioTrack:
    pub = await asyncio.wait_for(
        wait_for_track_publication(
            user_room,
            identity=_AGENT_IDENTITY,
            kind=rtc.TrackKind.KIND_AUDIO,
            wait_for_subscription=True,
        ),
        timeout=15.0,
    )
    track = pub.track
    assert isinstance(track, rtc.RemoteAudioTrack)
    return track


async def _wait_back_to_connected(room: rtc.Room, *, timeout: float = 15.0) -> None:
    await wait_for_event(
        room,
        "connection_state_changed",
        timeout=timeout,
        predicate=lambda state: state == rtc.ConnectionState.CONN_CONNECTED,
    )


@pytest.mark.skipif(
    not os.environ.get("LIVEKIT_URL"),
    reason="LIVEKIT_URL not set; skipping reconnect E2E tests "
    "(set LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET to enable)",
)
@pytest.mark.skipif(
    not hasattr(rtc, "SimulateScenarioKind"),
    reason="livekit-rtc lacks SimulateScenarioKind; run `make link-rtc-local`",
)
class TestReconnect:
    async def test_resume_preserves_publication_and_audio(self):
        """Resume must NOT churn the publication set, should fire
        `reconnected`, and audio must keep flowing."""
        room_name = _e2e_room_name("resume")

        async with (
            _connect_e2e_room(_USER_IDENTITY, room_name) as user_room,
            _connect_e2e_room(_AGENT_IDENTITY, room_name, agent=True) as agent_room,
            _agent_publishing_tone(agent_room) as (_output, _tone),
        ):
            await asyncio.wait_for(
                wait_for_participant(user_room, identity=_AGENT_IDENTITY),
                timeout=10.0,
            )
            track = await _await_subscribed_audio(user_room)

            async with AudioEnergyMonitor.watch(track) as mon:
                await mon.wait_for_audio(min_rms=_AUDIO_RMS_THRESHOLD, timeout=5.0)

                # Snapshot publication state before the disturbance.
                publications_before = _agent_audio_publications(user_room)
                assert len(publications_before) == 1, publications_before
                sid_before = publications_before[0].sid

                reconnected_fired = asyncio.Event()
                agent_room.on("reconnected", lambda: reconnected_fired.set())

                await simulate_resume(agent_room)

                # Engine cycles through Reconnecting -> Connected; wait for
                # the second transition.
                await _wait_back_to_connected(agent_room, timeout=15.0)

                # Brief grace window for any Reconnected dispatch.
                await asyncio.sleep(1.0)
                assert reconnected_fired.is_set(), (
                    "RoomEvent::Reconnected should be fired on a resume"
                )

                # Publication identity is preserved.
                publications_after = _agent_audio_publications(user_room)
                assert len(publications_after) == 1, publications_after
                assert publications_after[0].sid == sid_before, (
                    "publication SID changed on resume — engine should have preserved it"
                )

                # Audio continues to flow uninterrupted.
                await mon.assert_audio_continuous(min_rms=_AUDIO_RMS_THRESHOLD, duration=1.5)

    async def test_full_reconnect_republishes_once_and_audio_recovers(self):
        """Full reconnect must fire `reconnected` exactly once, end with
        exactly one audio publication (the SDK's auto-republish — the
        agents framework must not produce a duplicate), and audio must
        recover."""
        room_name = _e2e_room_name("full")

        async with (
            _connect_e2e_room(_USER_IDENTITY, room_name) as user_room,
            _connect_e2e_room(_AGENT_IDENTITY, room_name, agent=True) as agent_room,
            _agent_publishing_tone(agent_room) as (_output, _tone),
        ):
            await asyncio.wait_for(
                wait_for_participant(user_room, identity=_AGENT_IDENTITY),
                timeout=10.0,
            )
            track = await _await_subscribed_audio(user_room)

            async with AudioEnergyMonitor.watch(track) as mon:
                await mon.wait_for_audio(min_rms=_AUDIO_RMS_THRESHOLD, timeout=5.0)

                publications_before = _agent_audio_publications(user_room)
                assert len(publications_before) == 1

                # Bug regression check: should fire exactly once.
                reconnect_count = 0

                def _count(*_args) -> None:
                    nonlocal reconnect_count
                    reconnect_count += 1

                agent_room.on("reconnected", _count)

                await simulate_full_reconnect(agent_room)
                await wait_for_event(agent_room, "reconnected", timeout=20.0)

                # After the SDK's auto-republish the user observes
                # unpublish -> publish on the agent. Resubscribe.
                new_track = await _await_subscribed_audio(user_room)

                # Bug regression: must be exactly ONE audio publication.
                # Pre-fix, the agents framework's `_on_reconnected` raced
                # to publish a second track on top of the SDK's
                # auto-republished one.
                await asyncio.sleep(0.5)  # let any stray duplicate settle
                publications_after = _agent_audio_publications(user_room)
                assert len(publications_after) == 1, (
                    f"expected exactly 1 audio publication after full reconnect, "
                    f"saw {len(publications_after)}: {[p.sid for p in publications_after]}"
                )
                assert reconnect_count == 1, (
                    f"reconnected fired {reconnect_count} times; expected exactly 1"
                )

                async with AudioEnergyMonitor.watch(new_track) as new_mon:
                    await new_mon.wait_for_audio(min_rms=_AUDIO_RMS_THRESHOLD, timeout=10.0)
                    await new_mon.assert_audio_continuous(
                        min_rms=_AUDIO_RMS_THRESHOLD, duration=1.5
                    )
