"""Wait on a room for a participant, an attribute or a track."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

from livekit import rtc

from .. import _core


def _kinds(kind: Sequence[int] | int | None) -> list[int] | None:
    if kind is None:
        return None
    return [kind] if isinstance(kind, int) else list(kind)


def _room_session(room: rtc.Room) -> _core._core_RoomSession:
    from ..voice.room_io import Room

    return _core._core_create_room_session(Room.of(room))


async def wait_for_agent(
    room: rtc.Room,
    *,
    agent_name: str | None = None,
) -> rtc.RemoteParticipant:
    """
    Wait for an agent participant to join the room.

    Args:
        room: The room to wait for the agent in.
        agent_name: If provided, waits for an agent with matching lk.agent.name attribute.
                   If None, returns the first agent participant found.

    Returns:
        The agent participant.

    Raises:
        RuntimeError: If the room is not connected.
    """
    identity = await _core.wait_for_agent(_room_session(room), agent_name)
    return room.remote_participants[identity]


async def wait_for_participant_attribute(
    room: rtc.Room,
    *,
    identity: str,
    attribute: str,
    value: str,
) -> None:
    """Wait until a remote participant's attribute equals ``value``.

    Returns immediately if the attribute is already set. Raises
    :class:`RuntimeError` if the room is not connected, the participant is not
    present, the participant disconnects, or the room disconnects before the
    attribute is set.
    """
    await _core.wait_for_participant_attribute(_room_session(room), identity, attribute, value)


@overload
async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: Literal[False] = False,
) -> rtc.RemoteParticipant: ...


@overload
async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: Literal[True],
) -> rtc.Participant: ...


async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
    include_local: bool = False,
) -> rtc.Participant:
    """
    Returns a participant that matches the given identity. If identity is None, the first
    participant that joins the room will be returned.
    If the participant has already joined, the function will return immediately.

    When `include_local` is True, the local participant is also considered.
    """
    found = await _core.wait_for_participant(
        _room_session(room), identity, _kinds(kind), include_local
    )
    return room.local_participant if found is None else room.remote_participants[found]


@overload
async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: Literal[False] = False,
    wait_for_subscription: bool = False,
) -> rtc.RemoteTrackPublication: ...


@overload
async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: Literal[True],
    wait_for_subscription: bool = False,
) -> rtc.TrackPublication: ...


async def wait_for_track_publication(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
    include_local: bool = False,
    wait_for_subscription: bool = False,
) -> rtc.TrackPublication:
    """Returns a track publication matching the given identity and kind.
    If identity is None, the first track matching the kind will be returned.
    If a matching track is already published (and subscribed when
    ``wait_for_subscription`` is set), the function returns immediately.

    When `include_local` is True, tracks published by the local participant are also considered;
    local publications resolve on publish and ignore ``wait_for_subscription``.

    When ``identity`` is set, raises :class:`RuntimeError` if that participant
    disconnects before a matching publication is found, so callers don't wait
    on a participant that is gone.
    """
    publisher, sid = await _core.wait_for_track_publication(
        _room_session(room), identity, _kinds(kind), include_local, wait_for_subscription
    )
    participant: rtc.Participant = (
        room.local_participant if publisher is None else room.remote_participants[publisher]
    )
    return participant.track_publications[sid]
