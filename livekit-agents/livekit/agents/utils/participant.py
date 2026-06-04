from __future__ import annotations

import asyncio
from typing import Literal, overload

from livekit import rtc

from ..types import ATTRIBUTE_AGENT_NAME


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
    if not room.isconnected():
        raise RuntimeError("room is not connected")

    fut: asyncio.Future[rtc.RemoteParticipant] = asyncio.Future()

    def matches_agent(p: rtc.RemoteParticipant) -> bool:
        if p.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return False
        if agent_name is None:
            return True
        return p.attributes.get(ATTRIBUTE_AGENT_NAME) == agent_name

    def on_participant_connected(p: rtc.RemoteParticipant) -> None:
        if matches_agent(p) and not fut.done():
            fut.set_result(p)

    def on_attributes_changed(changed: list[str], p: rtc.Participant) -> None:
        if isinstance(p, rtc.RemoteParticipant) and matches_agent(p) and not fut.done():
            fut.set_result(p)

    def on_connection_state_changed(state: int) -> None:
        if state == rtc.ConnectionState.CONN_DISCONNECTED and not fut.done():
            fut.set_exception(RuntimeError("room disconnected while waiting for agent participant"))

    room.on("participant_connected", on_participant_connected)
    room.on("participant_attributes_changed", on_attributes_changed)
    room.on("connection_state_changed", on_connection_state_changed)

    try:
        # Check existing participants
        for p in room.remote_participants.values():
            if matches_agent(p):
                return p

        return await fut
    finally:
        room.off("participant_connected", on_participant_connected)
        room.off("participant_attributes_changed", on_attributes_changed)
        room.off("connection_state_changed", on_connection_state_changed)


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
    if not room.isconnected():
        raise RuntimeError("room is not connected")
    if identity not in room.remote_participants:
        raise RuntimeError(f"participant {identity!r} is not in the room")

    fut: asyncio.Future[None] = asyncio.Future()

    def _is_match(p: rtc.Participant) -> bool:
        return (
            isinstance(p, rtc.RemoteParticipant)
            and p.identity == identity
            and p.attributes.get(attribute) == value
        )

    def _on_attributes_changed(_changed: list[str], p: rtc.Participant) -> None:
        if _is_match(p) and not fut.done():
            fut.set_result(None)

    def _on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        if p.identity == identity and not fut.done():
            fut.set_exception(
                RuntimeError(f"participant {identity!r} disconnected while waiting for {attribute}")
            )

    def _on_connection_state_changed(state: int) -> None:
        if state == rtc.ConnectionState.CONN_DISCONNECTED and not fut.done():
            fut.set_exception(
                RuntimeError(f"room disconnected while waiting for {identity!r} {attribute}")
            )

    room.on("participant_attributes_changed", _on_attributes_changed)
    room.on("participant_disconnected", _on_participant_disconnected)
    room.on("connection_state_changed", _on_connection_state_changed)

    try:
        # defensive double check
        existing = room.remote_participants.get(identity)
        if existing is None:
            raise RuntimeError(f"participant {identity!r} is not in the room")
        if existing.attributes.get(attribute) == value:
            return
        await fut
    finally:
        room.off("participant_attributes_changed", _on_attributes_changed)
        room.off("participant_disconnected", _on_participant_disconnected)
        room.off("connection_state_changed", _on_connection_state_changed)


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
    if not room.isconnected():
        raise RuntimeError("room is not connected")

    fut = asyncio.Future[rtc.Participant]()

    def kind_match(p: rtc.Participant) -> bool:
        if kind is None:
            return True

        if isinstance(kind, list):
            return p.kind in kind

        return p.kind == kind

    def _on_participant_active(p: rtc.RemoteParticipant) -> None:
        if (identity is None or p.identity == identity) and kind_match(p):
            if not fut.done():
                fut.set_result(p)

    def _on_connection_state_changed(state: int) -> None:
        if state == rtc.ConnectionState.CONN_DISCONNECTED and not fut.done():
            fut.set_exception(RuntimeError("room disconnected while waiting for participant"))

    room.on("participant_active", _on_participant_active)
    room.on("connection_state_changed", _on_connection_state_changed)

    try:
        if include_local:
            local = room.local_participant
            if (identity is None or local.identity == identity) and kind_match(local):
                return local

        for p in room.remote_participants.values():
            if p.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE:
                _on_participant_active(p)
            if fut.done():
                break

        return await fut
    finally:
        room.off("participant_active", _on_participant_active)
        room.off("connection_state_changed", _on_connection_state_changed)


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
    """
    if not room.isconnected():
        raise RuntimeError("room is not connected")

    fut = asyncio.Future[rtc.TrackPublication]()

    def kind_match(k: rtc.TrackKind.ValueType) -> bool:
        if kind is None:
            return True

        if isinstance(kind, list):
            return k in kind

        return k == kind

    def _matches(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> bool:
        if not (identity is None or participant.identity == identity):
            return False
        if not kind_match(publication.kind):
            return False
        if wait_for_subscription and not (publication.subscribed and publication.track is not None):
            return False
        return True

    def _on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if not fut.done() and _matches(publication, participant):
            fut.set_result(publication)

    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if not fut.done() and _matches(publication, participant):
            fut.set_result(publication)

    def _on_local_track_published(
        publication: rtc.LocalTrackPublication, _track: rtc.Track
    ) -> None:
        if fut.done():
            return

        local = room.local_participant
        if (identity is None or local.identity == identity) and kind_match(publication.kind):
            fut.set_result(publication)

    def _on_connection_state_changed(state: int) -> None:
        if state == rtc.ConnectionState.CONN_DISCONNECTED and not fut.done():
            fut.set_exception(RuntimeError("room disconnected while waiting for track publication"))

    if wait_for_subscription:
        room.on("track_subscribed", _on_track_subscribed)
    else:
        room.on("track_published", _on_track_published)
    if include_local:
        room.on("local_track_published", _on_local_track_published)

    room.on("connection_state_changed", _on_connection_state_changed)

    try:
        if include_local:
            local = room.local_participant
            if identity is None or local.identity == identity:
                for local_publication in local.track_publications.values():
                    if kind_match(local_publication.kind):
                        return local_publication

        for p in room.remote_participants.values():
            for publication in p.track_publications.values():
                if _matches(publication, p):
                    fut.set_result(publication)
                    break
            if fut.done():
                break

        return await fut
    finally:
        if wait_for_subscription:
            room.off("track_subscribed", _on_track_subscribed)
        else:
            room.off("track_published", _on_track_published)
        if include_local:
            room.off("local_track_published", _on_local_track_published)
        room.off("connection_state_changed", _on_connection_state_changed)
