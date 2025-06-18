from __future__ import annotations

import asyncio

from livekit import rtc


async def wait_for_participant(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType | None = None,
) -> rtc.RemoteParticipant:
    """
    Returns a participant that matches the given identity. If identity is None, the first
    participant that joins the room will be returned.
    If the participant has already joined, the function will return immediately.
    """
    if not room.isconnected():
        raise RuntimeError("room is not connected")

    fut = asyncio.Future[rtc.RemoteParticipant]()

    def kind_match(p: rtc.RemoteParticipant) -> bool:
        if kind is None:
            return True

        if isinstance(kind, list):
            return p.kind in kind

        return p.kind == kind

    def _on_participant_connected(p: rtc.RemoteParticipant) -> None:
        if (identity is None or p.identity == identity) and kind_match(p):
            if not fut.done():
                fut.set_result(p)

    room.on("participant_connected", _on_participant_connected)

    try:
        for p in room.remote_participants.values():
            _on_participant_connected(p)
            if fut.done():
                break

        return await fut
    finally:
        room.off("participant_connected", _on_participant_connected)


async def wait_for_track_subscribed(
    room: rtc.Room,
    *,
    identity: str | None = None,
    kind: list[rtc.TrackKind.ValueType] | rtc.TrackKind.ValueType | None = None,
) -> rtc.Track:
    """Returns a remote track matching the given identity and kind.
    If identity is None, the first track matching the kind will be returned.
    If the track has already been published, the function will return immediately.
    """
    if not room.isconnected():
        raise RuntimeError("room is not connected")

    fut = asyncio.Future[rtc.Track]()

    def kind_match(t: rtc.Track) -> bool:
        if kind is None:
            return True

        if isinstance(kind, list):
            return t.kind in kind

        return t.kind == kind

    def _on_track_subscribed(
        track: rtc.Track | None,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if fut.done():
            return

        if (identity is None or participant.identity == identity) and track and kind_match(track):
            fut.set_result(track)

    room.on("track_subscribed", _on_track_subscribed)

    try:
        for p in room.remote_participants.values():
            for publication in p.track_publications.values():
                if publication.subscribed:
                    _on_track_subscribed(publication.track, publication, p)
                if fut.done():
                    break

        return await fut
    finally:
        room.off("track_subscribed", _on_track_subscribed)
