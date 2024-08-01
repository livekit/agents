from __future__ import annotations

import uuid

from livekit import rtc


def find_micro_track_id(room: rtc.Room, identity: str) -> str:
    p: rtc.RemoteParticipant | rtc.LocalParticipant | None = (
        room.remote_participants.get(identity)
    )
    if identity == room.local_participant.identity:
        p = room.local_participant

    if p is None:
        raise ValueError(f"participant {identity} not found")

    # find first micro track
    track_id = None
    for track in p.track_publications.values():
        if track.source == rtc.TrackSource.SOURCE_MICROPHONE:
            track_id = track.sid
            break

    if track_id is None:
        raise ValueError(f"participant {identity} does not have a microphone track")

    return track_id


def segment_uuid() -> str:
    return "SG_" + str(uuid.uuid4().hex)[:12]
