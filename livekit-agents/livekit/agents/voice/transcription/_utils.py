from __future__ import annotations

from livekit import rtc

from ...utils import shortuuid


def find_micro_track_id(room: rtc.Room, identity: str, audio_sources: list[rtc.TrackSource.ValueType] | None = None) -> str:
    p: rtc.RemoteParticipant | rtc.LocalParticipant | None = room.remote_participants.get(identity)
    if identity == room.local_participant.identity:
        p = room.local_participant

    if p is None:
        raise ValueError(f"participant {identity} not found")

    # Default audio sources if not provided
    if audio_sources is None:
        audio_sources = [rtc.TrackSource.SOURCE_MICROPHONE, rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO]

    # find first audio track
    track_id = None
    for track in p.track_publications.values():
        if track.source in audio_sources:
            track_id = track.sid
            break

    if track_id is None:
        raise ValueError(f"participant {identity} does not have an audio track")

    return track_id


def segment_uuid() -> str:
    return shortuuid("SG_")


def speech_uuid() -> str:
    return shortuuid("SP_")
