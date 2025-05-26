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
            room.off("participant_connected", _on_participant_connected)
            if not fut.done():
                fut.set_result(p)

    room.on("participant_connected", _on_participant_connected)

    for p in room.remote_participants.values():
        _on_participant_connected(p)
        if fut.done():
            break

    return await fut
