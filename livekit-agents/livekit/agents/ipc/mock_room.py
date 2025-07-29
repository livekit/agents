import functools
from typing import Any
from unittest.mock import AsyncMock, create_autospec

from livekit import rtc


@functools.cache
def create_mock_room() -> Any:
    MockRoom = create_autospec(rtc.Room, instance=True)
    MockRoom.local_participant = create_autospec(rtc.LocalParticipant, instance=True)
    MockRoom._info = create_autospec(rtc.room.proto_room.RoomInfo, instance=True)  # type: ignore
    MockRoom.isconnected.return_value = True
    MockRoom.name = "fake_room"
    MockRoom.metadata = ""
    MockRoom.num_participants = 2
    MockRoom.num_publishers = 2
    MockRoom.connection_state = rtc.ConnectionState.CONN_CONNECTED
    MockRoom.departure_timeout = 0
    MockRoom.empty_timeout = 0

    MockRoom.sid = AsyncMock(return_value="RM_fake_sid")

    mock_remote_participant = create_autospec(rtc.RemoteParticipant, instance=True)
    mock_remote_participant.identity = "fake_human"
    mock_remote_participant.sid = "PA_fake_human"
    mock_remote_participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    MockRoom.remote_participants = {mock_remote_participant.sid: mock_remote_participant}
    return MockRoom


if __name__ == "__main__":
    mock_room = create_mock_room()

    async def test() -> None:
        print("sid", await mock_room.sid())

    import asyncio

    asyncio.run(test())

    print("local_participant", mock_room.local_participant)
    print("isconnected", mock_room.isconnected())
    print("remote_participants", mock_room.remote_participants)
