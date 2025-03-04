from unittest.mock import create_autospec
from livekit import rtc

MockRoom = create_autospec(rtc.Room, instance=True)
MockRoom.local_participant = create_autospec(rtc.LocalParticipant, instance=True)
MockRoom._info = create_autospec(rtc.room.proto_room.RoomInfo, instance=True)

if __name__ == "__main__":
    mock_room = MockRoom

    print(mock_room.local_participant)
