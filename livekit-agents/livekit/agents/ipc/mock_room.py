from unittest.mock import create_autospec

from livekit import rtc

MockRoom = create_autospec(rtc.Room, instance=True)
MockRoom.local_participant = create_autospec(rtc.LocalParticipant, instance=True)
MockRoom._info = create_autospec(rtc.room.proto_room.RoomInfo, instance=True)  # type: ignore
MockRoom.isconnected.return_value = True
MockRoom.name = "fake_room"

mock_remote_participant = create_autospec(rtc.RemoteParticipant, instance=True)
mock_remote_participant.identity = "fake_human"
mock_remote_participant.sid = "PA_fake_human"
mock_remote_participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
MockRoom.remote_participants = {mock_remote_participant.sid: mock_remote_participant}

if __name__ == "__main__":
    mock_room = MockRoom

    print("local_participant", mock_room.local_participant)
    print("isconnected", mock_room.isconnected())
    print("remote_participants", mock_room.remote_participants)
