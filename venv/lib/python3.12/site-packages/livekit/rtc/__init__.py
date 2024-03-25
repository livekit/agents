# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LiveKit RTC SDK
"""

from ._proto.room_pb2 import (
    ConnectionQuality,
    ConnectionState,
    DataPacketKind,
    TrackPublishOptions,
    IceTransportType,
    ContinualGatheringPolicy,
    IceServer,
)
from ._proto.e2ee_pb2 import EncryptionType, EncryptionState
from ._proto.track_pb2 import StreamState, TrackKind, TrackSource
from ._proto.video_frame_pb2 import VideoBufferType, VideoRotation
from ._proto import stats_pb2 as stats
from .audio_frame import AudioFrame
from .audio_source import AudioSource
from .audio_stream import AudioStream, AudioFrameEvent
from .participant import LocalParticipant, Participant, RemoteParticipant
from .room import ConnectError, Room, RoomOptions, RtcConfiguration, DataPacket
from .track import (
    LocalAudioTrack,
    LocalVideoTrack,
    RemoteAudioTrack,
    RemoteVideoTrack,
    Track,
    LocalTrack,
    RemoteTrack,
    AudioTrack,
    VideoTrack,
)
from .e2ee import (
    E2EEManager,
    E2EEOptions,
    KeyProviderOptions,
    KeyProvider,
    FrameCryptor,
)
from .track_publication import (
    LocalTrackPublication,
    RemoteTrackPublication,
    TrackPublication,
)
from .video_frame import (
    VideoFrame,
)
from .video_source import VideoSource
from .video_stream import VideoStream, VideoFrameEvent
from .chat import ChatManager, ChatMessage

from .version import __version__

__all__ = [
    "ConnectionQuality",
    "ConnectionState",
    "DataPacketKind",
    "TrackPublishOptions",
    "IceTransportType",
    "ContinualGatheringPolicy",
    "IceServer",
    "EncryptionType",
    "EncryptionState",
    "StreamState",
    "TrackKind",
    "TrackSource",
    "VideoBufferType",
    "VideoRotation",
    "stats",
    "AudioFrame",
    "AudioSource",
    "AudioStream",
    "AudioFrameEvent",
    "LocalParticipant",
    "Participant",
    "RemoteParticipant",
    "ConnectError",
    "Room",
    "RoomOptions",
    "RtcConfiguration",
    "DataPacket",
    "LocalAudioTrack",
    "LocalVideoTrack",
    "RemoteAudioTrack",
    "RemoteVideoTrack",
    "Track",
    "LocalTrack",
    "RemoteTrack",
    "AudioTrack",
    "VideoTrack",
    "E2EEManager",
    "E2EEOptions",
    "KeyProviderOptions",
    "KeyProvider",
    "FrameCryptor",
    "LocalTrackPublication",
    "RemoteTrackPublication",
    "TrackPublication",
    "VideoFrame",
    "VideoSource",
    "VideoStream",
    "VideoFrameEvent",
    "ChatManager",
    "ChatMessage",
    "__version__",
]
