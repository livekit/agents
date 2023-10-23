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

import logging
from dataclasses import dataclass
from typing import Callable, Awaitable
import livekit.rtc as rtc

Event_Fn = Callable[..., Awaitable[None]]

class Agent:
    """This Agent class is a convenience class on top of a LocalParticipant and Room that handles boilerplate
       code for subscribing to tracks and handling participants joining and leaving the room. You can use this
       class without compromise because the LocalParticipant and Room are exposed as public attributes: self.participant
       and self.room.

       See /examples for usage examples.
    """

    @dataclass
    class OnVideoTrackEvent:
        track: rtc.VideoTrack
        participant: rtc.RemoteParticipant

    @dataclass
    class OnAudioTrackEvent:
        track: rtc.AudioTrack
        participant: rtc.RemoteParticipant

    @dataclass
    class OnTrackAvailableEvent:
        publication: rtc.TrackPublication
        participant: rtc.RemoteParticipant

    def __init__(self, *_, participant: rtc.LocalParticipant, room: rtc.Room):
        self.participant = participant
        self.room = room
        self.room.on("participant_connected", self._on_participant_connected_or_disconnected)
        self.room.on("participant_disconnected", self._on_participant_connected_or_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("track_published", self._on_track_published)
        self.room.on("data_received", self._on_data_received)
        self.participants = self.room.participants
        self.should_process_cb = None

        self._handle_existing_tracks()
        self.stream_tasks = set()

        # handlers
        self._on_audio_track_handler: Callable[[Agent.OnAudioTrackEvent], None] = \
            lambda: logging.warning("a handler was not provided for 'on_audio_track'")
        self._on_video_track_handler: Callable[[Agent.OnVideoTrackEvent], None] = \
            lambda: logging.warning("a handler was not provided for 'on_video_track'")
        self._on_track_available_handler: Callable[[Agent.OnTrackAvailableEvent], bool] = \
            lambda: logging.warning("a handler was not provided for 'on_track_availale'")

    async def cleanup(self):
        await self.room.disconnect()

    def _handle_existing_tracks(self):
        for participantKey in self.participants:
            for publicationKey in self.participants[participantKey].tracks:
                publication = self.participants[participantKey].tracks[publicationKey]
                self._on_track_published(publication, self.participants[participantKey])

    def on_video_track(self, handler: Callable[[OnVideoTrackEvent], None]):
        self._on_video_track_handler = handler

    def on_audio_track(self, handler: Callable[[OnAudioTrackEvent], None]):
        self._on_audio_track_handler = handler

    def on_track_available(self, handler: Callable[[OnTrackAvailableEvent], bool]):
        self._on_track_available_handler = handler

    def _on_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.Participant):
        # Don't do anything for our own tracks
        if participant.sid == self.participant.sid:
            return

        if self._on_track_available_handler(publication, participant):
            publication.set_subscribed(True)

    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind == 1:
            self._on_audio_track_handler(Agent.OnAudioTrackEvent(track, participant))
        elif publication.kind == 2:
            self._on_video_track_handler(Agent.OnVideoTrackEvent(track, participant))