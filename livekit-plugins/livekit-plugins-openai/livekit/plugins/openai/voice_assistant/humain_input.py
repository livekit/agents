import asyncio

from livekit import rtc


class HumanInput:
    def __init__(self, room: rtc.Room, participant: rtc.RemoteParticipant):
        room.on("track_published", self._subscribe_to_microphone)
        room.on("track_subscribed", self._subscribe_to_microphone)

        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        pass

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """Subscribe to the participant microphone if found"""

        if self._linked_participant is None:
            return

        @utils.log_exceptions(logger=logger)
        async def _read_audio_stream_task(audio_stream: rtc.AudioStream):
            bstream = utils.audio.AudioByteStream(
                proto.SAMPLE_RATE,
                proto.NUM_CHANNELS,
                samples_per_channel=proto.IN_FRAME_SIZE,
            )

            async for ev in audio_stream:
                for frame in bstream.write(ev.frame.data.tobytes()):
                    self._input_audio_ch.send_nowait(frame)

        for publication in self._linked_participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            if (
                publication.track is not None
                and publication.track != self._subscribed_track
            ):
                self._subscribed_track = publication.track  # type: ignore
                if self._read_micro_atask is not None:
                    self._read_micro_atask.cancel()

                self._read_micro_atask = asyncio.create_task(
                    _read_audio_stream_task(
                        rtc.AudioStream(
                            self._subscribed_track,  # type: ignore
                            sample_rate=proto.SAMPLE_RATE,
                            num_channels=proto.NUM_CHANNELS,
                        )
                    )
                )
                break
