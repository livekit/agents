"use client";

import { useCallback, useState } from "react";
import { Room, RemoteTrackPublication, Track, RoomEvent } from "livekit-client";

interface VideoTrack {
  participant: string;
  publication: RemoteTrackPublication;
  track: Track;
}

export function useVideoTrack(room: Room | null, participantIdentity?: string) {
  const [videoTracks, setVideoTracks] = useState<VideoTrack[]>([]);

  const subscribeToVideoTracks = useCallback(() => {
    if (!room) return;

    const updateTracks = () => {
      const tracks: VideoTrack[] = [];

      // Get video tracks from all participants or specific participant
      room.remoteParticipants.forEach((participant) => {
        if (participantIdentity && participant.identity !== participantIdentity) {
          return;
        }

        participant.trackPublications.forEach((publication) => {
          if (
            publication.track &&
            publication.track.kind === Track.Kind.Video
          ) {
            tracks.push({
              participant: participant.identity,
              publication: publication as RemoteTrackPublication,
              track: publication.track,
            });
          }
        });
      });

      setVideoTracks(tracks);
    };

    // Subscribe to track events
    room.on(RoomEvent.TrackSubscribed, updateTracks);
    room.on(RoomEvent.TrackUnsubscribed, updateTracks);
    room.on(RoomEvent.ParticipantConnected, updateTracks);
    room.on(RoomEvent.ParticipantDisconnected, updateTracks);

    // Initial update
    updateTracks();

    return () => {
      room.off(RoomEvent.TrackSubscribed, updateTracks);
      room.off(RoomEvent.TrackUnsubscribed, updateTracks);
      room.off(RoomEvent.ParticipantConnected, updateTracks);
      room.off(RoomEvent.ParticipantDisconnected, updateTracks);
    };
  }, [room, participantIdentity]);

  return {
    videoTracks,
    subscribeToVideoTracks,
  };
}

export function useAudioTrack(room: Room | null, participantIdentity?: string) {
  const [audioTracks, setAudioTracks] = useState<VideoTrack[]>([]);

  const subscribeToAudioTracks = useCallback(() => {
    if (!room) return;

    const updateTracks = () => {
      const tracks: VideoTrack[] = [];

      room.remoteParticipants.forEach((participant) => {
        if (participantIdentity && participant.identity !== participantIdentity) {
          return;
        }

        participant.trackPublications.forEach((publication) => {
          if (
            publication.track &&
            publication.track.kind === Track.Kind.Audio
          ) {
            tracks.push({
              participant: participant.identity,
              publication: publication as RemoteTrackPublication,
              track: publication.track,
            });
          }
        });
      });

      setAudioTracks(tracks);
    };

    room.on(RoomEvent.TrackSubscribed, updateTracks);
    room.on(RoomEvent.TrackUnsubscribed, updateTracks);
    updateTracks();

    return () => {
      room.off(RoomEvent.TrackSubscribed, updateTracks);
      room.off(RoomEvent.TrackUnsubscribed, updateTracks);
    };
  }, [room, participantIdentity]);

  return {
    audioTracks,
    subscribeToAudioTracks,
  };
}
