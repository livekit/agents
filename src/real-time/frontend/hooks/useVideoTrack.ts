"use client";

import { useCallback, useState } from "react";
import { Room, RemoteTrackPublication, Track } from "livekit-client";

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
      room.participants.forEach((participant) => {
        if (participantIdentity && participant.identity !== participantIdentity) {
          return;
        }

        participant.publications.forEach((publication) => {
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
    room.on("trackSubscribed", updateTracks);
    room.on("trackUnsubscribed", updateTracks);
    room.on("participantJoined", updateTracks);
    room.on("participantLeft", updateTracks);

    // Initial update
    updateTracks();

    return () => {
      room.off("trackSubscribed", updateTracks);
      room.off("trackUnsubscribed", updateTracks);
      room.off("participantJoined", updateTracks);
      room.off("participantLeft", updateTracks);
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

      room.participants.forEach((participant) => {
        if (participantIdentity && participant.identity !== participantIdentity) {
          return;
        }

        participant.publications.forEach((publication) => {
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

    room.on("trackSubscribed", updateTracks);
    room.on("trackUnsubscribed", updateTracks);
    updateTracks();

    return () => {
      room.off("trackSubscribed", updateTracks);
      room.off("trackUnsubscribed", updateTracks);
    };
  }, [room, participantIdentity]);

  return {
    audioTracks,
    subscribeToAudioTracks,
  };
}
