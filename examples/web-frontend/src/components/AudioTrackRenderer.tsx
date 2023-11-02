import { AudioTrack } from "livekit-client";
import { useEffect, useRef } from "react";

export function AudioTrackRenderer({ track }: { track: AudioTrack }) {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audioEl = audioRef.current!;
    track.attach(audioEl);
    return () => {
      track.detach(audioEl);
    };
  }, [track, track.mediaStreamTrack]);

  return <audio ref={audioRef} autoPlay muted={false} />;
}
