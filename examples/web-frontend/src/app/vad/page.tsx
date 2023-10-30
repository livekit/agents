"use client";

import { Room } from "@/components/Room";
import { Backend } from "@/lib/backend";
import {
  ParticipantLoop,
  ParticipantName,
  useConnectionState,
  useRemoteParticipants,
  useRoomInfo,
  useTrackToggle,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import { useCallback, useMemo } from "react";
import { v4 } from "uuid";

export default function VADPage() {
  const room = `vad-${v4()}`;
  return (
    <Room identity="caller" room={room}>
      <VAD />
    </Room>
  );
}

function VAD() {
  const state = useConnectionState();
  const room = useRoomInfo();
  const remoteParticpants = useRemoteParticipants();
  const { toggle, enabled, pending } = useTrackToggle({
    source: Track.Source.Microphone,
  });

  const addAgent = useCallback(async () => {
    await Backend.addAgent({ type: "vad", room: room.name });
  }, [room.name]);

  const micText = useMemo(() => {
    if (pending) {
      return "...";
    }

    return enabled ? "Disable Mic" : "Enable Mic";
  }, [enabled, pending]);

  if (state !== "connected") {
    return <div>Connecting...</div>;
  }

  return (
    <div>
      <div className="flex space-x-2">
        <button
          onClick={async () => {
            await addAgent();
          }}
        >
          Add Agent
        </button>
        <button
          onClick={async () => {
            await toggle();
          }}
        >
          {micText}
        </button>
      </div>
      <div className="flex flex-col">
        Remote Participants:
        <ParticipantLoop participants={remoteParticpants}>
          <ParticipantName />
        </ParticipantLoop>
      </div>
    </div>
  );
}
