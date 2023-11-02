"use client";

import { Room } from "@/components/Room";
import { Backend } from "@/lib/backend";
import toast, { Toaster } from "react-hot-toast";
import {
  ParticipantAudioTile,
  ParticipantLoop,
  ParticipantName,
  ParticipantTile,
  RoomAudioRenderer,
  TrackLoop,
  useConnectionState,
  useDataChannel,
  useLiveKitRoom,
  useParticipantContext,
  useParticipantInfo,
  useRemoteParticipant,
  useRemoteParticipants,
  useRoomContext,
  useRoomInfo,
  useStartAudio,
  useTrackToggle,
  useTracks,
} from "@livekit/components-react";
import { AudioTrack, RemoteParticipant, Track } from "livekit-client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { v4 } from "uuid";
import { AudioTrackRenderer } from "@/components/AudioTrackRenderer";

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
  const { message } = useDataChannel();
  const [agentAdding, setAgentAdding] = useState(false);
  const room = useRoomInfo();
  const remoteParticpants = useRemoteParticipants();
  const { toggle, enabled, pending } = useTrackToggle({
    source: Track.Source.Microphone,
  });

  useEffect(() => {
    if (!message) {
      return;
    }
    const strMessage = new TextDecoder().decode(message.payload);
    toast.success(strMessage);
  }, [message]);

  const addAgent = useCallback(async () => {
    if (agentAdding) {
      return;
    }
    setAgentAdding(true);
    try {
      await Backend.addAgent({ type: "vad", room: room.name });
    } finally {
      setAgentAdding(false);
    }
  }, [agentAdding, room.name]);

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
    <div className="flex flex-col w-full items-center">
      <Toaster />
      <div className="flex space-x-2">
        <button
          className="p-2 rounded-sm border hover:bg-gray-800"
          onClick={async () => {
            await addAgent();
          }}
        >
          {agentAdding ? "..." : "Add Agent"}
        </button>
        <button
          className="p-2 rounded-sm border hover:bg-gray-800"
          onClick={async () => {
            await toggle();
          }}
        >
          {micText}
        </button>
      </div>
      <RoomAudioRenderer />
      <div className="flex flex-col p-2">
        Remote Participants:
        <ParticipantLoop participants={remoteParticpants}>
          <ParticipantName />
        </ParticipantLoop>
      </div>
    </div>
  );
}
