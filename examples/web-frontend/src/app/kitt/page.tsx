"use client";

import { Room } from "@/components/Room";
import { Backend } from "@/lib/backend";
import toast, { Toaster } from "react-hot-toast";
import {
  ParticipantLoop,
  ParticipantName,
  useConnectionState,
  useDataChannel,
  useRemoteParticipants,
  useRoomInfo,
  useTrackToggle,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { v4 } from "uuid";

export default function STTPage() {
  const room = `kitt-${v4()}`;
  return (
    <Room identity="caller" room={room}>
      <KITT />
    </Room>
  );
}

function KITT() {
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
      await Backend.addAgent({ type: "kitt", room: room.name });
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
      <div className="flex flex-col p-2">
        Remote Participants:
        <ParticipantLoop participants={remoteParticpants}>
          <ParticipantName className="py-1" />
        </ParticipantLoop>
      </div>
    </div>
  );
}
