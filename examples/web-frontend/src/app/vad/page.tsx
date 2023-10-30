"use client";

import { Room } from "@/components/Room";
import { Backend } from "@/lib/backend";
import { useConnectionState, useRoomInfo } from "@livekit/components-react";
import { useCallback } from "react";
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

  const addAgent = useCallback(async () => {
    await Backend.addAgent({ type: "vad", room: room.name });
  }, [room.name]);

  if (state !== "connected") {
    return <div>Connecting...</div>;
  }

  return (
    <div>
      <button
        onClick={async () => {
          await addAgent();
        }}
      >
        Add Agent
      </button>
    </div>
  );
}
