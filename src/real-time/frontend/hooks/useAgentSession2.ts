"use client";

import { useEffect, useState } from "react";
import { Room, RoomEvent } from "livekit-client";
import { useAgentSession } from "@/components/app/agent-session-provider";

export function useAgentSession2(room: Room | null) {
  const { agentState, setAgentState } = useAgentSession();
  const [isAgent, setIsAgent] = useState(false);

  useEffect(() => {
    if (!room) return;

    const updateAgentState = () => {
      const agentParticipant = Array.from(room.remoteParticipants.values()).find(
        (p) => p.identity.startsWith("agent-") || p.identity === "agent"
      );

      if (agentParticipant) {
        setIsAgent(true);
        // Could implement more sophisticated state tracking here
      }
    };

    room.on(RoomEvent.ParticipantConnected, updateAgentState);
    room.on(RoomEvent.ParticipantDisconnected, updateAgentState);
    updateAgentState();

    return () => {
      room.off(RoomEvent.ParticipantConnected, updateAgentState);
      room.off(RoomEvent.ParticipantDisconnected, updateAgentState);
    };
  }, [room, setAgentState]);

  return {
    agentState,
    setAgentState,
    isAgent,
  };
}
