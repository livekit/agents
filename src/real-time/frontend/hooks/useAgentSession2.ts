"use client";

import { useEffect, useState } from "react";
import { Room, ParticipantEvent } from "livekit-client";
import { useAgentSession } from "@/components/app/agent-session-provider";

export function useAgentSession2(room: Room | null) {
  const { agentState, setAgentState } = useAgentSession();
  const [isAgent, setIsAgent] = useState(false);

  useEffect(() => {
    if (!room) return;

    const updateAgentState = () => {
      const agentParticipant = Array.from(room.participants.values()).find(
        (p) => p.identity.startsWith("agent-") || p.identity === "agent"
      );

      if (agentParticipant) {
        setIsAgent(true);
        // Could implement more sophisticated state tracking here
      }
    };

    room.on("participantJoined" as any, updateAgentState);
    room.on("participantLeft" as any, updateAgentState);
    updateAgentState();

    return () => {
      room.off("participantJoined" as any, updateAgentState);
      room.off("participantLeft" as any, updateAgentState);
    };
  }, [room, setAgentState]);

  return {
    agentState,
    setAgentState,
    isAgent,
  };
}
