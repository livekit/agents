"use client";

import { createContext, useContext, useState, ReactNode } from "react";

interface AgentSessionContextType {
  agentState: "idle" | "speaking" | "listening";
  setAgentState: (state: "idle" | "speaking" | "listening") => void;
  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;
  roomName: string;
  identity: string;
}

const AgentSessionContext = createContext<AgentSessionContextType | undefined>(
  undefined
);

export function AgentSessionProvider({ children }: { children: ReactNode }) {
  const [agentState, setAgentState] = useState<"idle" | "speaking" | "listening">(
    "idle"
  );
  const [isConnected, setIsConnected] = useState(false);

  // In a real app, these would be dynamic based on routing/params
  const roomName = "realtime";
  const identity = `user-${Date.now()}`;

  return (
    <AgentSessionContext.Provider
      value={{
        agentState,
        setAgentState,
        isConnected,
        setIsConnected,
        roomName,
        identity,
      }}
    >
      {children}
    </AgentSessionContext.Provider>
  );
}

export function useAgentSession() {
  const context = useContext(AgentSessionContext);
  if (!context) {
    throw new Error(
      "useAgentSession must be used within AgentSessionProvider"
    );
  }
  return context;
}
