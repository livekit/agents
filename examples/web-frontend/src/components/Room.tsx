"use client";
import { Backend } from "@/lib/backend";
import { LiveKitRoom } from "@livekit/components-react";
import { useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";

type Props = {
  children: React.ReactNode;
  room: string;
  identity: string;
};

export function Room({ children, identity, room }: Props) {
  const [token, setToken] = useState<string>("");
  const [wsUrl, setWsUrl] = useState<string>("");

  useEffect(() => {
    Backend.generateConnectionDetails({ room, identity }).then(
      ({ token, ws_url }) => {
        setToken(token);
        setWsUrl(ws_url);
        console.log(`ws_url: ${ws_url}`);
      }
    );
  }, [identity, room]);

  return (
    <LiveKitRoom
      serverUrl={wsUrl}
      connect={wsUrl !== "" && token !== ""}
      token={token}
      connectOptions={{ autoSubscribe: true }}
    >
      {children}
    </LiveKitRoom>
  );
}
