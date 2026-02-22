"use client";

import { useEffect, useRef, useState } from "react";
import { Participant, Room, RoomEvent } from "livekit-client";
import { useAgentSession } from "./agent-session-provider";
import { connectToRoom } from "@/lib/utils";
import { Loader2 } from "lucide-react";

export function AppView() {
  const { roomName, identity, setIsConnected } = useAgentSession();
  const roomRef = useRef<Room | null>(null);
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initRoom = async () => {
      try {
        setIsLoading(true);
        const room = await connectToRoom(roomName, identity);
        roomRef.current = room;
        setIsConnected(true);

        // Update participants when they change
        const updateParticipants = () => {
          setParticipants(Array.from(room.remoteParticipants.values()));
        };

        room.on(RoomEvent.ParticipantConnected, updateParticipants);
        room.on(RoomEvent.ParticipantDisconnected, updateParticipants);
        updateParticipants();

        setError(null);
      } catch (err) {
        console.error("Failed to connect to room:", err);
        setError(
          err instanceof Error ? err.message : "Failed to connect to room"
        );
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    };

    initRoom();

    return () => {
      if (roomRef.current) {
        roomRef.current.disconnect();
      }
    };
  }, [roomName, identity, setIsConnected]);

  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="w-8 h-8 animate-spin text-lk-accent" />
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Connecting to agent...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="flex flex-col items-center gap-4 p-4 text-center">
          <p className="text-red-600 dark:text-red-400">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-lk-accent text-white rounded-lg hover:opacity-90"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <SessionView room={roomRef.current} participants={participants} />
  );
}

function SessionView({
  room,
  participants,
}: {
  room: Room | null;
  participants: Participant[];
}) {
  if (!room) return null;

  // Mobile-first responsive layout
  return (
    <div className="w-full h-full flex flex-col md:flex-row gap-0 md:gap-4 p-0 md:p-4 bg-slate-50 dark:bg-slate-950">
      {/* Avatar Panel - Full width on mobile, half on desktop */}
      <div className="w-full md:w-1/2 h-1/2 md:h-full flex-shrink-0 bg-black rounded-none md:rounded-lg overflow-hidden">
        <AvatarPanel room={room} participants={participants} />
      </div>

      {/* Form and Chat Panel - Full width on mobile, half on desktop */}
      <div className="w-full md:w-1/2 h-1/2 md:h-full flex flex-col gap-2 md:gap-4 overflow-hidden">
        <ChatPanel room={room} participants={participants} />
        <FormPanel room={room} />
      </div>
    </div>
  );
}

function AvatarPanel({
  room,
  participants,
}: {
  room: Room;
  participants: Participant[];
}) {
  return (
    <div className="w-full h-full bg-black flex items-center justify-center">
      <p className="text-white text-sm">Avatar rendering here</p>
      <p className="text-slate-400 text-xs">
        Participants: {participants.length}
      </p>
    </div>
  );
}

function ChatPanel({
  room,
  participants,
}: {
  room: Room;
  participants: Participant[];
}) {
  return (
    <div className="flex-1 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        <div className="text-xs text-slate-500 dark:text-slate-400 text-center">
          Chat transcript
        </div>
      </div>
    </div>
  );
}

function FormPanel({ room }: { room: Room }) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-4">
      <div className="text-xs text-slate-500 dark:text-slate-400">
        Form area
      </div>
    </div>
  );
}
