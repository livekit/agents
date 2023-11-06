"use client";

import { Room } from "@/components/Room";
import { Backend } from "@/lib/backend";
import toast, { Toaster } from "react-hot-toast";
import {
  ParticipantLoop,
  ParticipantName,
  RoomAudioRenderer,
  useConnectionState,
  useDataChannel,
  useRemoteParticipants,
  useRoomInfo,
  useTrackToggle,
} from "@livekit/components-react";
import { Track, DataPacket_Kind } from "livekit-client";
import {
  useCallback,
  useEffect,
  useState,
  ChangeEvent,
  FormEvent,
} from "react";
import { v4 } from "uuid";

export default function TTSPage() {
  const room = `vad-${v4()}`;
  return (
    <Room identity="caller" room={room}>
      <TTS />
    </Room>
  );
}

function TTS() {
  const state = useConnectionState();
  const { message, send } = useDataChannel();
  const [agentAdding, setAgentAdding] = useState(false);
  const [text, setText] = useState("");
  const room = useRoomInfo();
  const remoteParticpants = useRemoteParticipants();
  const { toggle, enabled, pending } = useTrackToggle({
    source: Track.Source.Microphone,
  });

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    setText(event.target.value);
  };

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
      await Backend.addAgent({ type: "tts", room: room.name });
    } finally {
      setAgentAdding(false);
    }
  }, [agentAdding, room.name]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // Prevent the default form submit action
    await send(
      new TextEncoder().encode(JSON.stringify({ type: "tts", text })),
      {
        kind: DataPacket_Kind.RELIABLE,
        destination: remoteParticpants.map((p) => p.sid),
      }
    );
    setText(""); // Optionally, clear the input after the operation
  };

  if (state !== "connected") {
    return <div>Connecting...</div>;
  }

  return (
    <div className="flex flex-col w-full items-center space-y-2">
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
      </div>
      <form className="flex space-x-2" onSubmit={handleSubmit}>
        <input
          className="p-1 w-[200px] text-black"
          type="text"
          value={text}
          onChange={handleInputChange}
          placeholder="Type something..."
        />
        {/* Submit button is optional since pressing Enter will submit the form */}
        <button type="submit">Submit</button>
      </form>
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
