import { useCallback, useEffect, useRef, useState } from "react";
import { ConnectionDetails } from "./api/connection_details";
import {
  LiveKitRoom,
  ParticipantLoop,
  RoomAudioRenderer,
  TrackToggle,
  VideoConference,
  useLocalParticipant,
  useMediaTrack,
  useParticipantContext,
  useParticipantInfo,
  useRemoteParticipants,
} from "@livekit/components-react";
import "@livekit/components-styles";
import axios from "axios";
import { Track } from "livekit-client";

export default function Page() {
  return (
    <div className="flex flex-col">
      <div>TODO Example Links</div>
    </div>
  );
}
