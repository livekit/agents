'use client';

import { ParticipantKind, Track } from 'livekit-client';
import {
  VideoTrack,
  useParticipantTracks,
  useRemoteParticipants,
  useSessionContext,
  useVoiceAssistant,
} from '@livekit/components-react';
import { AgentControlBar } from '@/components/agents-ui/agent-control-bar';
import { cn } from '@/lib/shadcn/utils';

interface AvatarPanelProps {
  className?: string;
}

export function AvatarPanel({ className }: AvatarPanelProps) {
  const { agent } = useVoiceAssistant();

  const remoteParticipants = useRemoteParticipants();
  const worker = remoteParticipants.find(
    (p) =>
      p.kind === ParticipantKind.AGENT && p.attributes['lk.publish_on_behalf'] === agent?.identity
  );

  const agentTracks = useParticipantTracks(
    [Track.Source.Camera, Track.Source.ScreenShare],
    agent?.identity
  );
  const workerTracks = useParticipantTracks(
    [Track.Source.Camera, Track.Source.ScreenShare],
    worker?.identity
  );

  const trackRef =
    workerTracks.find((t) => t.source === Track.Source.Camera) ??
    workerTracks.find((t) => t.source === Track.Source.ScreenShare) ??
    agentTracks.find((t) => t.source === Track.Source.Camera) ??
    agentTracks.find((t) => t.source === Track.Source.ScreenShare);

  if (process.env.NODE_ENV === 'development' && (agent || worker)) {
    console.log(
      '[AvatarPanel] agent:',
      agent?.identity,
      'worker:',
      worker?.identity,
      'agentTracks:',
      agentTracks.length,
      'workerTracks:',
      workerTracks.length,
      'trackRef:',
      trackRef?.source
    );
  }

  const { isConnected } = useSessionContext();
  const videoWidth = trackRef?.publication.dimensions?.width ?? 0;
  const videoHeight = trackRef?.publication.dimensions?.height ?? 0;

  return (
    <div className={cn('flex flex-col items-center justify-start gap-5 p-6', className)}>
      <div className="relative flex aspect-square w-full max-w-xl min-w-0 items-center justify-center overflow-hidden rounded-lg bg-black">
        {trackRef ? (
          <VideoTrack
            width={videoWidth}
            height={videoHeight}
            trackRef={trackRef}
            className="size-full object-cover"
          />
        ) : null}
      </div>

      <AgentControlBar
        variant="livekit"
        isConnected={isConnected}
        controls={{
          microphone: true,
          leave: true,
          camera: false,
          screenShare: false,
          chat: false,
        }}
        className="w-full max-w-xl shrink-0"
      />
    </div>
  );
}
