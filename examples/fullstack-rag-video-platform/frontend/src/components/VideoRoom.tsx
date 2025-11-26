'use client';

import { useEffect, useState } from 'react';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useConnectionState,
  useRoomInfo,
  useTracks,
  VideoTrack,
  AudioTrack,
} from '@livekit/components-react';
import '@livekit/components-styles';
import { Track } from 'livekit-client';
import { Mic, MicOff, Video as VideoIcon, VideoOff, PhoneOff } from 'lucide-react';

interface VideoRoomProps {
  roomName: string;
  onDisconnect: () => void;
}

export function VideoRoom({ roomName, onDisconnect }: VideoRoomProps) {
  const [token, setToken] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    async function getToken() {
      try {
        const response = await fetch(`/api/token?roomName=${roomName}&participantName=user`);
        if (!response.ok) {
          throw new Error('Failed to get access token');
        }
        const data = await response.json();
        setToken(data.token);
        setIsLoading(false);
      } catch (err) {
        console.error('Error getting token:', err);
        setError('Failed to connect to room. Please try again.');
        setIsLoading(false);
      }
    }

    getToken();
  }, [roomName]);

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Connecting to video session...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-8 text-center">
        <div className="text-red-500 mb-4">{error}</div>
        <button
          onClick={onDisconnect}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <LiveKitRoom
        video={true}
        audio={true}
        token={token}
        serverUrl={process.env.NEXT_PUBLIC_LIVEKIT_URL || 'ws://localhost:7880'}
        data-lk-theme="default"
        className="h-[600px]"
      >
        <RoomContent onDisconnect={onDisconnect} />
        <RoomAudioRenderer />
      </LiveKitRoom>
    </div>
  );
}

function RoomContent({ onDisconnect }: { onDisconnect: () => void }) {
  const connectionState = useConnectionState();
  const roomInfo = useRoomInfo();
  const tracks = useTracks([Track.Source.Camera, Track.Source.Microphone]);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);

  return (
    <div className="relative h-full bg-gray-900">
      {/* Video Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 h-[calc(100%-80px)]">
        {tracks.map((track) => (
          <div
            key={track.publication.trackSid}
            className="relative bg-gray-800 rounded-lg overflow-hidden"
          >
            {track.source === Track.Source.Camera && (
              <VideoTrack
                trackRef={track}
                className="w-full h-full object-cover"
              />
            )}
            {track.source === Track.Source.Microphone && (
              <AudioTrack trackRef={track} />
            )}
            <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 px-3 py-1 rounded-full text-white text-sm">
              {track.participant.identity}
            </div>
          </div>
        ))}

        {/* Placeholder if no video tracks */}
        {tracks.filter(t => t.source === Track.Source.Camera).length === 0 && (
          <div className="bg-gradient-to-br from-blue-900 to-purple-900 rounded-lg flex items-center justify-center col-span-2">
            <div className="text-center text-white">
              <VideoIcon className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Waiting for video...</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls Bar */}
      <div className="absolute bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center space-x-2">
            <span className="text-white text-sm font-medium">
              Room: {roomInfo.name || 'Unknown'}
            </span>
            <span
              className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${
                connectionState === 'connected'
                  ? 'bg-green-500 text-white'
                  : 'bg-yellow-500 text-white'
              }`}
            >
              {connectionState}
            </span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsAudioEnabled(!isAudioEnabled)}
              className={`p-3 rounded-full transition-colors ${
                isAudioEnabled
                  ? 'bg-gray-700 hover:bg-gray-600'
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {isAudioEnabled ? (
                <Mic className="h-5 w-5 text-white" />
              ) : (
                <MicOff className="h-5 w-5 text-white" />
              )}
            </button>

            <button
              onClick={() => setIsVideoEnabled(!isVideoEnabled)}
              className={`p-3 rounded-full transition-colors ${
                isVideoEnabled
                  ? 'bg-gray-700 hover:bg-gray-600'
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {isVideoEnabled ? (
                <VideoIcon className="h-5 w-5 text-white" />
              ) : (
                <VideoOff className="h-5 w-5 text-white" />
              )}
            </button>

            <button
              onClick={onDisconnect}
              className="p-3 rounded-full bg-red-600 hover:bg-red-700 transition-colors"
            >
              <PhoneOff className="h-5 w-5 text-white" />
            </button>
          </div>

          <div className="w-32"></div>
        </div>
      </div>
    </div>
  );
}
