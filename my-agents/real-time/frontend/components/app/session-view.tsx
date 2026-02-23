'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useAgent, useSessionContext, useSessionMessages } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AvatarPanel } from '@/components/app/avatar-panel';
import { IntakeForm } from '@/components/app/intake-form';
import { TileLayout } from '@/components/app/tile-layout';
import { useRpcHandlers } from '@/hooks/useRpcHandlers';
import { EMPTY_FORM_DATA, type IntakeFormData } from '@/lib/form-fields';
import {
  AgentControlBar,
  type AgentControlBarControls,
} from '@/components/agents-ui/agent-control-bar';
import { ChatTranscript } from '@/components/app/chat-transcript';

interface SessionViewProps {
  appConfig: AppConfig;
}

const POST_SUBMIT_FALLBACK_MS = 90_000;

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  const { end, room, isConnected } = useSessionContext();
  const agent = useAgent();
  const { messages } = useSessionMessages();
  const [formData, setFormData] = useState<IntakeFormData>(EMPTY_FORM_DATA);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const hasSpokenRef = useRef(false);

  const controls: AgentControlBarControls = {
    leave: true,
    microphone: true,
    chat: false,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsScreenShare,
  };

  useEffect(() => {
    if (!isSubmitted) return;

    const fallbackTimer = setTimeout(() => {
      end();
    }, POST_SUBMIT_FALLBACK_MS);

    return () => clearTimeout(fallbackTimer);
  }, [isSubmitted, end]);

  useEffect(() => {
    if (!isSubmitted) return;

    if (agent.isFinished) {
      end();
      return;
    }

    if (agent.state === 'speaking') {
      hasSpokenRef.current = true;
    }

    if (hasSpokenRef.current && agent.state !== 'speaking') {
      end();
    }
  }, [isSubmitted, agent.state, agent.isFinished, end]);

  useRpcHandlers({
    room,
    isConnected,
    formData,
    setFormData,
    setIsSubmitted,
  });

  return (
    <section
      className="bg-background relative flex h-full w-full flex-col overflow-hidden"
      style={{ zIndex: 'var(--app-z-session)' }}
      {...props}
    >
      {/* Main content: chat/controls left, form right */}
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        {/* Left side: Video tiles, Chat Transcript and Controls */}
        <div className="flex max-h-[45%] w-full min-w-0 shrink-0 flex-col border-b md:max-h-none md:w-[46%] md:border-b-0">
          {/* TileLayout for agent avatar and user camera */}
          <div className="flex-shrink-0 h-[400px] w-full">
            <TileLayout chatOpen={true} />
          </div>
          {/* Chat and Controls Container */}
          <div className="flex flex-1 flex-col p-4 overflow-hidden">
            {/* Chat Transcript */}
            <div className="flex-1 overflow-y-auto mb-4">
              <ChatTranscript
                messages={messages}
                className="space-y-3"
              />
            </div>
            {/* Agent Control Bar */}
            <div className="pt-4 border-t">
              <AgentControlBar
                variant="livekit"
                controls={controls}
                isConnected={isConnected}
                onDisconnect={end}
              />
            </div>
          </div>
        </div>
        {/* Right side: Intake Form */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto">
          <IntakeForm
            formData={formData}
            onFormDataChange={setFormData}
            isSubmitted={isSubmitted}
            onSubmit={() => setIsSubmitted(true)}
            className="flex-1"
          />
        </div>
      </div>
    </section>
  );
};
