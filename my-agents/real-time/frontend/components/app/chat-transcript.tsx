'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { type ReceivedMessage, useAgent } from '@livekit/components-react';
import { AgentChatTranscript } from '@/components/agents-ui/agent-chat-transcript';
import { cn } from '@/lib/shadcn/utils';

const MotionContainer = motion.create('div');

const CONTAINER_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeOut',
        duration: 0.3,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut',
        duration: 0.3,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedMessage[];
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  className,
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  const { state: agentState } = useAgent();

  return (
    <div className="flex w-full flex-col h-full">
      <AnimatePresence>
        {!hidden && (
          <MotionContainer
            {...props}
            {...CONTAINER_MOTION_PROPS}
            className={cn('flex h-full w-full flex-col gap-4', className)}
          >
            <AgentChatTranscript
              agentState={agentState}
              messages={messages}
              className="w-full [&_.is-user>div]:rounded-[22px] [&>div>div]:px-4 md:[&>div>div]:px-6"
            />
          </MotionContainer>
        )}
      </AnimatePresence>
    </div>
  );
}
