'use client';

import { type VariantProps } from 'class-variance-authority';
import { PhoneOffIcon } from 'lucide-react';
import { useSessionContext } from '@livekit/components-react';
import { Button, buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/shadcn/utils';

export interface AgentDisconnectButtonProps
  extends React.ComponentProps<'button'>,
    VariantProps<typeof buttonVariants> {
  icon?: React.ReactNode;
  children?: React.ReactNode;
}

export function AgentDisconnectButton({
  icon,
  size = 'default',
  children,
  onClick,
  ...props
}: AgentDisconnectButtonProps) {
  const { end } = useSessionContext();
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    onClick?.(event);
    end();
  };

  return (
    <Button variant="destructive" size={size} onClick={handleClick} {...props}>
      {icon ?? <PhoneOffIcon />}
      {children ?? <span className={cn(size?.includes('icon') && 'sr-only')}>END CALL</span>}
    </Button>
  );
}
