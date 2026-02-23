import { type ComponentProps } from 'react';
import { Room } from 'livekit-client';
import { useEnsureRoom, useStartAudio } from '@livekit/components-react';
import { Button } from '@/components/ui/button';

/**
 * Props for the StartAudioButton component.
 */
export interface StartAudioButtonProps extends ComponentProps<'button'> {
  /**
   * The size of the button.
   * @defaultValue 'default'
   */
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'icon-sm' | 'icon-lg';
  /**
   * The variant of the button.
   * @defaultValue 'default'
   */
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  /**
   * The LiveKit room instance. If not provided, uses the room from context.
   */
  room?: Room;
  /**
   * The label text to display on the button.
   */
  label: string;
}

/**
 * A button that allows users to start audio playback.
 * Required for browsers that block autoplay of audio.
 * Only renders when audio playback is blocked.
 *
 * @extends ComponentProps<'button'>
 *
 * @example
 * ```tsx
 * <StartAudioButton label="Click to allow audio playback" />
 * ```
 */
export function StartAudioButton({
  size = 'default',
  variant = 'default',
  label,
  room,
  ...props
}: StartAudioButtonProps) {
  const roomEnsured = useEnsureRoom(room);
  const { mergedProps } = useStartAudio({ room: roomEnsured, props });

  return (
    <Button size={size} variant={variant} {...mergedProps}>
      {label}
    </Button>
  );
}
