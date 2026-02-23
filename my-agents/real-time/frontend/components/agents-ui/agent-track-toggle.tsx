import { type ComponentProps, Fragment } from 'react';
import { type VariantProps, cva } from 'class-variance-authority';
import { Track } from 'livekit-client';
import {
  LoaderIcon,
  MicIcon,
  MicOffIcon,
  MonitorOffIcon,
  MonitorUpIcon,
  VideoIcon,
  VideoOffIcon,
} from 'lucide-react';
import { Toggle, toggleVariants } from '@/components/ui/toggle';
import { cn } from '@/lib/shadcn/utils';

export const agentTrackToggleVariants = cva(['size-9'], {
  variants: {
    variant: {
      default: [
        'data-[state=off]:bg-destructive/10 data-[state=off]:text-destructive',
        'data-[state=off]:hover:bg-destructive/15',
        'data-[state=off]:focus-visible:ring-destructive/30',
        'data-[state=on]:bg-accent data-[state=on]:text-accent-foreground',
        'data-[state=on]:hover:bg-foreground/10',
      ],
      outline: [
        'data-[state=off]:bg-destructive/10 data-[state=off]:text-destructive data-[state=off]:border-destructive/20',
        'data-[state=off]:hover:bg-destructive/15 data-[state=off]:hover:text-destructive',
        'data-[state=off]:focus:text-destructive',
        'data-[state=off]:focus-visible:border-destructive data-[state=off]:focus-visible:ring-destructive/30',
        'data-[state=on]:hover:bg-foreground/10 data-[state=on]:hover:border-foreground/12',
        'dark:data-[state=on]:hover:bg-foreground/10',
      ],
    },
  },
  defaultVariants: {
    variant: 'default',
  },
});

function getSourceIcon(source: Track.Source, enabled: boolean, pending = false) {
  if (pending) {
    return LoaderIcon;
  }

  switch (source) {
    case Track.Source.Microphone:
      return enabled ? MicIcon : MicOffIcon;
    case Track.Source.Camera:
      return enabled ? VideoIcon : VideoOffIcon;
    case Track.Source.ScreenShare:
      return enabled ? MonitorUpIcon : MonitorOffIcon;
    default:
      return Fragment;
  }
}

/**
 * Props for the AgentTrackToggle component.
 */
export type AgentTrackToggleProps = VariantProps<typeof toggleVariants> &
  ComponentProps<'button'> & {
    /**
     * The variant of the toggle.
     * @defaultValue 'default'
     */
    variant?: 'default' | 'outline';
    /**
     * The track source to toggle (Microphone, Camera, or ScreenShare).
     */
    source: 'camera' | 'microphone' | 'screen_share';
    /**
     * Whether the toggle is in a pending/loading state.
     * When true, displays a loading spinner icon.
     * @defaultValue false
     */
    pending?: boolean;
    /**
     * Whether the toggle is currently pressed/enabled.
     * @defaultValue false
     */
    pressed?: boolean;
    /**
     * The default pressed state when uncontrolled.
     * @defaultValue false
     */
    defaultPressed?: boolean;
    /**
     * Callback fired when the pressed state changes.
     */
    onPressedChange?: (pressed: boolean) => void;
  };

/**
 * A toggle button for controlling track publishing state.
 * Displays appropriate icons based on the track source and state.
 *
 * @extends ComponentProps<'button'>
 *
 * @example
 * ```tsx
 * <AgentTrackToggle
 *   source={Track.Source.Microphone}
 *   pressed={isMicEnabled}
 *   onPressedChange={(pressed) => setMicEnabled(pressed)}
 * />
 * ```
 */
export function AgentTrackToggle({
  size = 'default',
  variant = 'default',
  source,
  pending = false,
  pressed = false,
  defaultPressed = false,
  className,
  onPressedChange,
  ...props
}: AgentTrackToggleProps) {
  const IconComponent = getSourceIcon(source as Track.Source, pressed ?? false, pending);

  return (
    <Toggle
      size={size}
      variant={variant}
      pressed={pressed}
      defaultPressed={defaultPressed}
      aria-label={`Toggle ${source}`}
      onPressedChange={onPressedChange}
      className={cn(
        agentTrackToggleVariants({
          variant: variant ?? 'default',
          className,
        })
      )}
      {...props}
    >
      <IconComponent className={cn(pending && 'animate-spin')} />
      {props.children}
    </Toggle>
  );
}
