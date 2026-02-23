'use client';

import { useEffect, useMemo, useState } from 'react';
import { type VariantProps, cva } from 'class-variance-authority';
import { LocalAudioTrack, LocalVideoTrack } from 'livekit-client';
import { useMaybeRoomContext, useMediaDeviceSelect } from '@livekit/components-react';
import { AgentTrackToggle } from '@/components/agents-ui/agent-track-toggle';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toggleVariants } from '@/components/ui/toggle';
import { cn } from '@/lib/shadcn/utils';

const selectVariants = cva(
  [
    'rounded-l-none shadow-none pl-2 ',
    'text-foreground hover:text-muted-foreground',
    'peer-data-[state=on]/track:bg-muted peer-data-[state=on]/track:hover:bg-foreground/10',
    'peer-data-[state=off]/track:text-destructive',
    'peer-data-[state=off]/track:focus-visible:border-destructive peer-data-[state=off]/track:focus-visible:ring-destructive/30',
    '[&_svg]:opacity-100',
  ],
  {
    variants: {
      variant: {
        default: [
          'border-none',
          'peer-data-[state=off]/track:bg-destructive/10',
          'peer-data-[state=off]/track:hover:bg-destructive/15',
          'peer-data-[state=off]/track:[&_svg]:!text-destructive',

          'dark:peer-data-[state=on]/track:bg-accent',
          'dark:peer-data-[state=on]/track:hover:bg-foreground/10',
          'dark:peer-data-[state=off]/track:bg-destructive/10',
          'dark:peer-data-[state=off]/track:hover:bg-destructive/15',
        ],
        outline: [
          'border border-l-0',
          'peer-data-[state=off]/track:border-destructive/20',
          'peer-data-[state=off]/track:bg-destructive/10',
          'peer-data-[state=off]/track:hover:bg-destructive/15',
          'peer-data-[state=off]/track:[&_svg]:!text-destructive',
          'peer-data-[state=on]/track:hover:border-foreground/12',

          'dark:peer-data-[state=off]/track:bg-destructive/10',
          'dark:peer-data-[state=off]/track:hover:bg-destructive/15',
          'dark:peer-data-[state=on]/track:bg-accent',
          'dark:peer-data-[state=on]/track:hover:bg-foreground/10',
        ],
      },
      size: {
        default: 'w-[180px]',
        sm: 'w-auto',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

/**
 * Props for the TrackDeviceSelect component. */
type TrackDeviceSelectProps = React.ComponentProps<typeof SelectTrigger> &
  VariantProps<typeof selectVariants> & {
    /**
     * The size of the select.
     * @defaultValue 'default'
     */
    size?: 'default' | 'sm';
    /**
     * The variant of the select.
     * @defaultValue 'default'
     */
    variant?: 'default' | 'outline' | null;
    /**
     * The type of media device (audioinput or videoinput).
     */
    kind: MediaDeviceKind;
    /**
     * The track source to control (Microphone, Camera, or ScreenShare).
     */
    track?: LocalAudioTrack | LocalVideoTrack | undefined;
    /**
     * Whether to request permissions for the media device.
     */
    requestPermissions?: boolean;
    /**
     * Callback when a media device error occurs.
     */
    onMediaDeviceError?: (error: Error) => void;
    /**
     * Callback when the device list changes.
     */
    onDeviceListChange?: (devices: MediaDeviceInfo[]) => void;
    /**
     * Callback when the active device changes.
     */
    onActiveDeviceChange?: (deviceId: string) => void;
  };

/**
 * A select component for selecting a media device.
 *
 * @extends ComponentProps<'button'>
 *
 * @example
 * ```tsx
 * <TrackDeviceSelect
 *   size="sm"
 *   variant="outline"
 *   kind="audioinput"
 *   track={micTrackRef}
 * />
 * ```
 */
function TrackDeviceSelect({
  kind,
  track,
  size = 'default',
  variant = 'default',
  className,
  requestPermissions = false,
  onMediaDeviceError,
  onDeviceListChange,
  onActiveDeviceChange,
  ...props
}: TrackDeviceSelectProps) {
  const room = useMaybeRoomContext();
  const [open, setOpen] = useState(false);
  const [requestPermissionsState, setRequestPermissionsState] = useState(requestPermissions);
  const { devices, activeDeviceId, setActiveMediaDevice } = useMediaDeviceSelect({
    room,
    kind,
    track,
    requestPermissions: requestPermissionsState,
    onError: onMediaDeviceError,
  });

  useEffect(() => {
    onDeviceListChange?.(devices);
  }, [devices, onDeviceListChange]);

  const handleOpenChange = (open: boolean) => {
    setOpen(open);
    if (open) {
      setRequestPermissionsState(true);
    }
  };

  const handleActiveDeviceChange = (deviceId: string) => {
    setActiveMediaDevice(deviceId);
    onActiveDeviceChange?.(deviceId);
  };

  const filteredDevices = useMemo(() => devices.filter((d) => d.deviceId !== ''), [devices]);

  if (filteredDevices.length < 2) {
    return null;
  }

  return (
    <Select
      open={open}
      value={activeDeviceId}
      onOpenChange={handleOpenChange}
      onValueChange={handleActiveDeviceChange}
    >
      <SelectTrigger className={cn(selectVariants({ size, variant }), className)} {...props}>
        {size !== 'sm' && (
          <SelectValue className="font-mono text-sm" placeholder={`Select a ${kind}`} />
        )}
      </SelectTrigger>
      <SelectContent position="popper">
        {filteredDevices.map((device) => (
          <SelectItem key={device.deviceId} value={device.deviceId} className="font-mono text-xs">
            {device.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/**
 * Props for the AgentTrackControl component.
 */
export type AgentTrackControlProps = VariantProps<typeof toggleVariants> & {
  /**
   * The type of media device (audioinput or videoinput).
   */
  kind: MediaDeviceKind;
  /**
   * The track source to control (Microphone, Camera, or ScreenShare).
   */
  source: 'camera' | 'microphone' | 'screen_share';
  /**
   * Whether the track is currently enabled/published.
   */
  pressed?: boolean;
  /**
   * Whether the control is in a pending/loading state.
   */
  pending?: boolean;
  /**
   * Whether the control is disabled.
   */
  disabled?: boolean;
  /**
   * Additional CSS class names to apply to the container.
   */
  className?: string;
  /**
   * Callback when the pressed state changes.
   */
  onPressedChange?: (pressed: boolean) => void;
  /**
   * Callback when a media device error occurs.
   */
  onMediaDeviceError?: (error: Error) => void;
  /**
   * Callback when the active device changes.
   */
  onActiveDeviceChange?: (deviceId: string) => void;
};

/**
 * A combined track toggle and device selector control.
 * Includes a toggle button and a dropdown to select the active device.
 * For microphone tracks, displays an audio visualizer.
 *
 * @example
 * ```tsx
 * <AgentTrackControl
 *   kind="audioinput"
 *   source={Track.Source.Microphone}
 *   pressed={isMicEnabled}
 *   audioTrack={micTrackRef}
 *   onPressedChange={(pressed) => setMicEnabled(pressed)}
 *   onActiveDeviceChange={(deviceId) => setMicDevice(deviceId)}
 * />
 * ```
 */
export function AgentTrackControl({
  kind,
  variant = 'default',
  source,
  pressed,
  pending,
  disabled,
  className,
  onPressedChange,
  onMediaDeviceError,
  onActiveDeviceChange,
}: AgentTrackControlProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-0 rounded-md',
        variant === 'outline' && 'shadow-xs [&_button]:shadow-none',
        className
      )}
    >
      <AgentTrackToggle
        variant={variant ?? 'default'}
        source={source}
        pressed={pressed}
        pending={pending}
        disabled={disabled}
        onPressedChange={onPressedChange}
        className="peer/track group/track focus:z-10 has-[~_button]:rounded-r-none has-[~_button]:border-r-0 has-[~_button]:pr-2 has-[~_button]:pl-3"
      />
      {kind && (
        <TrackDeviceSelect
          size="sm"
          kind={kind}
          variant={variant}
          requestPermissions={false}
          onMediaDeviceError={onMediaDeviceError}
          onActiveDeviceChange={onActiveDeviceChange}
          className={cn([
            'relative',
            'before:bg-border before:absolute before:inset-y-0 before:left-0 before:my-2.5 before:w-px has-[~_button]:before:content-[""]',
            !pressed && 'before:bg-destructive/20',
          ])}
        />
      )}
    </div>
  );
}
