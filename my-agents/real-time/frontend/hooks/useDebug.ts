import * as React from 'react';
import { LogLevel, setLogLevel } from 'livekit-client';
import { useRoomContext } from '@livekit/components-react';

export const useDebugMode = (options: { logLevel?: LogLevel; enabled?: boolean } = {}) => {
  const room = useRoomContext();
  const logLevel = options.logLevel ?? 'debug';
  const enabled = options.enabled ?? true;

  React.useEffect(() => {
    if (!enabled) {
      setLogLevel('silent');
      return;
    }

    setLogLevel(logLevel ?? 'debug');

    // @ts-expect-error this is a global variable
    window.__lk_room = room;

    return () => {
      // @ts-expect-error this is a global variable
      window.__lk_room = undefined;
      setLogLevel('silent');
    };
  }, [room, enabled, logLevel]);
};
