"use client";

import { useEffect, useCallback } from "react";
import { Room } from "livekit-client";
import { FormState, RpcMethodName } from "@/lib/rpc-types";

interface RpcHandlers {
  onUpdateField?: (fieldId: string, value: string) => void;
  onGetFormState?: () => Promise<FormState>;
  onSubmitForm?: (data: Record<string, string>) => Promise<boolean>;
}

export function useRpcHandlers(room: Room | null, handlers: RpcHandlers) {
  useEffect(() => {
    if (!room) return;

    // TODO: Update RPC registration for LiveKit v2 API
    // The API has changed - need to use room.localParticipant.registerRpcMethod
    // For now, RPC is disabled until proper implementation
    
    console.log("RPC handlers: Feature requires LiveKit v2 API update");

    // Cleanup
    return () => {
      // Cleanup RPC handlers if needed
    };
  }, [room, handlers]);
}

export function useRpcCall(room: Room | null) {
  return useCallback(
    async <T,>(method: string, data?: unknown): Promise<T> => {
      if (!room) {
        throw new Error("Not connected to room");
      }
      
      // TODO: Update RPC call for LiveKit v2 API
      console.warn("RPC call: Feature requires LiveKit v2 API update");
      throw new Error("RPC not yet implemented for LiveKit v2");
    },
    [room]
  );
}
