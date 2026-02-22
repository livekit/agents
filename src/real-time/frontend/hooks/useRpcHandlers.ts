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

    // Register updateField RPC method
    if (handlers.onUpdateField) {
      room.registerRpcMethod(
        "updateField",
        async (data: { fieldId: string; value: string }) => {
          handlers.onUpdateField!(data.fieldId, data.value);
          return { success: true };
        }
      );
    }

    // Register getFormState RPC method
    if (handlers.onGetFormState) {
      room.registerRpcMethod("getFormState", async () => {
        const state = await handlers.onGetFormState!();
        return state;
      });
    }

    // Register submitForm RPC method
    if (handlers.onSubmitForm) {
      room.registerRpcMethod(
        "submitForm",
        async (data: Record<string, string>) => {
          const success = await handlers.onSubmitForm!(data);
          return { success };
        }
      );
    }

    return () => {
      // Cleanup if needed
    };
  }, [room, handlers]);
}

export function useRpcCall(room: Room | null) {
  return useCallback(
    async <T,>(method: string, data?: unknown): Promise<T> => {
      if (!room) {
        throw new Error("Not connected to room");
      }
      return room.sendRpc(method, data) as Promise<T>;
    },
    [room]
  );
}
