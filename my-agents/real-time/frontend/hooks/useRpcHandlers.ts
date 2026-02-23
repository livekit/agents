'use client';

import { useEffect, useRef } from 'react';
import { RpcError, type RpcInvocationData } from 'livekit-client';
import type { Room } from 'livekit-client';
import type { IntakeFormData } from '@/lib/form-fields';
import { isValidFieldName } from '@/lib/form-fields';

interface UseRpcHandlersOptions {
  room: Room | undefined;
  isConnected: boolean;
  formData: IntakeFormData;
  setFormData: (data: IntakeFormData | ((prev: IntakeFormData) => IntakeFormData)) => void;
  setIsSubmitted: (submitted: boolean) => void;
}

export function useRpcHandlers({
  room,
  isConnected,
  formData,
  setFormData,
  setIsSubmitted,
}: UseRpcHandlersOptions) {
  const formDataRef = useRef(formData);
  formDataRef.current = formData;

  useEffect(() => {
    if (!room || !isConnected) {
      return;
    }

    room.registerRpcMethod('updateField', async (data: RpcInvocationData) => {
      try {
        const { fieldName, value } = JSON.parse(data.payload) as {
          fieldName: string;
          value: string;
        };
        if (!isValidFieldName(fieldName)) {
          throw new RpcError(1500, 'Invalid field name', JSON.stringify({ fieldName }));
        }
        setFormData((prev) => ({ ...prev, [fieldName]: value }));
        return JSON.stringify({ success: true, fieldName, value });
      } catch (error) {
        if (error instanceof RpcError) throw error;
        throw new RpcError(1500, 'Failed to update field');
      }
    });

    room.registerRpcMethod('getFormState', async () => {
      return JSON.stringify(formDataRef.current);
    });

    room.registerRpcMethod('submitForm', async () => {
      setIsSubmitted(true);
      return JSON.stringify({ success: true });
    });

    return () => {
      room.unregisterRpcMethod('updateField');
      room.unregisterRpcMethod('getFormState');
      room.unregisterRpcMethod('submitForm');
    };
  }, [room, isConnected, setFormData, setIsSubmitted]);
}
