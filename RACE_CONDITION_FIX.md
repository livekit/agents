## Race Condition Fix: Soft-Ack Grace Period Tracking

### Problem
The grace period logic was using `current_text` to determine if a soft-ack was filtered, but `current_text` depends on VAD events firing. If VAD doesn't fire again after filtering a soft-ack, `current_text` remains empty, causing the grace period to incorrectly interrupt the agent.

### Solution
Use a dedicated flag `_soft_ack_detected_in_grace_period` that is:
1. **Reset at the start of each grace period** (when `_check_for_soft_ack_after_delay` begins)
2. **Set to True when a soft-ack is filtered during the grace period** (only if the grace period task is still running)
3. **Reset after the grace period completes** (in the finally block of the async task)
4. **Reset when non-soft-ack input is received** (to handle edge cases)

### Key Changes

**1. Grace Period Initialization** (on_vad_inference_done)
- Local variable `grace_period_soft_ack_detected` stores the state for THIS grace period
- Instance variable `_soft_ack_detected_in_grace_period` is reset after grace period completes (finally block)

**2. Interim/Final Transcript Handlers** (on_interim_transcript, on_final_transcript)
- When filtering a soft-ack, check if a grace period task is active: `if self._vad_grace_period_task and not self._vad_grace_period_task.done()`
- Only set the flag if a grace period is actually running
- This prevents stale flags from previous grace periods

**3. Grace Period Logic** (_check_for_soft_ack_after_delay)
- Uses the local `grace_period_soft_ack_detected` variable which is independent for each grace period
- Always resets the instance flag in finally block to clean up

### Race Condition Scenario (BEFORE)
```
T0: VAD fires, current_text='' → Start grace period
T1: STT produces "Yeah" → Interim filter sets flag to True, returns
T2: Grace period expires → Checks flag=True, blocks interrupt ✓ (but with race condition potential)
T3: Another VAD fires from agent speech → Could set flag again incorrectly
```

### Race Condition Scenario (AFTER)
```
T0: VAD fires → Start grace period (grace_period_soft_ack_detected=False locally)
T1: STT produces "Yeah" → Check if grace period active, set flag=True only if active
T2: Grace period expires → Use local variable, always reset instance flag
T3: Another VAD fires → New grace period, fresh local variable, no cross-contamination
```

### Benefits
- ✅ Each grace period is independent with its own tracking variable
- ✅ No state leakage from previous grace periods
- ✅ Flag automatically cleaned up when grace period completes
- ✅ Multiple concurrent VAD events don't interfere with each other
- ✅ Instance flag only set when an active grace period exists
