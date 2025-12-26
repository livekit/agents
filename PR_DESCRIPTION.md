## Summary

Fixes #4243

This PR fixes the phantom VAD activity issue that caused unwanted interruptions when using STT turn detection with `resume_false_interruption=True`.

---

## Problem

When using STT turn detection (especially with Deepgram) and `resume_false_interruption=True`, the agent incorrectly interrupted speech during the false interruption timeout period. This caused:

1. Agent transitions from "thinking" to "listening" without actual user speech
2. The `llm_node` gets cancelled unexpectedly
3. Console mode users particularly affected (~30% reproduction rate)

**User-Reported Behavior:**
> "the agent state changed from thinking to listening without the user state changing to speaking"

---

## Root Cause

In `on_final_transcript()` (agent_activity.py), after pausing the speech and starting the false interruption timer, the code **unconditionally** called `_interrupt_paused_speech()`:

```python
def on_final_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None = None) -> None:
    # ...
    if self._audio_recognition and self._turn_detection not in ("manual", "realtime_llm"):
        self._interrupt_by_audio_activity()  # PAUSES speech when use_pause=True

        if speaking is False and self._paused_speech and timeout:
            self._start_false_interruption_timer(timeout)  # Start timer to resume

    # BUG: This ALWAYS runs, defeating the pause mode!
    self._interrupt_paused_speech_task = asyncio.create_task(
        self._interrupt_paused_speech(...)  # Cancels timer and interrupts!
    )
```

The `_interrupt_paused_speech()` method:
1. Cancels the false interruption timer
2. Calls `interrupt()` on the paused speech

This defeated the entire purpose of the pause mode, which was designed to allow speech to resume if the interruption was a false positive.

---

## Solution

When in pause mode (`resume_false_interruption=True`, `false_interruption_timeout` set, audio supports pause), return early after starting the timer. Let the timer decide whether to:
- **Resume** - if it was a false interruption (no real user speech)
- **Let end-of-turn handle it** - if it was a real interruption

```python
if self._audio_recognition and self._turn_detection not in ("manual", "realtime_llm"):
    self._interrupt_by_audio_activity()

    # NEW: Check if we're in pause mode
    opt = self._session.options
    use_pause = opt.resume_false_interruption and opt.false_interruption_timeout is not None
    can_pause = self._session.output.audio and self._session.output.audio.can_pause

    if use_pause and can_pause and self._paused_speech:
        if speaking is False and (timeout := opt.false_interruption_timeout) is not None:
            self._start_false_interruption_timer(timeout)
        return  # KEY FIX: Don't call _interrupt_paused_speech in pause mode

    # ... existing code for non-pause mode ...
```

---

## Testing

### Unit Tests
All 15 existing agent session tests pass:

```
tests/test_agent_session.py::test_events_and_metrics PASSED
tests/test_agent_session.py::test_tool_call PASSED
tests/test_agent_session.py::test_interruption[False-5.5] PASSED
tests/test_agent_session.py::test_interruption[True-5.5] PASSED
tests/test_agent_session.py::test_interruption_options PASSED
tests/test_agent_session.py::test_interruption_by_text_input PASSED
tests/test_agent_session.py::test_interruption_before_speaking[False-3.5] PASSED
tests/test_agent_session.py::test_interruption_before_speaking[True-3.5] PASSED
tests/test_agent_session.py::test_generate_reply PASSED
tests/test_agent_session.py::test_preemptive_generation[True-0.8] PASSED
tests/test_agent_session.py::test_preemptive_generation[False-1.1] PASSED
tests/test_agent_session.py::test_interrupt_during_on_user_turn_completed[False-0.0] PASSED
tests/test_agent_session.py::test_interrupt_during_on_user_turn_completed[False-2.0] PASSED
tests/test_agent_session.py::test_interrupt_during_on_user_turn_completed[True-0.0] PASSED
tests/test_agent_session.py::test_interrupt_during_on_user_turn_completed[True-2.0] PASSED

======================== 15 passed in 76.00s ========================
```

### What Tests Verify

1. **False interruption with `resume_false_interruption=True`** - Tests that speech correctly resumes after false interruption timeout
2. **False interruption with `resume_false_interruption=False`** - Tests backward compatibility
3. **Interruption options** - Tests various interruption configurations
4. **Preemptive generation** - Tests preemptive generation with interruptions

### Manual Testing Recommended

For production verification, test with:
1. Console mode + Deepgram STT + STT turn detection
2. Verify no phantom "thinking→listening" transitions
3. Verify legitimate interruptions still work correctly

---

## Backward Compatibility

- **No breaking changes** - Only affects users with `resume_false_interruption=True`
- **Default behavior preserved** - Users with `resume_false_interruption=False` see no change
- **Expected behavior change** - Agent now correctly waits for false interruption timeout instead of immediately interrupting

---

## Impact Analysis

**Console Mode**
- Primary fix target - Resolves phantom interruptions
- Users should no longer see unexpected "thinking→listening" transitions

**WebRTC Mode**
- Same fix applies
- Improves false interruption handling for all users with `resume_false_interruption=True`

**Realtime LLM**
- Not affected (already skipped in code at line 1280)

**Manual Turn Detection**
- Not affected (already skipped in code at line 1297-1299)

---

## Edge Cases Handled

1. **User speaks again during timeout** → Timer cancelled in `on_start_of_speech()` (already implemented)
2. **Real end-of-turn detected** → `_user_turn_completed_task` still interrupts correctly
3. **Session close** → `_interrupt_paused_speech` called during cleanup
4. **Audio doesn't support pause** → Falls through to immediate interrupt (correct behavior)

---

## Files Changed

- `livekit-agents/livekit/agents/voice/agent_activity.py` - Fix in `on_final_transcript()` method

---

## Future Considerations

1. **Framework-Wide Audit** - Other methods that call `_interrupt_paused_speech` should be reviewed
2. **Metrics** - Consider adding metrics for false interruption events
3. **Documentation** - Update docs to explain `resume_false_interruption` behavior more clearly
