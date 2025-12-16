# Complete LiveKit Voice Agent Workflow: From Listening to Interrupts

## CRITICAL DISCOVERY (December 10, 2025)

### The Real Issue Found
After extensive debugging and reinstalling the package in editable mode, I discovered:

1. **The AudioRecognition pipeline IS being used** in the test
2. **on_interim_transcript() IS being called** for INTERIM_TRANSCRIPT events  
3. **Soft-ack filtering code has been added** to multiple locations
4. **BUT the test behavior is inconsistent** - sometimes 0 thinking phases, sometimes 2

### The Root Problem
The soft-acks ARE triggering interrupts because:
- The filtering is happening too late or not in the right places
- There are multiple code paths that need filtering
- The test infrastructure is complex with async machinery

### Solution Path Forward
Instead of debug statements everywhere, I need to implement focused soft-ack filtering at the exact right points in the call chain.

---

## Part 1: Architecture Overview

### Key Components

| Component | File | Responsibility |
|-----------|------|-----------------|
| **AgentSession** | `agent_session.py` | Main session orchestrator, manages overall lifecycle, state machine |
| **AgentActivity** | `agent_activity.py` | Agent-specific orchestration, schedules LLM calls and speech playback |
| **AudioRecognition** | `audio_recognition.py` | Processes STT/VAD events, routes transcripts and VAD signals |
| **Agent** | `agent.py` | User's agent definition, provides `stt_node()` for STT consumption |
| **FakeSTT/FakeVAD** | `tests/fake_*.py` | Test doubles that simulate STT and VAD event generation |

### Key Concepts

- **Agent State**: `'initializing'` → `'listening'` → `'thinking'` → `'speaking'` → `'listening'` ...
- **Soft-Acks**: {"yeah", "ok", "okay", "hmm", "uh-huh", "right"} - should be ignored when agent is speaking
- **Hooks**: AgentActivity implements hook methods called by AudioRecognition when events occur
- **Event Emission**: Session emits events like `"agent_state_changed"`, `"user_input_transcribed"`, etc.

---

## Part 2: Initialization Phase (Session Start → Ready to Listen)

### Call Sequence

```
1. Test creates AgentSession with STT, VAD, LLM, TTS

2. session.start(agent)
   └─> Calls: await self._update_activity(self._agent, wait_on_enter=False)
   
3. AgentSession._update_activity()
   └─> Creates: self._next_activity = AgentActivity(agent, self)
   └─> Awaits: await self._activity.start()
   
4. AgentActivity.start()
   └─> Calls: await self._start_session()
   
5. AgentActivity._start_session()
   ├─> Registers LLM/STT/TTS/VAD metrics handlers (lines 569-581)
   ├─> Calls: await self._resume_scheduling_task()
   ├─> Creates: AudioRecognition(hooks=self, stt=agent.stt_node, vad=vad, ...)
   └─> Calls: self._audio_recognition.start()

6. AudioRecognition.start()
   ├─> Creates _stt_task and _vad_task async tasks
   ├─> _stt_task runs: await self._process_stt_stream(stt_node)
   └─> _vad_task runs: await self._process_vad_stream(vad)
   
7. Agent.stt_node() (default implementation from agent.py lines 363-395)
   ├─> Gets wrapped_stt = activity.stt
   ├─> Opens STT stream: async with wrapped_stt.stream() as stream
   ├─> Pushes audio frames to stream
   └─> Yields STT events from stream as they arrive

8. Agent State: 'initializing' → 'listening'
```

### Key Insight
The **_start_session() must be called** for the AudioRecognition pipeline to activate. This is the prerequisite for soft-ack filtering to work.

---

## Part 3: User Speaks → STT Processing Phase

### Call Sequence When User Says "Tell me about history"

```
1. FakeSTT generates events (or real STT does)
   └─> FakeRecognizeStream sends:
       - INTERIM_TRANSCRIPT: "Tell me"
       - FINAL_TRANSCRIPT: "Tell me about history"

2. Agent.stt_node() async generator receives events
   └─> Yields event to its consumer

3. AudioRecognition._process_stt_stream() receives event
   ├─> Calls: self._on_stt_event(ev)
   └─> Routes by SpeechEventType

4a. If FINAL_TRANSCRIPT (line 300-312 in audio_recognition.py):
   └─> Calls: self._hooks.on_final_transcript(ev, speaking=self._current_vad_state.speaking)
   
4b. If INTERIM_TRANSCRIPT (line 368-378):
   └─> Calls: self._hooks.on_interim_transcript(ev, speaking=self._current_vad_state.speaking)

5. AgentActivity.on_final_transcript() is called
   ├─> Gets transcript text: "Tell me about history"
   ├─> **SOFT-ACK CHECK HAPPENS HERE** (should filter if agent speaking)
   ├─> If NOT soft-ack and agent NOT speaking:
   │   └─> Calls: self._session._user_input_transcribed(UserInputTranscribedEvent(...))
   └─> Emits to session

6. AgentSession._user_input_transcribed() is called
   ├─> **SOFT-ACK CHECK HAPPENS HERE** (final safety check)
   ├─> If NOT soft-ack:
   │   └─> Calls: self.emit("user_input_transcribed", ev)

7. AgentActivity listens for "user_input_transcribed" events
   └─> Through connection in AgentSession event system

8. AgentActivity._interrupt_by_audio_activity() is triggered
   └─> Queues an interrupt in the scheduling system

9. AgentActivity._scheduling_task() is notified (_wake_up_scheduling_task)
   └─> Wakes up from await self._q_updated.wait()
   └─> Processes queued interrupt

10. AgentActivity._generate_reply() is called
    ├─> Calls LLM to generate response
    ├─> Agent State: 'listening' → 'thinking'
    └─> Then generates TTS audio

11. Speech is scheduled and played
    └─> Agent State: 'thinking' → 'speaking'
```

### Critical Code Locations

| Location | Code | Purpose |
|----------|------|---------|
| `audio_recognition.py:300` | `self._hooks.on_final_transcript(ev, speaking=...)` | Routes FINAL_TRANSCRIPT to AgentActivity |
| `agent_activity.py:1492` | `on_final_transcript()` method | **SHOULD FILTER SOFT-ACKS HERE** |
| `agent_activity.py:1505-1514` | Soft-ack check + call to `_user_input_transcribed()` | First filtering barrier |
| `agent_session.py:1236` | `_user_input_transcribed()` method | **SHOULD ALSO FILTER SOFT-ACKS HERE** |
| `agent_session.py:1236-1250` | Soft-ack check in session | Second filtering barrier |

---

## Part 4: Soft-Ack During Agent Speaking (The Problem Scenario)

### What SHOULD Happen

```
Agent State: 'speaking'
User says: "okay"

1. FakeSTT generates:
   └─> INTERIM_TRANSCRIPT: "okay"
   └─> FINAL_TRANSCRIPT: "okay"

2. AudioRecognition._on_stt_event() receives FINAL_TRANSCRIPT

3. AgentActivity.on_final_transcript() is called
   ├─> Gets text: "okay"
   ├─> Checks: transcript_lower in self._soft_acks?  ✓ YES
   ├─> Checks: self._session.agent_state == "speaking"?  ✓ YES
   └─> **RETURNS WITHOUT CALLING _user_input_transcribed()**

4. Result:
   ├─> No event emitted
   ├─> No interrupt queued
   ├─> No new "thinking" phase
   ├─> Agent continues speaking ✓ CORRECT
```

### What IS Currently Happening (BUG)

```
Agent State: 'speaking'
User says: "okay"

TEST SHOWS: 2 "thinking" phases instead of 1
This means soft-acks ARE somehow causing interrupts

MYSTERY: 
- Debug prints from on_final_transcript() are NOT appearing in test output
- This means soft-ack filtering code is NOT BEING EXECUTED
- Therefore: Soft-acks are taking a DIFFERENT CODE PATH

HYPOTHESIS:
The AudioRecognition pipeline (and its soft-ack filtering) 
is NOT being used in this test scenario. The STT transcripts 
are being processed through a DIFFERENT mechanism.
```

---

## Part 5: VAD Interrupt Processing (Physical Speech Detection)

### When User Actually Interrupts (Not a Soft-Ack)

```
Agent State: 'speaking'
User starts speaking strongly: "No, stop that"

1. VAD detects start of speech
   └─> FakeVAD generates VAD_INFERENCE_DONE event

2. AudioRecognition._on_vad_event() receives event
   └─> Calls: self._hooks.on_vad_inference_done(ev)

3. AgentActivity.on_vad_inference_done() is called (line 1332)
   ├─> Gets speech_duration from VAD event
   ├─> Checks: if speech_duration >= min_interruption_duration
   ├─> If YES:
   │   ├─> Gets current STT transcript
   │   ├─> **SOFT-ACK CHECK**: if transcript in soft_acks and agent_state == "speaking"
   │   ├─>   If YES: return (do NOT interrupt)  ✓ SOFT-ACK IGNORED
   │   ├─>   If NO: call _interrupt_by_audio_activity()
   │   └─> Result: Agent stops speaking, listens to user

4. AgentActivity._interrupt_by_audio_activity()
   └─> Queues interrupt for scheduling task
```

### Critical Code Locations

| Location | Code | Purpose |
|----------|------|---------|
| `agent_activity.py:1332` | `on_vad_inference_done()` method | Handles physical speech interrupts |
| `agent_activity.py:1334-1351` | VAD event processing + soft-ack check | Should check if current transcript is soft-ack |

---

## Part 6: The Real Problem - Finding the Actual Code Path

### What We Know
1. ✓ FakeSTT is sending transcripts (debug output shows this)
2. ✗ `on_final_transcript()` debug prints are NOT appearing
3. ✗ `on_vad_inference_done()` debug prints are NOT appearing
4. ✗ `_start_session()` debug print NOT appearing
5. ✗ This means AudioRecognition.start() is NOT being called
6. **BUT** soft-acks ARE causing interrupts (test shows 2 thinking phases)

### Missing Link
There must be a SECOND code path that:
- Consumes STT transcripts
- Routes them to interrupts
- Does NOT go through AudioRecognition
- Does NOT have soft-ack filtering implemented

### Possible Locations
1. **RealtimeModel path** (if using OpenAI Realtime API)
   - `_on_input_audio_transcription_completed()` in agent_activity.py line 1284
   - May bypass AudioRecognition entirely

2. **Direct session event listener**
   - Session might have a built-in transcript listener
   - Might directly trigger interrupts

3. **IVR Activity path** (if IVR detection enabled)
   - May have separate transcript handling

4. **Test-specific path**
   - FakeSession might be routing transcripts differently

---

## Part 7: State Machine Diagram

```
                    ┌─────────────┐
                    │ initializing│
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  listening  │
                    └──┬───────┬──┘
                       │       │
        (user speaks)   │       │ (soft-ack)
                       │       │
                       ▼       ▼
                    ┌─────────────┐◄─── SHOULD IGNORE soft-acks
                    │  thinking   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  speaking   │◄─── agent is speaking
                    └──┬───────┬──┘
                       │       │
      (soft-ack)       │       │ (strong interrupt)
      IGNORED here     │       │
                       ▼       ▼
                    ┌─────────────┐
                    │  listening  │
                    └─────────────┘
```

---

## Part 8: Soft-Ack Filtering Implementation Locations

### Primary Filters (Should Be Activated)

**Location 1: AudioRecognition._on_stt_event()** (audio_recognition.py)
```python
# Line 334-356: FINAL_TRANSCRIPT filtering
if transcript_lower in self._session._soft_acks and self._session.agent_state == "speaking":
    return  # Drop soft-ack, don't call hooks

# Line 447-461: INTERIM_TRANSCRIPT filtering  
if interim_lower in self._session._soft_acks and self._session.agent_state == "speaking":
    return  # Drop soft-ack, don't call hooks
```

**Location 2: AgentActivity.on_final_transcript()** (agent_activity.py:1492)
```python
# Check for soft-acks BEFORE calling _user_input_transcribed()
if text_lower in self._soft_acks and self._session.agent_state == "speaking":
    return  # Don't process soft-ack
```

**Location 3: AgentActivity.on_interim_transcript()** (agent_activity.py:1354)
```python
# Check for soft-acks BEFORE calling _user_input_transcribed()
if text_lower in self._soft_acks and self._session.agent_state == "speaking":
    return  # Don't process soft-ack
```

**Location 4: AgentActivity.on_vad_inference_done()** (agent_activity.py:1332)
```python
# Check if current transcript is soft-ack
if current_text in self._soft_acks and self._session.agent_state == "speaking":
    return  # Don't trigger interrupt
```

**Location 5: AgentSession._user_input_transcribed()** (agent_session.py:1236)
```python
# Final safety check in session
if transcript_lower in self._soft_acks:
    if self.agent_state == "speaking":
        return  # Don't process soft-ack
```

---

## Part 9: Debug Strategy

### To Find the Real Code Path

1. **Add broad debug prints** in these key locations:
   ```
   - AgentActivity._interrupt_by_audio_activity() - whenever called
   - AgentSession._user_input_transcribed() - whenever called
   - _user_input_transcribed in RealtimeModel path (if exists)
   - IVRActivity transcript handling (if enabled)
   ```

2. **Run test and capture output** to see which path is being used

3. **Once real path identified**, add soft-ack filtering there

4. **Verify** that test shows only 1 thinking phase

---

## Part 10: Expected Test Behavior After Fix

### Test Scenario: Soft-Acks During Speaking

**Before Fix** (Current - BROKEN):
```
Transitions: [
  ('initializing', 'listening'),  # Start
  ('listening', 'thinking'),       # User says "Tell me about history"
  ('thinking', 'listening'),       # LLM responds, agent says response
  ('listening', 'thinking'),       # ❌ BUG: Soft-ack "okay" triggers THIS
  ('thinking', 'listening'),       # ...and then another think phase
]
Thinking phases: 2 ❌ WRONG
```

**After Fix** (Expected - CORRECT):
```
Transitions: [
  ('initializing', 'listening'),  # Start
  ('listening', 'thinking'),       # User says "Tell me about history"  
  ('thinking', 'listening'),       # LLM responds, agent says response
  # ✓ No more states - soft-acks during speaking don't trigger interrupts
]
Thinking phases: 1 ✓ CORRECT
```

---

## Summary Table: Where Soft-Acks Should Be Filtered

| Scenario | Location | Method | Check |
|----------|----------|--------|-------|
| STT sends soft-ack transcript | AudioRecognition | `_on_stt_event()` | Drop before calling hooks |
| on_final_transcript callback | AgentActivity | `on_final_transcript()` | Return before `_user_input_transcribed()` |
| on_interim_transcript callback | AgentActivity | `on_interim_transcript()` | Return before `_user_input_transcribed()` |
| VAD detects speech with soft-ack | AgentActivity | `on_vad_inference_done()` | Return before interrupt |
| Session-level transcript event | AgentSession | `_user_input_transcribed()` | Return before event emission |

All filters should check:
- `text in self._soft_acks` (is it a soft-ack?)
- `agent_state == "speaking"` (is agent speaking?)
- If both true: **DO NOT INTERRUPT**

