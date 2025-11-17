#Overview

This project implements a production-grade Voice Interruption Handling Extension for LiveKit Agents, designed to eliminate false interruptions caused by filler sounds (“uh”, “umm”, “hmm”, “haan”, etc.) while preserving instant, natural, real-time user control when meaningful speech or commands occur.

LiveKit’s built-in VAD pauses TTS anytime any speech-like audio is detected.
This leads to frequent false positives — especially in multilingual environments (English + Hindi + Hinglish).

This extension adds an intelligent interruption layer without modifying the LiveKit SDK, using only public transcription and agent-state events.

What this extension solves

Prevents TTS from stopping on meaningless filler sounds

Immediately interrupts TTS when genuine user intent appears (“stop”, “ruk”, “wait”, “band karo”)

Handles multilingual filler sets (English, Hindi, Hinglish, Devanagari)

Applies ASR confidence filtering for background noise

Provides a configurable and easily expandable architecture

This results in smoother, more natural, human-like conversations with far fewer unwanted pauses.


# Usage

## 1. Attach the interrupt handler to your AgentSession

```python
from livekit_agents.extensions.attach_interrupt_handler import attach_interrupt_handler

session = AgentSession(
    vad=...,
    stt=...,
    llm=...,
    tts=...,
)

handler = attach_interrupt_handler(session)
await session.start(agent=my_agent, room=ctx.room)
```

The extension automatically:

- Subscribes to LiveKit events
  - `user_input_transcribed`
  - `agent_state_changed`
- Tracks whether the agent is currently speaking
- Filters filler audio
- Calls `session.interrupt()` when real interruption is detected

**No changes to your agent code are required.**

---

# Configuration

All behavior is fully configurable using environment variables or passed-in parameters.

You may set any of these before running your agent:

## Ignored filler words (English + Hindi by default)

```bash
export LIVEKIT_IGNORED_WORDS="uh,umm,hmm,haan,um,er,achha,arey"
```

These are ignored **only when the agent is speaking**.

## Hard interruption commands

```bash
export LIVEKIT_COMMAND_WORDS="stop,wait,no,hold on,pause,ruk,ruko,ruk jao,band karo"
```

Any command triggers **immediate interruption**, even if ASR confidence is low.

## ASR confidence threshold

```bash
export LIVEKIT_MIN_CONFIDENCE=0.5
```

Segments below this confidence are treated as noise.

## Language mode

```bash
export LIVEKIT_LANGUAGE_MODE="auto"
```

- `auto` — detect Hindi/English/Hinglish using token analysis
- `en` — English-only mode
- `hi` — Hindi-only mode

## Runtime override (optional)

You can dynamically update settings:

```python
handler.update_config(ignored_words=["umm", "haan"], min_confidence=0.6)
```
