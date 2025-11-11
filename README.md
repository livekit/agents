# LiveKit Interrupt-Aware Voice Agent  
**Branch:** `feature/livekit-interrupt-handler-ahmad-raza`

This feature adds **filler-aware voice interrupt handling** to LiveKit Voice Agents.  
The agent can now **ignore meaningless interjections** (e.g., *“um”*, *“haan”*, *“hmm”* from multiple languages - Hindi, Eng, Tamil, Telgu, etc.) while it is speaking, but **stop TTS immediately** when a **real spoken interruption** occurs.

---

## 🎯 What Changed

### 1. Added **Filler Detection Logic**
- Maintains a **custom dictionary** of culturally diverse filler words.
- Uses STT confidence score to ignore background murmurs.
- Added `is_filler(transcript, confidence)` rule-based classifier.

### 2. Modified Agent Behavior
- Overrode `on_transcription()` inside an `InterruptAgent` class.
- If agent is speaking:
  - Ignore filler-only speech.
  - Stop TTS instantly when meaningful speech is detected.

### 3. Updated Voice Pipeline (to avoid quota + plugin issues)
| Component | Plugin Used | Model |
|----------|------------|--------|
| VAD | `silero` | default vad |
| Speech-to-Text (STT) | `deepgram.STT` | `"nova"` |
| Language Model (LLM) | `groq.LLM` | `"llama-3.1-8b-instant"` |
| Text-to-Speech (TTS) | `deepgram.TTS` | `"aura-asteria-en"` ✅ Stable + free tier usable |

---

## ✅ Final Working Code

```python
import os
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli
)
from livekit.plugins import silero, deepgram, groq

# A multilingual dictionary of filler words.
# These are ignored when deciding whether to interrupt.
IGNORED_FILLERS = {
    "uh", "um", "umm", "hmm", "mmm", "er", "haan",
    "you know", "i mean", "like", "basically", "literally",
    "kinda", "sorta",
    "accha", "yaani", "toh", "matlab", "arey",
    "ano", "eto", "nage", "ehh", "keu-rae", "geu",
    "yani", "bas", "shufu",
    "pues", "donc", "allora", "vale",
    "seri", "apdi", "athu", "enna",
    "ante", "andi", "em",
    "andre", "matte",
    "appo", "alle", "entha"
}

# Minimum STT confidence required to consider a spoken interruption real.
CONFIDENCE_THRESHOLD = 0.60


def is_filler(transcript, confidence):
    """
    Returns True if the user's speech is likely just filler / hesitation.
    Used to prevent false interruptions while the agent is speaking.
    """
    words = transcript.lower().strip().split()

    # If recognition confidence is very low → treat as background / irrelevant sound.
    if confidence < CONFIDENCE_THRESHOLD:
        return True

    # If every word is in the filler dictionary → ignore.
    return all(word in IGNORED_FILLERS for word in words)


class InterruptAgent(Agent):
    """
    Custom agent that listens while speaking and interrupts itself
    only when the user's speech contains meaningful content.
    """
    def __init__(self):
        super().__init__(
            instructions="You are a polite assistant. Speak clearly and conversationally."
        )

    async def on_transcription(self, event):
        # Only react if the agent is currently speaking (i.e., performing TTS).
        if self.session.agent.is_speaking:
            transcript = event.text
            confidence = event.confidence or 1.0

            # Check if the transcript is filler.
            if is_filler(transcript, confidence):
                print(f"[IGNORED FILLER] {transcript}")
                return

            # If meaningful speech is detected → stop speaking immediately.
            print(f"[VALID INTERRUPTION] {transcript}")
            self.session.stop_speaking()


async def entrypoint(ctx: JobContext):
    """
    This function initializes the audio pipeline and starts the agent session.
    """
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),                 # Voice activity detection
        stt=deepgram.STT(model="nova"),        # Real-time speech-to-text
        llm=groq.LLM(model="llama-3.1-8b-instant"),  # Reasoning + text generation
        tts=deepgram.TTS(model="aura-asteria-en"),   # Natural speech synthesis
    )

    agent = InterruptAgent()

    # Begin voice session inside the room.
    await session.start(agent=agent, room=ctx.room)

    # Attach interruption handler.
    session.on("transcription", agent.on_transcription)


if __name__ == "__main__":
    # CLI entrypoint — enables running `python interrupt_handler_agent.py dev`
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
