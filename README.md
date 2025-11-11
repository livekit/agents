### LiveKit is an open-source platform for building real-time audio and video applications using WebRTC. It provides the necessary infrastructure for applications that need scalable, low-latency communication, such as video conferencing, and also includes tools for building more advanced AI-powered applications like voice agents. Developers can use the self-hosted or cloud-managed deployments, client SDKs, and features like selective forwarding to create and customize their real-time

# LiveKit Interrupt-Aware Voice Agent  
**Branch:** `feature/livekit-interrupt-handler-ahmad-raza`

This feature adds **filler-aware voice interrupt handling** to LiveKit Voice Agents.  
The agent can now **ignore meaningless interjections** (e.g., *“um”*, *“haan”*, *“hmm”* from multiple languages - Hindi, Eng, Tamil, Telgu, etc.) while it is speaking, but **stop TTS immediately** when a **real spoken interruption** occurs.

---

## What Changed

### 1. Added **Filler Detection Logic**
- Maintains a **custom dictionary** of culturally diverse filler words.
- Uses STT confidence score to ignore background murmurs.
- Added `is_filler(transcript, confidence)` rule-based classifier.

### 2. Interruption Handling & Auto-Resume

This agent intelligently distinguishes between filler sounds (e.g., "umm", "haan", "hmm") and meaningful interruptions (e.g., "wait", "stop", "hold on") to provide natural turn-taking.

- If the user makes filler sounds while the agent is speaking → the agent continues.
- If the user speaks meaningful words → the agent stops immediately.
- Once the user finishes speaking (VAD silence) → the agent automatically resumes its response.

### Run the Agent
```bash
python interrupt_handler_agent.py dev
```


### 3. Modified Agent Behavior
- Overrode `on_transcription()` inside an `InterruptAgent` class.
- If agent is speaking:
  - Ignore filler-only speech.
  - Stop TTS instantly when meaningful speech is detected.

### 4. Updated Voice Pipeline (to avoid quota + plugin issues)
| Component | Plugin Used | Model |
|----------|------------|--------|
| VAD | `silero` | default vad |
| Speech-to-Text (STT) | `deepgram.STT` | `"nova"` |
| Language Model (LLM) | `groq.LLM` | `"llama-3.1-8b-instant"` |
| Text-to-Speech (TTS) | `deepgram.TTS` | `"aura-asteria-en"` Stable + free tier usable |

---

## Final Working Code

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




IGNORED_FILLERS = {
     "hmm", "hmmm", "hmm", "hmmhmm", "mmm", "mmmm", "mmh", "mm-hmm",
    "mm-mm", "mhm", "mhmm", "hm", "hm?", "hmh", "hmhmm", "uh", "uhh",
    "uhhh", "um", "umm", "ummm", "uh-huh", "uh-uh", "huh", "huh",
    "hnnn", "hunh", "humm", "hummm", "um-hmm", "mm", "mm", "mmhmm",
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
    "appo", "alle", "entha",
    "continue", "go on", "carry on", "keep going", "keep it up", "proceed",
    "move forward", "press on", "push on", "keep at it", "hold on",
    "stay on it", "keep moving", "resume", "persist", "maintain",
    "keep working", "follow through", "advance", "keep progressing",
    "further", "stick with it", "march on", "keep doing", "remain at it",
    "don't stop", "continue onward", "onward", "carry forward", "keep pushing",
    "keep rolling", "go ahead", "take it forward", "move ahead", "back to it",
    "keep steady", "keep going as is", "hold course", "stay the course",
    "keep pace", "progress", "push forward", "maintain momentum",
    "drive ahead", "keep advancing", "keep flowing", "continue the same",
    "keep following", "keep the momentum", "move along",
}

CONFIDENCE_THRESHOLD = 0.60


def is_filler(transcript, confidence):

    words = transcript.lower().strip().split()

    # Ignore low-confidence audio (breathing, background room noise)
    if confidence < CONFIDENCE_THRESHOLD:
        return True

    # Ignore very short hesitation sounds like "mhmm", "mm", "huh"
    if len(words) == 1 and len(words[0]) <= 3:
        return True

    # Ignore utterances made entirely of filler tokens
    if all(w in IGNORED_FILLERS for w in words):
        return True

    # Otherwise → contains meaningful speech → treat as interruption
    return False


class InterruptAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a polite assistant. Speak clearly and conversationally.",
        )
        self.interrupted = False  # NEW

    async def on_transcription(self, event):
        # Only consider interruptions while the agent is speaking
        if self.session.agent.is_speaking:
            transcript = event.text
            confidence = event.confidence or 1.0

            if is_filler(transcript, confidence):
                print(f"[IGNORED FILLER] {transcript}")
                return

            # Meaningful user speech → interrupt
            print(f"[VALID INTERRUPTION] {transcript}")
            self.session.stop_speaking()
            self.interrupted = True  # NEW

    async def on_vad(self, event):
        # event.speech == False → silence detected
        if self.interrupted and not event.speech:
            print("[USER FINISHED SPEAKING] → Resuming agent response...")
            self.interrupted = False
            await self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()


    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=deepgram.TTS(model="aura-asteria-en"),  # ✅ NEW TTS
    )


    agent=InterruptAgent()

    await session.start(
        agent=agent,
        room=ctx.room
    )

    session.on("transcription", agent.on_transcription)
    session.on("vad", agent.on_vad)   # Auto resume on silence


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

```
## 🧪 Steps to Test

1. Start the agent worker:
   ```bash
   python interrupt_handler_agent.py dev
Open LiveKit Console → Sandbox → Web Voice Agent

Connect to your room:

Room name: test-room
While the agent is speaking, try:

Expected Test Behavior
User Says	Agent Speaking?	Expected Behavior
"umm hmm haan"	Yes	Agent continues speaking
"wait one second"	Yes	Agent stops immediately
"umm okay stop"	Yes	Agent stops (contains meaning)
"umm"	No	Agent waits for further speech
You stop speaking	Yes → Silence	Agent resumes talking

## Known Limitations
Issue	Notes
No ML-based filler detection yet	Currently rule-based filtering only
Strong accents may misclassify as real speech	Could be improved via phoneme-based ASR or turn-taking models
TTS may cut its final syllable when interrupted	Expected behavior when enforcing real-time immediate stop

## Environment Details
Component	Version
Python	3.10.11
livekit-agents	1.2.18
OS Tested	Windows 11
Runtime	cli.run_app(dev) interactive mode

## Required Environment Variables
Ensure these are exported before running:

bash
```
export LIVEKIT_URL="wss://YOURPROJECT.livekit.cloud"
export LIVEKIT_API_KEY="YOUR_KEY"
export LIVEKIT_API_SECRET="YOUR_SECRET"
export GROQ_API_KEY="YOUR_GROQ_KEY"
export DEEPGRAM_API_KEY="YOUR_DEEPGRAM_KEY"
If using Windows PowerShell:
```

powershell
```
setx LIVEKIT_URL "wss://YOURPROJECT.livekit.cloud"
setx LIVEKIT_API_KEY "YOUR_KEY"
setx LIVEKIT_API_SECRET "YOUR_SECRET"
setx GROQ_API_KEY "YOUR_GROQ_KEY"
setx DEEPGRAM_API_KEY "YOUR_DEEPGRAM_KEY"
```
