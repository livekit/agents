# 🎙️Smart Interruption Handler for LiveKit Agents  

---

## 🎯 Overview  

When working with **LiveKit Agents**, I noticed a recurring issue —  
the agent stops talking **as soon as it hears any user speech**, even when the user just says short fillers like *“hmm”*, *“uh”*, or *“yes”*.  

This behavior feels unnatural in voice conversations, because these sounds often **don’t mean “stop”** — they’re just part of how humans listen and respond.  

### 🧩 Problem  
- LiveKit’s default **turn detection** treats *any* speech as an interruption signal.  
- The agent immediately calls `session.interrupt()` whenever VAD detects user audio.  
- Result: The agent stops midway through speaking, even for filler words or background noise.  

### 💡 Solution  
To make the interaction more natural, I implemented a **Smart Interruption Handler** — a layer that listens **semantically**, not just acoustically.  

Instead of stopping on any detected voice, the agent now:
- **Understands** what the user said using semantic embeddings,  
- **Ignores** short fillers like *“hmm”*, *“yes”*, *“okay”*,  
- **Interrupts only** when it detects clear intent like *“stop”*, *“cancel”*, or *“wait”*.  

The result: conversations feel smoother, human-like, and far more responsive.

---

## ⚙️ Solution  

The key difference in my approach is **how** user speech is evaluated.  

### 🔍 Instead of:
Maintaining a hardcoded dictionary or word list for filler and stop words.

### 🚀 I used:
A **semantic approach** with **Sentence Transformers** and **text embeddings**.  
This allows the model to understand the *meaning* of the user’s speech, not just the exact words.  

So whether a user says “please stop talking”, “no, that’s fine”, or “enough already” —  
they all semantically map to the “stop” intent, even if the exact phrase isn’t in our list.  

Meanwhile, words like “uh-huh”, “hmm”, “okay” have embeddings that cluster near filler examples,  
so they’re automatically recognized as **non-interruptive**.

---

## 🧱 Architecture Overview  

Here’s how the final system fits within LiveKit’s voice agent pipeline:

```plaintext
          ┌───────────────────────────────────────────────┐
          │               LiveKit Session                 │
          │-----------------------------------------------│
          │  Audio In  →  VAD → Turn Detector → STT       │
          └────────────┬──────────────────────────────────┘
                       │
                       ▼
              [ Smart Interruption Layer ]
                       │
             ┌────────────────────────────────┐
             │ User Speech → Embedding Model  │
             │ Compare with filler & intent   │
             │ If filler → Ignore             │
             │ If intent → session.interrupt()│
             └────────────────────────────────┘
                       │
                       ▼
          ┌───────────────────────────────────────────────┐
          │             Agent Response Control            │
          │-----------------------------------------------│
          │ Continue Speaking  OR  Stop gracefully        │
          └───────────────────────────────────────────────┘
```
## 🎥 Video
[Watch Demo](https://drive.google.com/file/d/1RS9-F7PXJw2MAxE3jSzB1eVZh-iViieU/view?usp=sharing)

## 🧪 How to Test or Recreate It  

Follow these steps to run the Smart Interruption Handler locally:
Clone the Repository  
```bash
cd livekit-agent-voice
uv sync
```
add the following plugins
```bash
uv add \
  "livekit-agents[silero,turn-detector]~=1.2" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"
```
create .env.local add your livekit keys
```bash
LIVEKIT_API_KEY=<your API Key>
LIVEKIT_API_SECRET=<your API Secret>
LIVEKIT_URL=wss://.livekit.cloud
```
run the command to download weight files
```bash
uv run agent.py download-files
```
Start your agent in console mode to run inside your terminal:
```bash
uv run agent.py console
```

for reference see
- [Voice AI Quick Start](https://docs.livekit.io/agents/start/voice-ai/)

### 📚 Reference Docs
- [Livekit Examples](https://github.com/livekit-examples/agent-starter-python)
- [Turn Detection & Interruptions](https://docs.livekit.io/agents/build/turns/)
- [Text Input Pipeline](https://docs.livekit.io/agents/build/text/)
- [Python SDK Reference](https://docs.livekit.io/reference/python/v1/livekit/agents/)
- [Turn Detector Plugin Docs](https://docs.livekit.io/reference/python/v1/livekit/plugins/turn_detector/)
