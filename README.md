# SalesCode.ai - Smart Interruption Handler Solution

This agent demonstrates a robust, context-aware interruption handling system for the LiveKit Agents framework. It successfully distinguishes between meaningful user speech and simple filler words ("um", "haan", "okay"), solving the core challenge and both bonus objectives.

## 🎯 Objective

The goal was to prevent filler words from falsely interrupting the agent's speech, while still allowing genuine interruptions.

## 💡 The Solution: The "VADectomy" & Smart STT Gatekeeper

The core problem is that the default AgentSession uses a raw Voice Activity Detection (VAD) "fast path" to trigger interruptions. This is too sensitive and fires on any noise, including its own echo or user fillers, before the STT can analyze the words.

My solution defeats this by:

### Performing a "VADectomy"

In the AgentSession setup, I set `vad=None`. This surgically removes the "dumb" fast path, forcing the agent to rely only on the "smart" STT-based path for interruption signals.

### Creating RobustAgent

I subclassed the standard Agent to create RobustAgent.

### Overriding stt_node

This RobustAgent intercepts the STT event stream. It acts as a "gatekeeper" for the START_OF_SPEECH event, which is now the only event that can trigger an interruption.

### Implementing Context-Aware Logic

The `_is_speaking()` method checks if the agent is currently playing audio.

**If Agent is Speaking:** The gatekeeper holds all START_OF_SPEECH events and inspects the first few transcribed words.

- If the words are in the IGNORED_FILLERS list, it discards the event (🛡️ IGNORED FILLER).
- If the words are "real," it releases the event, triggering a valid interruption (🔓 GATE OPEN).

**If Agent is Silent:** The gatekeeper is disabled, and all speech events pass through instantly (🟢 PASS-THROUGH), allowing for normal conversation.

## 🏆 Bonus Challenges Solved

### Scalable & Language-Agnostic

The IGNORED_FILLERS list is dynamically loaded from a .env variable (IGNORED_FILLERS_CSV). This allows the filter to be configured for any language (e.g., Hinglish: "haan", "acha") without touching the agent's code.

### Dynamic Runtime Updates

RobustAgent is equipped with two LLM function tools: `add_ignored_filler(word)` and `remove_ignored_filler(word)`. A user can simply tell the agent, "Please ignore the word 'test'," and the agent will update its filter list live, mid-call.

## ⚙️ How to Run

### 1. Setup Environment

```bash
# 1. Clone your fork
git clone ...
cd agents

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install "livekit-agents[openai,silero,deepgram,turn-detector]" python-dotenv
```

### 2. Configure .env File

Create a `.env` file in the root agents directory with your API keys and the filler list.

```
# LiveKit Credentials
LIVEKIT_URL=wss://YOUR-PROJECT.livekit.cloud
LIVEKIT_API_KEY=API_KEY
LIVEKIT_API_SECRET=API_SECRET

# Model Credentials
OPENAI_API_KEY=sk-YOUR_KEY
DEEPGRAM_API_KEY=YOUR_KEY

# --- Solution Configuration ---
# Comma-separated list of multi-language fillers to ignore
IGNORED_FILLERS_CSV=um,umm,uh,hmm,hmmm,haan,han,achha,acha,ok,okay,like,yeah,right,a,i
```

### 3. Run the Agent

```bash
python3 submission_agent.py dev
```

## 🧪 How to Test the Solution

**IMPORTANT:** You must wear headphones for testing. This prevents the agent's own audio from creating echo and confusing the STT.

### Test 1: Real Speech - Agent Silent

**Connect:** Open the LiveKit Agents Playground and connect.

**Start Agent:** The agent will greet you.

**Action:** Say "Hello".

**Expected:** Agent responds. Logs show 🟢 PASS-THROUGH.

### Test 2: Filler - Agent Speaking

**Action:** Ask "Tell me a long story about the Roman Empire." While it is speaking, say "Hmm...".

**Expected:** Agent keeps talking. Logs show 🛡️ IGNORED FILLER.

### Test 3: Real Command - Agent Speaking

**Action:** While it is still talking, say "Wait, stop that."

**Expected:** Agent stops immediately. Logs show 🔓 GATE OPEN.

### Test 4: Bonus - Dynamic Update

**Action:** When it's your turn, say: "Please add 'banana' to the ignored fillers list."

**Expected:** Agent calls the tool and confirms. Logs show Dynamically ADDED filler: 'banana'.

### Test 5: Verify Update

**Action:** Ask for another long story. While it speaks, say "Banana."

**Expected:** Agent keeps talking. Logs show 🛡️ IGNORED FILLER ... 'banana'.
