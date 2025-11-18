# 🎙️ LiveKit Voice Agent — Interruption Handling Layer
### **NSUT Internship Assignment — Final Submission - By Yash Gupta**

This project enhances a standard LiveKit voice agent by adding an **interruption handling layer** which intelligently distinguishes between *filler utterances* and *real interruption commands* while strictly keeping LiveKit’s internal VAD untouched

---

## 🗂 Project Structure

Below is the complete directory layout:
```bash
.
├── agent/
│   ├── config.py
│   ├── entrypoint.py
│   ├── session_manager.py
│   └── state.py
│
├── interrupt_handler/
│   ├── constants.py
│   ├── middleware.py
│   └── utils.py
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 What Changed

### 🔹 1. New Module: `interrupt_handler/`
| File | Description |
|------|-------------|
| `constants.py` | Lists of filler words, command words, and ASR thresholds |
| `middleware.py` | Main Core logic to classify Speech transcripts into filler/speech/command and handle interruptions |
| `utils.py` | Text normalization, word matching, helper utilities |

### 🔹 2. Updated Voice Agent
Located inside the `agent/` directory:

| File | Description |
|------|-------------|
| `entrypoint.py` | Agent setup, STT/LLM/TTS initialization, hook installation |
| `session_manager.py` | Turn management, event handling, interruption handling |
| `state.py` | Tracks agent speaking state |
| `config.py` | Reads `.env` and provides runtime config |

### 🔹 3. Updated Model Parameters

The system now uses:

- **Deepgram Nova-3** → STT  
- **OpenAI GPT-4.1-mini** → LLM  
- **Cartesia Sonic-2** → TTS  


---

## 🚀 What Features Works (✅ Verified with Manual Testing): 

- **Filler Suppression While Agent Speaks**  
  Words like *“umm”, “haan”, “uhh”, “hmm”* are ignored when the agent is speaking to avoid false interruptions.

- **Command-Based Interruption**  
  Commands such as *“stop”, “wait”, “hold on”, “pause”* immediately interrupt the agent’s speech and return control to the user.

- **Filler-as-Speech When Agent Is Silent**  
  If the agent is not speaking, fillers are treated as valid intent, ensuring the agent responds naturally.

- **Confidence-Aware Handling**  
  Low-confidence transcripts from STT are ignored, reducing false triggers caused by background noise.

- **External Middleware Architecture**  
  The semantic interruption logic is built entirely as an external layer without altering LiveKit’s VAD or internal components.

---

## 🧪 Steps to Test

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/guptayash03/agents.git
cd agents/livekit-interrupt-handler-yash
```

### 2️⃣ Install Requirements & Prepare Environment

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Create your environment file:
```bash
cp .env.example .env
```

### Fill in your API keys inside .env :

```bash
# LiveKit Cloud Credentials :

LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# Deepgram (Speech-to-Text) : 

DEEPGRAM_API_KEY=

# OpenAI (LLM for GPT-4.1-mini) : 

OPENAI_API_KEY=

# Cartesia (Text-to-Speech) :

CARTESIA_API_KEY=
```

### Start The Voice Agent:
```bash
python -m agent.entrypoint dev
```

### Test Agent in LiveKit Agent Playground:
```bash
## Open this Link
https://agents-playground.livekit.io/

## Then Login --> And Connect to the Agent
```
---

##  🛠️ Environment Details

| Component      | Version                  |
| -------------- | ------------------------ |
| **Python**     | `3.12.x` (recommended)   |
| **Livekit**    | latest stable            |

---

## 📦 Core Dependencies

| Library                  | Purpose                     |
| ------------------------ | --------------------------- |
| `livekit-agents`         | Core voice agent framework  |
| `livekit-plugins-silero` | VAD engine                  |
| `deepgram-sdk`           | Streaming STT               |
| `openai`                 | LLM (chat completions)      |
| `elevenlabs`             | TTS voice synthesis         |
| `python-dotenv`          | Environment variable loader |
| `pydantic`               | Type-safe models            |
| `aiohttp/httpx`          | Async HTTP clients          |

---

## 🔹 Known Issues : 

- **Background Noise Sensitivity**
- **Slightly Unstable Behavior During Rapid Turn Changes**
- **Micro-pauses in agent speech during filler words**