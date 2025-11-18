# 🎙️ LiveKit Voice Agent — Interruption Handler - YASH GUPTA
### **NSUT Internship Assignment — Final Submission**

# 📌 Overview

This project enhances a standard LiveKit real-time voice agent by adding a **interruption handling layer** that significantly improves conversational quality and user experience. The system intelligently distinguishes between *filler utterances* and *real interruption commands* while strictly maintaining LiveKit’s internal VAD pipeline without modification, as required by the NSUT Internship assignment.

### 🔍 What Features Works : 

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

## 🚀 What Changed

### 🔹 1. New Module: `interrupt_handler/`
| File | Description |
|------|-------------|
| `constants.py` | Lists of filler words, command words, and ASR thresholds |
| `middleware.py` | Core logic to classify transcripts into filler/speech/command |
| `utils.py` | Text normalization, word matching, helper utilities |

### 🔹 2. Updated Voice Agent
Located inside the `agent/` directory:

| File | Description |
|------|-------------|
| `entrypoint.py` | Agent setup, STT/LLM/TTS initialization, hook installation |
| `session_manager.py` | Turn management, event handling, interruption handling |
| `state.py` | Tracks agent speaking state |
| `config.py` | Reads `.env` and provides runtime config |

### 🔹 3. Updated Model Stack

The system now uses:

- **Deepgram Nova-3** → STT  
- **OpenAI GPT-4.1-mini** → LLM  
- **Cartesia Sonic-2** → TTS  


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
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

## 🧪 Steps to Test

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/livekit-interrupt-handler
cd livekit-interrupt-handler
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