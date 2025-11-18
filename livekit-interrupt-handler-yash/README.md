# 🎙️ LiveKit Voice Agent — Semantic Interruption Layer  
### **NSUT Internship Assignment — Final Submission**

This repository contains a fully functional LiveKit AI voice agent extended with a **semantic interruption handling layer** that improves conversation flow by intelligently ignoring filler words, identifying real commands, and ensuring smooth human–agent interaction without modifying LiveKit’s internal VAD pipeline.

---

# 📌 Overview

This project enhances a standard LiveKit real-time voice agent by adding a **semantic interruption handling layer** that significantly improves conversational quality and user experience. The system intelligently distinguishes between *filler utterances* and *real interruption commands* while strictly maintaining LiveKit’s internal VAD pipeline without modification, as required by the NSUT Internship assignment.

### 🔍 Key Capabilities

- **Filler Suppression While Agent Speaks**  
  Words like *“umm”, “haan”, “uhh”, “hmm”* are ignored when the agent is speaking to avoid false interruptions.

- **Command-Based Interruption**  
  Commands such as *“stop”, “wait”, “hold on”, “pause”* immediately interrupt the agent’s speech and return control to the user.

- **Filler-as-Speech When Agent Is Silent**  
  If the agent is not speaking, fillers are treated as valid intent, ensuring the agent responds naturally.

- **Confidence-Aware Handling**  
  Low-confidence transcripts from STT are ignored, reducing false triggers caused by background noise.

- **Low-Latency, Real-Time Behaviour**  
  The system enforces user turn completion explicitly, ensuring fast, consistent responses from the LLM.

- **External Middleware Architecture**  
  The semantic interruption logic is built entirely as an external layer without altering LiveKit’s VAD or internal components.

This approach results in a highly stable, natural, and intuitive voice interaction system that meets all technical and behavioural specifications of the assignment.


---

## 🚀 What Changed

This submission introduces a fully modular **semantic interruption system** layered on top of a standard LiveKit voice agent. All enhancements are implemented externally without modifying LiveKit’s internal VAD pipeline, fully matching the NSUT Internship requirements.


### 🔹 1. New Module: `interrupt_handler/`
| File | Description |
|------|-------------|
| `constants.py` | Lists of filler words, command words, and thresholds |
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

Together, these provide fast, accurate, and natural real-time voice interaction.


## 🗂 Project Structure

The project is organized into two main modules:

- **`agent/`** — The primary LiveKit voice agent (STT → LLM → TTS)
- **`interrupt_handler/`** — Custom semantic interruption middleware

Below is the complete directory layout:

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


## 🔧 What Features Work

This system delivers a fully functional, real-time conversational voice agent with a semantic interruption layer designed to improve the natural flow of interaction.  
All core features required by the internship task are implemented and tested.

### ✔️ 1. Filler Suppression While Agent Speaks
Fillers such as:
- “umm”
- “haan”
- “uhh”
- “hmm”
- “accha”

are **ignored when the agent is speaking**, preventing unnecessary interruptions.

### ✔️ 2. Command-Based Interruption
Real interruption commands like:
- “stop”
- “wait”
- “hold on”
- “pause”
- “excuse me”

trigger an immediate: session.interrupt()

The agent’s speech stops instantly, handing control back to the user.

### ✔️ 3. Filler-as-Speech When Agent Is Silent
If the agent is not speaking, the same filler words are treated as **normal speech**, ensuring the LLM still responds naturally.

Example:
> User: “uhh…”  
→ Agent processes and responds.

### ✔️ 4. Confidence-Aware Transcript Handling
Low-confidence STT outputs (background noise, murmurs, distant voices) are automatically ignored to reduce false triggers.

### ✔️ 5. Reliable Turn Management
After meaningful user input, the system enforces:session.end_user_turn()

This ensures:
- Faster LLM responses  
- Fewer dropped utterances  
- More consistent interaction loops  

### ✔️ 6. Full Compatibility with LiveKit Agents v1.3.x
All event handling is updated to use: transcription_completed

instead of deprecated message events.

### ✔️ 7. VAD Pipeline Remains Untouched
The custom interruption logic is layered **externally**, ensuring:
- No modification to LiveKit’s VAD  
- Assignment compliance  
- Clean, maintainable architecture  

### ✔️ 8. End-to-End Voice Agent Pipeline
With:
- **Deepgram Nova-3** for STT  
- **OpenAI GPT-4.1-mini** for LLM  
- **Cartesia Sonic-2** for TTS  

the system provides fast, smooth, low-latency real-time voice interactions.

### ✔️ 9. Modular and Testable Codebase
All interruption logic is isolated in:
interrupt_handler/
constants.py
middleware.py
utils.py

ensuring clarity and easy future extension.

Overall, the system provides a natural, stable, and intelligent conversational experience while strictly meeting all assignment constraints.



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

Fill in your API keys inside .env before running the agent:
LiveKit Cloud Credentials :

LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

Deepgram (Speech-to-Text) : 

DEEPGRAM_API_KEY=

OpenAI (LLM for GPT-4.1-mini) : 

OPENAI_API_KEY=

Cartesia (Text-to-Speech) :

CARTESIA_API_KEY=