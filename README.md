# Chat Processor AI — Intelligent Voice Assistant with Advanced Interruption Handling
### *NSUT Assignment – B.Tech CSE (Artificial Intelligence), 3rd Year*

This repository contains my implementation of a **multilingual, filler-aware, interruption-safe voice assistant** built using the **LiveKit Agents** framework.  
The goal of this work was to create a more natural and intelligent voice interaction system that understands the difference between *filler noise* and *real user interruptions* — something real conversational AI systems must handle correctly.

I reorganized and renamed the modules into a clean package called **`chat_processor`** to separate custom logic from the default example structure.

---

## 🚀 Features & Capabilities

### ✔ Filler Words Are Ignored While Agent Speaks
The agent does **not** interrupt itself when the user says:
- `uh`
- `umm`
- `hmm`
- `haan`
- or any other filler word you add dynamically

### ✔ Filler Words Count as Valid Input When Agent Is Silent
If the agent is not speaking, even “umm” becomes `"user_speech"`.

### ✔ Real User Interruptions Instantly Stop the Agent
Works for both languages:

**English:**  
`stop`, `wait`, `hold on`, `no`  

**Hindi (Latin script):**  
`ruko`, `band`, `nahi`

### ✔ English + Hindi Language Detection  
The model automatically switches word-lists based on STT language tags.

### ✔ Live Dynamic Updates  
Modify `agent_profile.json` and the agent updates **without restarting**.

### ✔ Clean Comparison Against Baseline  
Two agents included:
- `main_agent.py` → full intelligent behavior  
- `default_agent.py` → plain LiveKit behavior (interrupts on everything)

---

## 📁 Project Structure

chat_processor/
agent_config.py
intent_detector.py
memory_manager.py
settings_loader.py

main_agent.py
default_agent.py
agent_profile.json


### **Module Responsibilities**

#### `agent_config.py`
- Loads all environment variables
- Builds a central `Settings` object
- Handles filler words, command words, default language, dynamic config path

#### `intent_detector.py`
Implements the core classification logic:
- `"ignore_filler"`
- `"interrupt_agent"`
- `"user_speech"`

#### `memory_manager.py`
Tracks agent & user speaking states:
- `is_agent_speaking()`
- `is_user_speaking()`

#### `settings_loader.py`
- Loads filler & command words
- Watches JSON file for live updates
- Handles multilingual behavior

#### `main_agent.py`
- Integrates STT → VAD → LLM → TTS
- Uses custom filler-aware interruption logic
- Applies multilingual rules
- Uses `min_interruption_words=2` to avoid false interrupts

#### `default_agent.py`
- No custom logic
- Uses LiveKit’s default interruption behavior
- Useful for comparison

---

## 🧠 Intelligent Classification Logic (Summary)

1. **If the agent is not speaking**  
   → **Everything counts as user speech**

2. **If the agent is speaking:**
   - Contains interrupt word → `"interrupt_agent"`
   - All tokens are fillers → `"ignore_filler"`
   - Very short confirmations → `"ignore_filler"`
   - Anything else → `"interrupt_agent"`

This improves conversation flow and reduces accidental cutoffs.

---

## ⚙ Installation & Setup

### 1. Clone the repository

git clone <repo>
cd <project-root>


2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows


3. Install dependencies
pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.3"
pip install python-dotenv

4. Create a .env file
LIVEKIT_URL=your_url
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

OPENAI_API_KEY=sk-xxxx
LIVEKIT_INFERENCE_USE_DIRECT_OPENAI=1

DEFAULT_LANGUAGE=en

FILLER_CONFIDENCE_THRESHOLD=0.6
FILLER_CONFIG_PATH=./agent_profile.json


5. Create the dynamic config file
agent_profile.json


▶ Running the Agents
python main_agent.py console

Run the baseline agent
python default_agent.py console


🧪 Verified Behaviour During Testing

✔ Correctly ignores filler words during TTS

✔ Correctly treats fillers as speech when agent is silent

✔ Interrupts instantly on real interruption commands

✔ Selects correct language profile based on STT

✔ Reacts instantly to JSON updates

✔ Baseline agent interrupts on all noise (intended behavior)




