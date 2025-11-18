# 🎙️ LiveKit Voice Interruption Handling – InterruptHandler Plugin
### Robust Real-Time Filler Filtering & Natural Conversational Flow

---

## 📘 Overview

This project introduces a custom **InterruptHandler** plugin for LiveKit Agents that improves responsiveness by distinguishing between:

- 🟡 **Filler speech** — “uh”, “umm”, “hmm”, “haan” → *ignored while TTS is speaking*  
- 🔴 **Real user commands** — “stop”, “wait”, “no”, etc. → *immediately interrupt TTS*

The plugin enhances LiveKit’s default VAD **without modifying the base algorithm**, ensuring clean and natural conversation flow.

---

## 🎯 Objective

Build a modular plugin that:

- Hooks into `AgentSession` voice events  
- Filters ASR transcripts in real time  
- Interrupts speech only when needed  
- Supports configurable filler/command lists  
- Produces structured JSON logs for evaluators  
- Works under noisy/rapid real-world speech conditions  

---

## 🆕 What Changed

### 📁 New Modules Added
```
plugins/interrupt_handler/
│
├── handler.py          # Core classification + event logic
├── __init__.py         # Public exports
├── demo_interrupts.py  # Manual testing / simulation
└── README.md           # Internal technical documentation
```

### ✏️ Modified Files
None — plugin is fully self-contained.

### ⚙️ New Config Parameters
Environment-configurable:

- `INTERRUPT_HANDLER_IGNORED_WORDS`
- `INTERRUPT_HANDLER_COMMAND_WORDS`
- `INTERRUPT_HANDLER_CONFIDENCE_THRESHOLD`
- `INTERRUPT_HANDLER_UNCERTAIN_THRESHOLD`
- `INTERRUPT_HANDLER_INTERIM_COMMAND_THRESHOLD`
- `INTERRUPT_HANDLER_LOG_FILE`

---

## ⚙️ Technical Approach

### 🧠 Core Logic Summary
1. Normalize transcript → tokens  
2. Extract confidence scores  
3. Check TTS speaking state  
4. If speaking:
   - Pure filler → ignore  
   - Command phrase → interrupt  
   - High-confidence speech → interrupt  
5. If quiet:
   - Pass all speech normally  
6. Log structured decision  
7. Trigger callback when interruption is needed  

### 🔤 Default Ignored Filler Words
```
["uh", "umm", "hmm", "haan"]
```

### 🔄 Async Event Handling
- Uses `asyncio.Lock` for concurrency  
- Attaches to:
  - `agent_state_changed`
  - `user_state_changed`
  - `user_input_transcribed`
- Supports sync + async stop callbacks  

### 📊 Logging Strategy
Each decision is logged with:

- tokens  
- confidences  
- transcript  
- classification  
- TTS speaking state  
- VAD state  
- timestamp  
- metadata  

### 🛡️ Error Handling
- Safe fallback defaults  
- Prevent duplicate attachment  
- Sanitized JSON logging  
- Missing confidences → estimated fallback  

---

## ✅ What Works

- 🎯 Accurate filler detection  
- 🛑 Reliable command phrase detection (even mixed with fillers)  
- ⚡ Real-time interruption with no added lag  
- 🔁 Runtime update of ignored words (`update_ignored_words([...])`)  
- 📄 Detailed JSON logging  
- 🧪 Verified via unit + integration tests  
- 🔌 Compatible with AgentSession OR manual event feeding  

---

## ⚠️ Known Issues

- No automatic language detection  
- Noisy ASR can produce rare false positives  
- word-level confidences may not always be present  
- VAD currently logged only (not used for decisions)  

---

## 🧪 Steps to Test

### 🛠️ Environment Setup
```bash
python 3.9+
pip install -e .[dev]
```

### 🔗 Attach the InterruptHandler
```python
from plugins.interrupt_handler import InterruptHandler, InterruptHandlerConfig

handler = InterruptHandler(
    session,
    config=InterruptHandlerConfig.from_env()
)

handler.attach()
```

### 🧪 Test Options

#### ▶️ Option A — Demo Script
```bash
python -m plugins.interrupt_handler.demo_interrupts
```

#### 🤖 Option B — Automated Tests
```bash
pytest -q
```

#### 🎤 Option C — LiveKit Manual Testing
```python
handler.on_tts_state(True)
```

Now speak:
- “uh” → ignored  
- “stop” → agent stops instantly  

---

## 🖥 Environment Details

### 🐍 Python Version
- Python **3.9+**

### 📦 Dependencies
- livekit-agents  
- asyncio  
- logging  
- json  
- dataclasses  
- typing  
- inspect  
- pytest  
- pytest-asyncio  

---

## 🔍 Verified Test Cases

- Pure filler → ignored  
- Mixed filler + command → accepted  
- Quiet mode → passthrough  
- Low-confidence → uncertain  
- Full integration: “please stop” interrupts  

---

## ⭐ Bonus Features

- 🔄 Runtime update of ignored words  
- 🗂 Optional file logging  
- 🔌 Reusable stop-callback for custom runtimes  
- 🎧 Manual ASR event integration support  

---

## 📝 Additional Notes

- Supports multi-token commands: “not that”  
- Best results with ASR `words_meta`  
- VAD logged for future improvements  

---


