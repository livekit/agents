# Filler-Aware Voice Agent Extensions

This branch introduces a robust, modular extension designed to eliminate **false interruptions** in LiveKit Voice Agents caused by filler words such as **“uh”, “umm”, “hmm”, “haan”, “okay”**, and other low-value utterances during agent speech.

The solution adds a **filler-aware STT wrapper**, **dynamic runtime filler loading**, and **English + Hindi + Hinglish normalization**, all while keeping LiveKit’s internal VAD logic untouched.  
The result is uninterrupted, smooth, and natural conversational flow.

---

# 📋 Table of Contents

### 🚀 Getting Started
- [`basic_agent.py`](./basic_agent.py) - Voice agent using filler-aware STT filtering and improved interruption control.

### 🎙️ Filler & Interruption Control
- [`extensions/filler_aware_adapter.py`](./extensions/filler_aware_adapter.py)  
  STT wrapper that suppresses filler-only speech while the agent is speaking.
  
- [`extensions/filler_manager.py`](./extensions/filler_manager.py)  
  Dynamic filler + command word loader with hot reload.
  
- [`extensions/config.py`](./extensions/config.py)  
  Shared configuration for environment-based and file-based filler lists.

- [`extensions/filler_words/`](./extensions/filler_words/)  
  Organized `.txt` files for English, Hindi, Hinglish fillers and command phrases.

### 🎯 Core Behaviors Demonstrated
- Suppresses filler words while agent is speaking  
- Real commands always interrupt  
- Multilingual normalization (EN + HI + Hinglish)  
- Short, low-confidence STT chunks ignored  
- Dynamic runtime filler updates  
- No changes to LiveKit's internal VAD  

### 📡 Real-Time Interaction Improvements
- Significant reduction in false agent pauses  
- Smooth long speech segments without disruption  
- Dynamic filler lists, editable without restart  
- Fully compatible with LiveKit’s audio pipeline  

---

# ✅ What Changed

This branch adds the following new components:

### **✅ 1. Filler-Aware STT Wrapper**
`extensions/filler_aware_adapter.py`
- Wraps existing STT (`assemblyai/universal-streaming`)
- Filters fillers only when the agent is speaking
- Allows commands to interrupt
- Marks short, low-confidence segments as ignorable

### **✅ 2. Multilingual Filler Manager**
`extensions/filler_manager.py`
- Loads filler words from `extensions/filler_words/*.txt`
- Supports English, Hindi, Hinglish, mixed variants
- Hot reloads anytime files change
- Separates fillers vs. commands logically

### **✅ 3. Enhanced Configuration Layer**
`extensions/config.py`
- Loads env-provided fillers (`IGNORED_FILLERS`)
- Merges file-based fillers from folder
- Exposes global thresholds for confidence & token limits

### **✅ 4. Updated Example Agent**
`basic_agent.py` updated to:
- Wrap STT with `FillerAwareAdapter`
- Add `is_agent_speaking()` detection via `_audio_out.is_speaking`
- Tune preemptive generation  
- Tune turn detection (`unlikely_threshold=0.80`)  
- Use Silero VAD for stability  
- Log metrics for evaluation  

---

# ✅ What Works (Verified)

The following behaviors are confirmed through real testing:

### ✅ **1. Filler suppression works**
While the agent is speaking, saying:

uh
umm
hmm
haan
accha
hmm okay
okk
ummm


➡ **Agent continues speaking immediately**  
➡ **No pause, no stutter, no interruption**  

---

### ✅ **2. Real user interruptions still work**

Valid interrupt commands:

stop
pause
wait
hold on
ruk jao
bas
thamba


➡ Agent **immediately stops speaking**

---

### ✅ **3. Multilingual + Hinglish support**

Examples normalized:

| Spoken | Normalized |
|--------|------------|
| haan ji | haan |
| acha | accha |
| okayyy | okay |
| hmm ok | hmm okay |
| uhhh | uh |
| han | haan |

---

### ✅ **4. Dynamic hot reload works**

Editing any `.txt` file in:

extensions/filler_words/

automatically updates filler lists during runtime:

[Dynamic Reload] Updated fillers=34 commands=6


---

### ✅ **5. Zero changes to LiveKit internals**

✅ Does **not** patch VAD internals  
✅ Does **not** override SDK methods  
✅ Uses only safe public API (`STT`, `RecognizeStream`, `_audio_out.is_speaking`)  

---

# ⚠ Known Issues / Limitations

- STT may occasionally mis-detect fillers as meaningful words → interruption occurs.  
- Extremely noisy environments may still trigger VAD.  
- Overlapping real user speech + filler may lead to partial interruption.  
- If LiveKit changes `_audio_out.is_speaking` internals, detection logic may require a small update.

---

# ▶️ Steps To Test

### ✅ 1. Install dependencies
```bash
pip install -r requirements.txt

✅ 2. Start agent in console mode
```bash
python basic_agent.py console

✅ 3. Speak while agent is talking

Say:
uh
umm
haan
okay
accha
hmm

Expected:
✅ No interruption
✅ No pause
✅ Agent continues smoothly

✅ 4. Test interruption commands

Say:
stop
wait
pause
ruk jao


Expected:
✅ Agent stops instantly

✅ 5. Test dynamic reload

1. Open extensions/filler_words/fillers_hindi.txt

2. Add:
arey

3. Save the file

4. Console should show:

[Dynamic Reload] Updated fillers=...

Now the agent treats “arey” as filler.

# 🛠 Environment Details
✅ Tested On

Python 3.10 - 3.12

LiveKit Agents v1.2.18

AssemblyAI Universal Streaming STT

OpenAI GPT-4.1-mini LLM

Cartesia Sonic-2 TTS

Silero VAD

Multilingual Turn Detector

✅ Optional Environment Variables
export IGNORED_FILLERS='["uh","umm","hmm"]'
export INTERRUPT_COMMANDS='["stop","wait","pause"]'
export MIN_CONFIDENCE_AGENT_SPEAKING="0.35"
export SHORT_SEGMENT_TOKENS="5"


# 📁 Project Structure

examples/voice_agents/
│── basic_agent.py
│── extensions/
│     ├── filler_aware_adapter.py
│     ├── filler_manager.py
│     ├── config.py
│     └── filler_words/
│           ├── english.txt
│           ├── hindi.txt
│           └── custom.txt

