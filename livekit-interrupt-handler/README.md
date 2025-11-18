# LiveKit Interrupt Handler — Custom Extension

This module implements a lightweight interrupt-handling system for LiveKit-style conversational agents.  
It was developed as part of an interview task and is fully self-contained under:

agents/livekit-interrupt-handler/

yaml
Copy code

The project includes:
- A production-ready `InterruptHandler` class
- Unit tests
- A simulator
- Optional audio-hook for future STT integration

---

## 1. What Changed (Overview)

The following components were added to the main `agents` repository:

### **New Folder**
livekit-interrupt-handler/

markdown
Copy code

### **New Files**
|          File            | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `interrupt_handler.py`   | Core handler logic (filler filtering, stop commands, agent-speech state        machine).                                                                                           |
| `simulator.py`           | Script to simulate transcription events without real audio.            |
| `tests/test_handler.py`  | Unit tests validating core logic.                                      |
| `examples/demo_script.sh`| Quick-start example runner.                                         
| `README.md`              | Documentation for this component.|

### **New Behavior / Logic**
- Ignores filler words *(uh, umm, hmm)* **only when the agent is speaking**.
- Detects stop/interruption commands such as:
  - **stop**, **wait**, **pause**, **hold on**, **never mind**
- Treats all speech as **interrupts** when the agent is speaking.
- Treats all speech as **user input** when the agent is silent.
- Filters out low-confidence transcription events based on a configurable threshold.
- Optional method `on_audio_event()` prepared for future audio-to-text integration.

---

## 2. What Works (Verified Features)

The following were tested manually and via automated unit tests:

### ✔ Filler Detection  
- When agent is speaking, `"uh"` / `"umm"` / `"hmm"` are **ignored**  
- Callback `on_ignored_filler()` fires correctly

### ✔ Stop Command Detection  
- `"wait"`, `"stop"`, `"pause"`, `"hold on"`, etc. trigger `on_valid_interrupt()`

### ✔ Agent-Speaking State Machine  
- When the agent is speaking, **any** user speech becomes an interrupt  
- When the agent is silent, text becomes normal user input

### ✔ Confidence Threshold  
- Low-confidence inputs (`< threshold`) are ignored

### ✔ Full Unit Test Coverage  
All three provided tests pass:

3 passed in 0.20s

yaml
Copy code

### ✔ Simulator Working
Running `simulator.py` shows correct behavior for test strings.

---

## 3. Known Issues / Limitations

These are **expected and acceptable** given the assignment scope:

### ❗ No Real Audio Processing  
`on_audio_event()` is a placeholder only.  
Real speech-to-text (STT) is **NOT** implemented in this task.

### ❗ No Integration with LiveKit Audio Pipeline  
This module does **not** connect to:
- RTC audio streams  
- TTS cancellation  
- Whisper/Vosk engines  

### ❗ No Partial Transcription Handling  
Only full transcription strings are handled.

### ❗ Filler Detection Is Exact-Match Only  
Filler words must match exactly (e.g., `"ummm..."` will not match).

---

## 4. Steps to Test (How to Run Everything)

### **A. Unit Tests**
From inside `livekit-interrupt-handler/`:

```bash
pytest -q tests
Expected:

Copy code
3 passed
B. Simulator (Text-Based)
Run the simulator:

bash
Copy code
python simulator.py
Try these:

Input	Agent Speaking	Expected
uh	True	ignored filler
umm	False	normal user speech
wait	True/False	interrupt
hello	False	user speech
hello	True	interrupt

C. Optional Audio Pipeline (Future Integration)
The method exists:

python
Copy code
await handler.on_audio_event(audio_bytes, simulate_text="stop")
But real audio decoding must be added later using:

Vosk

Whisper

Google Speech-to-Text

Azure Speech

(Not required for this assignment.)

5. Environment Details
Python Version
Python 3.10+ recommended

Tested on Python 3.11

Dependencies
The module uses:

nginx
Copy code
asyncio
pytest
python-dotenv
Optional (for future real audio):

nginx
Copy code
SpeechRecognition
PyAudio
Vosk / Whisper
Installation
Inside the livekit-interrupt-handler/ folder:

bash
Copy code
pip install -r requirements.txt
Branch Information
Work completed under feature branch:


bash
Copy code
feature/livekit-interrupt-handler-ROHIT-25607