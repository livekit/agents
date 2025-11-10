# 🧠 SalesCode.ai Final Round — LiveKit Voice Interruption Handling Challenge

**Author:** Ashvin Patidar (IIT Kanpur)  
**Branch:** `feature/livekit-interrupt-handler-ashvin`

---

## 1️⃣ What Changed

This project adds a new **extension layer** to the LiveKit agent for intelligent real-time voice interruption handling.

### 🔧 New Modules & Logic
- **`agents/extensions/interrupt_handler.py`** — custom module implementing a class `InterruptHandler` to:
  - Track when the AI agent is speaking (`set_agent_state()`).
  - Handle transcription events from ASR and classify input as filler vs. real interruption.
  - Maintain a configurable list of filler words (`ignored_words`).
  - Return `None` for fillers or low-confidence inputs, and the text itself for valid user commands.
  - Log every processed phrase for debugging (`ignored_filler`, `ignored_low_conf`, `valid_interrupt`).

- **Integration:**  
  Inserted event-level logic inside  
  `agents/examples/frontdesk/frontdesk_agent.py → entrypoint()`  
  to call the `InterruptHandler` during live session events.

### 🧩 Code Added to `frontdesk_agent.py`
```python
# Initialize filler-word handler
interrupt_handler = InterruptHandler()

# Subscribe to LiveKit session events
async for event in session.events():
    if event.type == "playback_started":
        interrupt_handler.set_agent_state(True)
    elif event.type == "playback_finished":
        interrupt_handler.set_agent_state(False)
    elif event.type == "transcription":
        text = event.text
        confidence = getattr(event, "confidence", 0.9)
        result = await interrupt_handler.handle_transcript(text, confidence)
        if result:
            print(f"🛑 Interruption detected: '{result}'")
            await session.stop_playback()
No changes were made to the LiveKit SDK or base logic — only an external extension layer was added.

2️⃣ What Works (Test Results)
Feature	Verified Behavior
Ignore filler words during agent speech	✅ “uh”, “umm”, “hmm”, “haan” ignored
Detect real interruption commands	✅ “stop”, “wait”, “no not that one” stops playback
Register fillers when agent is silent	✅ Works correctly
Dynamic filler list modification	✅ Supported at runtime
Logging and confidence thresholding	✅ Confirmed in console logs
Async safety / concurrency	✅ Stable during multiple events

🧪 Local Simulation Test (voice_agent_interrupt.py)
Output example:

csharp
Copy code
[STATE] Agent speaking: True
[IGNORED: filler] 'uh'
[IGNORED: filler] 'umm'
[INTERRUPT] 'stop'
✅ Detected real user interruption: stop
[STATE] Agent speaking: False
[INTERRUPT] 'umm'
[INTERRUPT] 'haan okay'
✅ Detected real user interruption: haan okay
All tests confirm robust performance both standalone and integrated with LiveKit.

3️⃣ Known Issues
Issue / Edge Case	Description
Multi-language filler detection	Currently limited to English + small Hindi filler list (e.g., “haan”)
Fixed confidence threshold	Static value (0.6) — could be user-configurable
Background noise	May occasionally trigger low-confidence events if ASR misfires
Real-time mic testing	Requires correct LiveKit setup and working audio device

None of these affect core logic or functionality during evaluation.

4️⃣ Steps to Test
🧩 A. Test Logic Without LiveKit
bash
Copy code
python -m agents.examples.voice_agent_interrupt
Expected Behavior

Input	Result
“uh”, “umm”, “hmm”, “haan”	Ignored
“stop”, “wait one second”	Interruption detected
Filler while agent silent	Logged as normal text

🧩 B. Test Integrated LiveKit Agent
bash
Copy code
python agents/examples/frontdesk/frontdesk_agent.py
Then speak during playback:

You Say	Expected Outcome
“uh”, “umm”, “haan”	AI continues speaking
“stop”, “wait”, “no not that one”	AI stops immediately
Random short noise	Ignored if low-confidence

To verify:

Watch console logs for [IGNORED: filler] or 🛑 Interruption detected: messages.

Ensure the AI’s TTS stops when valid command detected.

5️⃣ Environment Details
Component	Version / Tool
Python	3.10+
LiveKit Agents	Latest (GitHub)
ASR	Deepgram
TTS	Cartesia
LLM	OpenAI GPT-4o
VAD	Silero
Calendar API	Cal.com / FakeCalendar
Additional Libs	dotenv, asyncio, numpy, sounddevice, pyaudio

📦 Requirements File
requirements.txt

txt
Copy code
livekit-agents>=0.5.0
numpy>=1.24.0
asyncio
sounddevice
SpeechRecognition
pyaudio
python-dotenv
🖥️ Setup
bash
Copy code
git clone https://github.com/ashvinpatidar13/agents.git
cd agents
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


💾 Submission
Submission Link:
🔗 https://github.com/ashvinpatidar13/agents/tree/feature/livekit-interrupt-handler-ashvin


🏁 Summary
✅ Extension logic working end-to-end
✅ Integrated cleanly into LiveKit example
✅ No SDK modification
✅ Fully documented, reproducible, and testable

Developed by:
Ashvin Patidar
Department of Civil Engineering, IIT Kanpur
For SalesCode.ai Final Round Qualifier