🧠 SalesCode.ai Final Round — LiveKit Voice Interruption Handling Challenge

Author: Ashvin Patidar (IIT Kanpur)
Branch: feature/livekit-interrupt-handler-ashvin

🔍 Overview

This project extends the LiveKit Conversational AI agent to intelligently handle real-time user interruptions in voice-based conversations.

By default, LiveKit’s TTS pauses anytime a user speaks — even for fillers like “uh”, “umm”, “haan”, “hmm”, which makes dialogue unnatural.
This solution introduces a modular extension layer called InterruptHandler that makes the AI more human-like and context-aware.

⚙️ Objectives Fulfilled
Requirement	Implementation
Ignore filler words during AI speech	✅ Configurable multilingual filler list
Detect real user commands (“stop”, “wait one second”)	✅ Immediate playback halt via session.stop_playback()
Log and categorize all events	✅ Console + in-memory logging
Maintain async safety	✅ Non-blocking, event-safe design
Integrate without modifying LiveKit SDK	✅ Plug-in extension layer
Production-grade test suite	✅ voice_agent_interrupt.py simulation script
🧩 New Module Added
📁 agents/extensions/interrupt_handler.py
🧠 Core Class: InterruptHandler
Function	Description
set_agent_state(True/False)	Tracks whether AI is currently speaking
handle_transcript(text, confidence)	Filters fillers and identifies real interruptions
add_ignored_word(word)	Dynamically extend filler list during runtime
Logging	Maintains detailed logs of ignored vs valid interruptions
Default Ignored Words (Extended Set)

Supports English + Hindi/Hinglish + casual mixed speech:

{
    # English
    "uh", "umm", "hmm", "huh", "ah", "oh", "er", "mmm", "like", "you know", "i mean",
    # Hindi/Hinglish
    "haan", "haina", "arey", "accha", "acha", "bas", "theek hai", "hmm okay",
    "toh", "matlab", "yaar", "arre", "haan na", "hmmm haan",
    # Mixed/casual
    "okay okay", "hmm okay", "haan okay", "hmm haan", "huh okay"
}

🧩 Integration with LiveKit (Main Update)

Integrated seamlessly inside
agents/examples/frontdesk/frontdesk_agent.py → entrypoint()

✅ Final Integration Snippet
# --- Initialize handler ---
interrupt_handler = InterruptHandler()

# --- Event loop integration ---
async for event in session.events():
    if event.type == "playback_started":
        interrupt_handler.set_agent_state(True)
    elif event.type == "playback_finished":
        interrupt_handler.set_agent_state(False)
    elif event.type == "transcription":
        text = getattr(event, "text", "").strip()
        confidence = getattr(event, "confidence", 0.9)

        result = await interrupt_handler.handle_transcript(text, confidence)
        if result:
            print(f"🛑 Interruption detected: '{result}'")
            await session.stop_playback()

✅ Why This Matters

Hooks directly into LiveKit’s real-time event stream.

Responds instantly to valid interruptions (e.g., “stop”, “wait”).

Ignores filler chatter during AI speech.

Works asynchronously with all base plugins (Deepgram STT, Cartesia TTS, OpenAI GPT-4o, etc.)

🧪 Local Simulation Test
📁 agents/examples/voice_agent_interrupt.py

An advanced standalone tester that simulates LiveKit’s audio stream.
It tests 25+ phrases (English + Hinglish) and provides color-coded output.

🧾 Example Output
--- Simulating: Agent Speaking ---
[STATE] Agent speaking: True
[IGNORED: filler] 'uh'
[IGNORED: filler] 'umm'
✅ Detected real user interruption: stop
✅ Detected real user interruption: wait one second
[STATE] Agent speaking: False
✅ Detected user speech: stop please
✅ Detected user speech: arre ek minute

--- Test Summary ---
Total phrases tested : 26
Ignored fillers      : 17
Detected interruptions: 9

🧠 Purpose

Unit tests the logic before full LiveKit integration.

Ensures accuracy, robustness, and multilingual resilience.

Confirms that fillers are ignored and genuine interruptions halt playback.

🧱 Changes to frontdesk_agent.py
Area	Description
entrypoint()	Added InterruptHandler and event subscription
Session loop	Tracks playback_started, playback_finished, transcription events
Behavior	Auto-stops playback on real interruption
Resilience	Handles low-confidence or empty transcripts gracefully
Code cleanup	Fixed indentation, safe default branches, and logging consistency

Note:
👉 No changes were made to LiveKit SDK internals.
All updates are additive and modular.

⚙️ Environment & Requirements
Component	Version / Tool
Python	3.10+
LiveKit Agents	Latest (from GitHub)
STT	Deepgram
TTS	Cartesia
LLM	OpenAI GPT-4o
VAD	Silero
Calendar	Cal.com / FakeCalendar
Additional	dotenv, asyncio, numpy, sounddevice, pyaudio
📦 requirements.txt
livekit-agents>=0.5.0
numpy>=1.24.0
asyncio
sounddevice
SpeechRecognition
pyaudio
python-dotenv

🧩 How to Run
🧪 Option A — Test Logic Only
python -m agents.examples.voice_agent_interrupt


Expected Behavior:

You Say	Outcome
“uh”, “umm”, “haan”	Ignored
“stop”, “wait one second”	Interruption detected
Filler when AI silent	Registered normally
🧩 Option B — Full LiveKit Integration
python agents/examples/frontdesk/frontdesk_agent.py


During AI speech:

User Says	Expected Outcome
“uh”, “umm”, “haan”	AI continues speaking
“stop”, “wait”, “no not that one”	AI stops immediately
Random mic noise	Ignored (low confidence)
🧾 Test Verification

✅ Ignored fillers verified through simulation
✅ Real interruptions trigger session.stop_playback()
✅ Logs confirm correct event classification
✅ Async stability confirmed in continuous runs

⚠️ Known Limitations
Issue	Description
Multi-language detection	Hindi + English supported; other languages can be added easily
Confidence threshold	Static (0.6); can be made adaptive
Mic noise sensitivity	May vary slightly with ASR model accuracy
🧠 Internal Flow Summary
┌──────────────┐
│  User Speaks │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ LiveKit STT Event   │
└─────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ InterruptHandler.handle_transcript │
├──────────────────────────────┤
│ - Filters fillers (uh, haan) │
│ - Checks confidence          │
│ - Detects “stop” / “wait”    │
└────────────┬─────────────────┘
             │
   ┌─────────┴──────────┐
   │  Filler → Ignore   │
   │  Command → Stop AI │
   └────────────────────┘

💾 Submission
Submission Link:
🔗 https://github.com/ashvinpatidar13/agents/tree/feature/livekit-interrupt-handler-ashvin

🏁 Summary

✅ Modular, async-safe filler interruption handler
✅ Clean LiveKit integration (no SDK modification)
✅ Tested with multilingual fillers
✅ Complete documentation and test coverage
✅ Ready for real-time voice production use

Developed by:
🎓 Ashvin Patidar , Roll No. 220243
Department of Civil Engineering, IIT Kanpur
For SalesCode.ai Final Round Qualifier