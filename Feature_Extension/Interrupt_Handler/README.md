
🚀 Overview

This module extends LiveKit’s Voice Agent pipeline with a smart interruption layer that filters out meaningless filler sounds while ensuring instant response to real user speech.

It does not modify LiveKit’s internal VAD; it works purely through public events (agent_speech_started, agent_speech_ended, transcription) and calls session.interrupt() when needed.

📂 Directory Summary
Feature_Extension/
└─ Interrupt_Handler/
   ├─ src/
   │  ├─ config.py          # All configurable parameters (multi-lang + runtime updates)
   │  ├─ classifier.py      # Text classification logic (decides interrupt/ignore)
   │  ├─ controller.py      # Orchestrator binding LiveKit events
   │  ├─ state.py           # Tracks agent speaking state
   │  ├─ logkit.py          # Logging utilities
   │  └─ __init__.py
   │
   ├─ examples/
   │  ├─ minimal_agent.py   # Local demo (no credentials)
   │  └─ run_worker.py      # Full LiveKit integration
   │
   ├─ tests/
   │  ├─ test_classifier.py
   │  └─ test_controller.py
   │
   ├─ .env.example          # Sample environment configuration
   └─ README.md             # Documentation (this file)

🧩 What Changed

New module: Interrupt_Handler/

Introduced a clean, modular interruption layer under Feature_Extension/Interrupt_Handler/src/.

Added configuration and runtime customization support.

Implemented multilingual (English + Hindi/Hinglish) filler and command detection.

Added micro-debounce buffering for rapid partials.

Provided a full example agent and pytest suite.

File	Purpose
config.py	Manages filler words, hard command phrases, thresholds, runtime updates, and language packs.
classifier.py	Classifies user utterances (LOW_CONF, FILLER_ONLY, HARD_INTENT, CONTENT).
controller.py	Core orchestration — hooks LiveKit events, ignores filler while speaking, calls session.interrupt() on real speech.
state.py	Tracks if agent TTS is currently speaking (thread-safe).
logkit.py	Unified logging with env-based level.
examples/*	Demonstration scripts (local + full integration).
tests/*	Unit + integration tests verifying correctness.
✅ What Works
Feature	Status
Ignores filler words while agent is speaking	✅
Registers same words when agent is silent	✅
Stops immediately on real user input	✅
Real-time async / non-blocking	✅
Configurable lists & thresholds	✅
Dynamic runtime updates	✅ (Bonus #1)
Multi-language filler/command support	✅ (Bonus #2)
Clean logs, modular structure, pytest validation	✅
⚙️ Bonus Features Implemented
1️⃣ Dynamic Runtime Updates

Lists of filler and hard-command phrases can be changed without restarting the agent.

from Feature_Extension.Interrupt_Handler.src.config import IHConfig
cfg = IHConfig.from_env()
cfg.add_fillers(["acha", "arey"], lang="hi")
cfg.add_commands(["ek second"], lang="hi")


Useful for per-customer or per-language fine-tuning in live systems.

2️⃣ Multi-Language / Code-Mixed Detection

Built-in packs: English + Hindi/Hinglish
Configurable via .env →

IH_LANGS=en,hi


Examples:

Utterance	Agent Speaking?	Behavior
“uh umm hmm”	✅	Ignored
“umm okay stop”	✅	Interrupt
“thoda ruk please”	✅	Interrupt
“haan umm acha”	✅	Ignored
same while agent quiet	❌	Registered as user input

This satisfies the bonus for multi-language filler detection.

🧠 Known Issues / Edge Cases

Confidence values depend on STT model quality; noisy microphones may cause misclassification at very low ASR confidence.

In extremely rapid user-agent turn-taking (<150 ms), a filler-to-intent transition may trigger a delayed interrupt; tweak IH_DEBOUNCE_MS.

Currently supports en and hi; other languages can be added by editing default_lang_packs() in config.py.

🧪 Steps to Test
🧩 1. Local Simulation (no credentials)
cd Feature_Extension/Interrupt_Handler
python -m examples.minimal_agent


Output will log:

[IGNORED filler] -> agent continues
[INTERRUPT hard_intent] -> agent stops

🧩 2. Run Automated Tests
pytest -q

🧩 3. Full LiveKit Run (with keys)

Copy .env.example → .env

Fill your LIVEKIT_* and OPENAI_API_KEY

Run:

python -m examples.run_worker


Speak during agent TTS:

“umm hmm” → ignored

“stop” or “wait one second” → interrupt

🔧 Environment Details
Key	Description
Python	≥ 3.10
Dependencies	livekit-agents, livekit-plugins-openai, python-dotenv, pytest
Config	.env (see .env.example)
.env.example
IH_LANGS=en,hi
IH_FILLERS=uh,umm,um,hmm,haan,huh,er,eh,mmm,arey,acha
IH_HARD_PHRASES=stop,wait,pause,hold on,cancel,no,ruk,ruko,thoda ruk,ek second
IH_MIN_CONFIDENCE=0.5
IH_MIN_CONTENT_TOKENS=2
IH_MIN_DURATION_MS=250
IH_DEBOUNCE_MS=200
IH_LOG_LEVEL=INFO

🧾 What Changed (Summary Table)
Area	Before	After
Interruption logic	N/A	Added semantic filter layer
Configurability	Static	Runtime-updatable via class methods
Language support	English only	English + Hindi/Hinglish
Testing	None	Full pytest suite
Integration	Manual stop	Event-driven async orchestrator
📊 What Works (Validated)
Test	Result
Filler-only ignored while speaking	✅
Filler registered when quiet	✅
Hard phrase triggers interrupt	✅
Low confidence ignored	✅
Multi-lingual mixed input	✅
Runtime add/remove works	✅
⚙️ Environment & Dependencies

Python: 3.10 or newer

OS: Linux / macOS / Windows

Libraries:

livekit-agents

livekit-plugins-openai

python-dotenv

pytest

Install with:

pip install -r requirements.txt

🔬 Steps to Verify (Manual QA)
Scenario	Expected
Agent speaking + “uh”	continues
Agent speaking + “umm okay stop”	stops immediately
Agent quiet + “umm”	registers speech
Low confidence murmur	ignored
⚙️ Implementation Notes

All logic runs outside core VAD (extension layer).

Thread-safe via SpeechGate lock.

200 ms micro-debounce merges fast partials.

Uses configurable .env params.

Fully async and non-blocking.
