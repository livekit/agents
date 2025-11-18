# Voice Interrupt Handler (extension)

Lightweight extension that attaches to a running `AgentSession` at runtime (no LiveKit SDK changes). It filters ASR transcripts to reduce false interruptions and allow intentional interrupts.

## What changed
Added a self-contained extension `voice_interrupt` under `extensions/voice_interrupt_handler/` that:

- Ignores filler-only user utterances (configurable list) **while the agent is speaking**.
- Treats filler words as valid user input when the agent is idle.
- Immediately interrupts the agent on meaningful commands (e.g. `stop`, `wait`).
- Attaches at runtime via `session.on(...)` and calls `session.interrupt()`; **no SDK code modified**.

## Files added (paths relative to repo root)
- `extensions/voice_interrupt_handler/voice_interrupt/__init__.py`  
- `extensions/voice_interrupt_handler/voice_interrupt/handler.py` — core `InterruptHandler` class (event subscriptions, state tracking, transcript filtering, interrupt calls).  
- `extensions/voice_interrupt_handler/voice_interrupt/plugin.py` — `attach_interrupt_handler(session, ...)` convenience wrapper.  
- `extensions/voice_interrupt_handler/voice_interrupt/tests/test_handler.py` — pytest unit tests covering primary behaviors.  
- `extensions/voice_interrupt_handler/examples/run_with_extension.py` — mock-first demo shipped with the repo (uses deterministic mock if real LiveKit not available).  
- `extensions/voice_interrupt_handler/examples/console_worker_no_plugins.py` — CLI worker example that uses console IO and attaches the extension (works without importing plugin modules).  
- `extensions/voice_interrupt_handler/examples/console_agent_minimal.py` — minimal attempt to call `session.console()` (kept for reference; may not exist on all SDK versions).  
- `extensions/voice_interrupt_handler/examples/console_tts_with_keyboard_stt.py` — offline TTS + keyboard-as-STT demo (guaranteed audible demo on Windows).  
- `extensions/voice_interrupt_handler/examples/console_tts_with_keyboard_stt_debug_force_stop.py` — debug helper to force-stop TTS playback during recording.  
- `extensions/voice_interrupt_handler/README.md` — this file.

## Quick start — Windows PowerShell (mock demo)
1. Activate your venv (adjust path if different):
```powershell
cd C:\path\to\livekit-agent
.\.venv\Scripts\Activate.ps1

2. Make the extension importable in this shell:
```powershell
$env:PYTHONPATH = "$PWD;$PWD\extensions\voice_interrupt_handler"

3. Run the mock demo (no LiveKit/cloud keys required):
```powershell
python .\extensions\voice_interrupt_handler\examples\console_tts_with_keyboard_stt.py