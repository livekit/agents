# LiveKit Interrupt Handler (feature/livekit-interrupt-handler-<yourname>)

This branch adds an "Interrupt Handler" extension that filters filler words during TTS
so the LiveKit agent continues speaking on short fillers (e.g., "uh", "umm") but
still responds to real interruptions (e.g., "stop", "wait").

## What changed
- `livekit_interrupt_handler.py` — new module providing `InterruptHandler`.
- `main.py` — example integration CLI (uses a small `AgentStub` for testing).
- `tests/test_interrupt_handler.py` — pytest-asyncio tests.
- `.env.example`, `requirements.txt`.

## Features
- Ignores configured filler words while agent is speaking.
- Treats same fillers as valid speech when the agent is not speaking.
- Confidence-based filtering for low-confidence background murmur.
- Configurable `IGNORED_WORDS` env var or runtime `handler.set_ignored_words([...])`.
- Logs ignored vs valid interruptions for debug and metrics.
- Async-safe via `asyncio.Lock()`.

## How to run (local testing)
1. Create branch in your fork (see git commands below).
2. Create a Python virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
