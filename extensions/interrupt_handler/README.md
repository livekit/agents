# LiveKit Voice Interruption Handling - Extension Layer

This repository contains an extension that filters filler sounds during agent speech,
while still allowing real interruptions. It was implemented for the SalesCode.ai
LiveKit Voice Interruption Handling challenge. (See original problem statement file.)

## What changed / What this is

- A standalone `InterruptHandler` class that sits on top of transcription events
  and agent speaking state.
- Optional `aiohttp` endpoint to update ignored words at runtime.
- Example script `examples/run_local_simulation.py` to locally validate behavior.
- Unit tests under `tests/`.

## Features implemented

- Configurable ignored words (via env or runtime REST).
- Always-recognize interrupt commands (stop/wait/pause/no).
- Confidence threshold to ignore low-confidence transcriptions while agent speaks.
- Async / thread-safe state handling.
- Logging: ignored events and valid interruptions logged separately.

## How to run locally (simulation)

1. Create virtualenv and install:

````python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Run simulation:
python -m examples.run_local_simulation

3. Run tests:
```pytest-q


## Integration with LiveKit Agents (high-level)
See `examples/integration_example.md` for the exact hook points:
- Call `handler.set_agent_speaking(True|False)` when your agent starts/stops TTS.
- Call `handler.on_transcription(asr)` whenever LiveKit delivers a transcription.
- If `on_transcription` returns `interrupt`, call your agent stop/pause API immediately.

## Environment / Versions
- Python 3.9+
- aiohttp, pytest
- See `pyproject.toml` for minimal metadata.

## Known issues / extension ideas
- ASR vocab mismatch: filler tokens must appear in ASR text. For languages other than English,
add corresponding filler tokens to the list or use language-specific lists.
- Multi-language detection: can be improved using fast heuristics or NLU classification.
- If ASR doesn't provide confidence, the module assumes confidence=1.0 — you may want to
modify behavior if your ASR lacks reliable confidence scores.

## Files referenced
The original challenge spec (SalesCode.ai LiveKit Voice Interruption Handling) is included
in the code submission and used as the design guide.

````
