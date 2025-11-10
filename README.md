# 🎯 LiveKit Voice Interruption Handling — SalesCode.ai Challenge

## What Changed
- Added `agents/extensions/interrupt_handler.py` — introduces async `InterruptFilter` to handle filler-word interruptions.
- Updated `agent_activity.py` to integrate the filter safely (no changes to LiveKit core logic).
- Added unit tests: `tests/test_interrupt_filter.py`.

## What Works
- Ignores filler-only inputs (e.g., “uh”, “umm”, “hmm”) during TTS.
- Instantly stops on real user speech (e.g., “stop”, “wait”).
- Async-safe and integrates cleanly into LiveKit’s voice event loop.
- Verified via `pytest` — key tests pass ✅.

## Known Issues
- Filler list currently static (not dynamically updatable).
- Multi-language filler support not yet added.
- Some unrelated LiveKit plugin tests may fail (missing API keys).

## Steps to Test
```powershell
# Ensure you're inside the SalesCode_AI project directory

# Checkout the feature branch
git checkout feature/salescodeai-karthik

# Setup virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install minimal dependencies
pip install -r requirements_min.txt

# Run key test files
pytest -q tests/test_interrupt_filter.py     # expect 4 passed
pytest -q tests/test_transcription_filter.py # expect 11 passed
