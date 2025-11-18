# LiveKit Interrupt Handler

This mini project implements an interrupt handler that:

- Ignores filler words (uh, umm, hmm) when the agent is speaking.
- Accepts those same words normally when the agent is quiet.
- Detects stop words like "stop", "wait", "pause".
- Works as an extension layer (no LiveKit core modification).
- Includes a simulator and tests.

How to run the simulator:
python simulator.py

How to run tests:
pytest -q
