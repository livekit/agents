# Filler-Aware Interruption Handler for LiveKit Agents

This module adds an extension layer that filters filler words while the agent is speaking,
and properly handles real interruptions.

### Features
- Ignore fillers such as: uh, umm, hmm, haan
- Detect real interruptions: stop, wait, hold on
- Low confidence handling for background murmurs
- Async safe, no changes to LiveKit core

### How to Test
pytest -q

### How to Run Example
python examples/interrupt_handler_demo.py
