
### Directory Summary How file is orgenised 
salescode_interrupt_handler/
│
├── config.py                  # All configurable parameters
├── interrupt_handler.py       # Core logic
├── logging_utils.py           # Logging helpers
├── example_agent.py           # Working integration
├── __init__.py
│
├── main.py                    # Worker bootstrap
│
├── tests/
│   └── test_interrupt_handler.py
│
└── README_SalesCode_Interrupts.md  # Documentation

###    first paste your own key in .env file 

### Run

   ```bash
   python main.py console 


# Filler-Aware Interruption Handling for LiveKit Voice Agents

This implementation is an extension layer on top of `AgentSession` for the
SalesCode.ai LiveKit Voice Interruption Handling Challenge.

It **does not modify** LiveKit's core VAD. All behavior uses public events
(`user_input_transcribed`, `agent_state_changed`) and `session.interrupt()`.

## What Changed

- New package: `salescode_interrupt_handler/`
  - `config.py` — configurable:
    - filler words (e.g., `uh`, `umm`, `hmm`, `haan`)
    - hard interruption phrases (e.g., `stop`, `wait`, `hold on`)
    - thresholds for confidence & token counts
  - `logging_utils.py` — unified logging.
  - `interrupt_handler.py` — `FillerAwareInterruptController`:
    - tracks `agent_speaking` via `agent_state_changed`
    - listens to `user_input_transcribed`
    - while agent is speaking:
      - ignores pure fillers & low-confidence murmur
      - triggers `session.interrupt()` only on:
        - configured command phrases, or
        - sufficiently meaningful non-filler speech.
    - while agent is not speaking:
      - does not interfere; normal LiveKit behavior continues.
  - `example_agent.py` — demo:
    - builds `AgentSession` with `allow_interruptions=False`
    - attaches `FillerAwareInterruptController`
    - runs a simple sales assistant agent.

- `main.py`
  - Boots a LiveKit Worker using `WorkerOptions(entrypoint_fnc=entrypoint)`.

- `tests/test_interrupt_handler.py`
  - Verifies core behavior (fillers ignored, real interruptions respected).

## How It Works

1. Worker runs `example_agent.entrypoint`.
2. `entrypoint`:
   - `await ctx.connect()`
   - constructs `AgentSession` (STT → LLM → TTS)
   - sets `allow_interruptions=False`
   - attaches `FillerAwareInterruptController`
   - starts the session.
3. Controller logic:
   - If agent speaking:
     - "uh / umm / hmm / haan" etc. → **ignored**
     - `"umm okay stop"` → **hard interrupt** → `session.interrupt()`
     - `"no not that one"` / `"i have a question"` → **real interrupt** → `session.interrupt()`
     - low-confidence `"hmm yeah"` → ignored if below `min_confidence`.
   - If agent not speaking:
     - all input is handled normally; fillers are *not* auto-discarded.

## Steps to Test

1. Run:

   ```bash
   python main.py console 



