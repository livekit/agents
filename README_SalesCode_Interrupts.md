
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

###

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



### Bonus task 

## Bonus Features Implemented

### 1. Runtime-Configurable Ignore & Interrupt Lists

The interruption logic is not hard-coded. The `InterruptConfig` class exposes
helpers to **modify behavior at runtime**:

- `add_filler_words([...])` / `remove_filler_words([...])`
- `add_hard_phrases([...])` / `remove_hard_phrases([...])`

The `FillerAwareInterruptController.refresh_hard_commands()` method rebuilds its
internal command caches based on the updated config. This allows:

- Rapid tuning without redeploying the worker.
- Experimentation per-customer or per-language profile.
- Future admin/API endpoints to plug directly into these methods.

This satisfies the bonus requirement of _“Dynamically update ignored-word lists during runtime.”_

### 2. Multi-Language / Code-Mixed Filler & Command Support

The implementation is designed to handle **English + Hindi / Hinglish** and can
be easily extended to additional languages:

- `filler_words` includes tokens like:
  - English: `uh`, `umm`, `um`, `hmm`, `hmmm`, `huh`, `ah`, `oh`, `mmm`
  - Hindi / Hinglish: `haan`, `arey`, `acha`
- `hard_interrupt_phrases` includes:
  - English: `stop`, `wait`, `hold on`, `one second`, `listen`,
    `no not that one`, `not this`
  - Hindi / Hinglish: `bas`, `ruk`, `ruko`, `thoda ruk`, `thodi der ruk`, `ek second`

Because detection is:
- lowercased,
- token-based for single-word commands,
- substring-based for multi-word phrases,

it correctly handles **mixed utterances**, for example:

- `“umm thoda ruk please”` → immediate interrupt
- `“haan umm acha”` (while agent is speaking) → ignored as filler
- same words when the agent is silent → treated as normal user input

This satisfies the bonus requirement of _“Implement multi-language filler detection (e.g., Hindi + English mix).”_


