# LiveKit Interrupt Handler — feature/livekit-interrupt-handler-<yourname>

This feature adds an extension layer that distinguishes filler-only user speech (e.g. "uh", "umm") from genuine interruptions (e.g. "stop", "wait") while an agent is speaking — without modifying LiveKit's VAD.

## What changed
- `src/livekit_interrupt_filter.py` — the core async, thread-safe filter that:
  - ignores configurable filler words while the agent is speaking
  - registers filler as real speech when the agent is quiet
  - always treats explicit command words (configurable) as valid interrupts
  - uses ASR confidence thresholds to avoid false positives
  - supports dynamic ignored-word updates (programmatically or via `IGNORED_WORDS` env var and SIGHUP reload)
  - logs ignored/valid interruptions separately

- `examples/agent_with_filter_demo.py` — runnable demo and an integration sketch to drop into a LiveKit agent.

- `tests/test_interrupt_filter.py` — basic unit tests.

## Config parameters
- `IGNORED_WORDS` (env var): comma-separated filler words, e.g. `uh,umm,hmm,haan`.
- `filler_confidence_threshold` (constructor): float default `0.8`. If ASR confidence ≥ this and text contains only filler tokens → ignored while agent is speaking.
- `ignore_when_confidence_less_than` (constructor): float default `0.5`. If ASR confidence < this while agent speaking → ignored unless contains explicit command.

Programmatic options via `InterruptFilter(...)` constructor:
- `ignored_words`, `filler_confidence_threshold`, `ignore_when_confidence_less_than`, `extra_command_words`.

## How it works (quick)
1. ASR/transcription callback receives `(text, confidence, is_final)`.
2. Agent code queries whether agent TTS is playing (e.g. `agent.tts.is_playing()`).
3. Call `await filter.handle_transcription_event(text, confidence, is_final, agent_speaking)`.
4. The filter returns `{ action: 'ignore'|'stop'|'register', reason: ..., cleaned_text: ... }`.
5. Your agent acts accordingly:
   - `ignore` — drop event
   - `stop` — immediately stop/pause TTS and process user text
   - `register` — process user text as normal

## What works (tested)
- Filler-only words ignored while agent speaks.
- Same fillers registered as speech when agent quiet.
- Mixed filler + command recognized as command (agent stops).
- Low-confidence events while agent speaks ignored unless command present.
- Dynamic updates via `filter.update_ignored_words(new_list)`.

## Known issues / caveats
- This module relies on ASR transcripts & a confidence score; behaviour depends on ASR accuracy.
- `command_words` are heuristic; extend them for your domain/language.
- For multilingual setups, extend `IGNORED_WORDS` and `DEFAULT_COMMAND_WORDS` for each language.
- On Windows, SIGHUP reload might not be supported — call `filter.reload_from_env()` programmatically instead.

## Steps to test locally (no LiveKit required)
1. Checkout branch:
   ```bash
   git checkout -b feature/livekit-interrupt-handler-<yourname>
