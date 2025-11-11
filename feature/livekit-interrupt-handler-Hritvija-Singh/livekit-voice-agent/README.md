# LiveKit Voice Agent – Interrupt-Aware VAD Layer

This worker demonstrates how to wrap LiveKit’s built-in VAD with a state-aware
interceptor that:

- lets normal turn detection pass through with zero additional latency, and
- spins up a streaming ASR “decision window” while the agent is speaking so we
  can detect genuine barge-ins (“stop”, “wait”, etc.) without being confused by
  fillers (“uh”, “umm”, “haan”).

The implementation lives in:

- `agent.py` – bootstraps the worker, wires the `VADExtension`, environment
  configuration, and interruption handler.
- `vad_extension.py` – the extension layer that monitors agent state, mirrors
  VAD events, and manages the low-latency streaming ASR intercept.

---

## How It Works

1. **Agent state tracking** – the extension subscribes to the session’s
   `agent_state_changed` event. Whenever the agent enters the `"speaking"`
   state, we flip `AgentSpeechState` to `SPEAKING`; any other state maps to
   `IDLE`.

2. **Selective interception** – on every `START_OF_SPEECH` event from the base
   VAD:
   - If the agent is idle, the event is forwarded unchanged.
   - If the agent is speaking, we *swallow* the event, start queuing raw audio
     frames, and open a fresh streaming STT connection.

3. **Decision window** – the intercept runs a configurable timer
   (`LIVEKIT_DECISION_TIMEOUT`, default `0.7s`). Interim transcripts are
   normalized and compared against a configurable ignored-word list
   (`LIVEKIT_IGNORED_FILLERS`). The first non-filler token marks the interruption
   as real.

4. **Outcomes**  
   - *Valid interruption*: we log the transcript and call `session.interrupt(force=True)`
     so the agent pauses immediately.  
   - *Only fillers*: when the timer expires with no real words, we log the
     ignored fillers and drop the intercept; the user never hears a pause.

5. **Streaming STT** – by default we reuse the session’s STT factory
   (`assemblyai/universal-streaming:en`). Each intercept builds a transient
   stream; see **Latency tips** if you need even faster response.

---

## Configuration

You can tune the behaviour via environment variables (load via `.env.local` or
your preferred secrets manager):

| Variable | Description | Default |
| --- | --- | --- |
| `LIVEKIT_IGNORED_FILLERS` | Comma-separated fillers to ignore in any language. If the value is empty or unset, we fall back to a curated list covering common English and Hindi disfluencies (`uh, um, er, ah, oh, hmm, huh, haan, haanji, haina, achha, arey`). | fallback list |
| `LIVEKIT_DECISION_TIMEOUT` | Length (seconds) of the interruption decision window. Increase if your STT provider needs more setup time. | `0.7` |

Example `.env.local` snippet:

```env
LIVEKIT_IGNORED_FILLERS=uh,umm,haan,haina,achha
LIVEKIT_DECISION_TIMEOUT=1.0
```

---

## Running the Worker

1. Install dependencies (uses the repo’s workspace wiring):

```bash
uv sync
```

2. Populate environment variables (see above).  

3. Start the worker with the console interface:

```bash
cd livekit-voice-agent
../.venv/bin/python agent.py console
```

4. Talk over the agent while it speaks. Inspect the logs:
   - `INFO Valid interruption detected…` → the agent pauses.
   - `INFO Ignoring filler words…` → fillers were discarded and playback continues.

---

## Latency Tips

- **Quick win**: increase `LIVEKIT_DECISION_TIMEOUT` to `1.0`–`1.2` seconds if
  your provider needs extra setup time.
- **Provider swap**: plug in a faster streaming STT (e.g. Deepgram) by changing
  the model the session uses; the extension automatically reuses it.
- **Advanced**: keep a single streaming STT connection open inside
  `VADExtension` and reuse it across interrupts. This removes handshake latency
  entirely but requires more involved lifecycle management.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| “ImportError: cannot import name 'agents'” | Running from source without the workspace package path. | Ensure you start the worker via `agent.py` in this repo; it auto-injects the sibling `livekit-agents` directory into `sys.path`. |
| Interruption never triggers | Check logs for ignored fillers; expand your filler list or lower `LIVEKIT_DECISION_TIMEOUT`. |
| Frequent false positives | Add extra tokens to `LIVEKIT_IGNORED_FILLERS` or reduce the decision timeout. |

---

Feel free to adapt the extension for other languages or to integrate with your own dialog manager. Contributions welcome! ***

