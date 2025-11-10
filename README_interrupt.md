# Interruption Handling Extension (Filler-Aware) for LiveKit Agents

## What Changed
This branch adds a lightweight **interruption layer** that distinguishes filler noises from real user interruptions while the agent is speaking—**without touching VAD**.
**Note: All the modules are added under `livekit-agents\livekit\extension` path of this repository**
**New modules (under `livekit/extension/`):**
- `config.py` — Tunable knobs:
  - `ignored_words: set[str]` (e.g., `uh, umm, hmm, haan`)
  - `priority_words: set[str]` (e.g., `stop, wait, hold on`)
  - `asr_conf_min: float` (default `0.55`)
  - `filler_ratio_min: float` (default `0.70`)
  - `debug_logging: bool` (default `True`)
- `filters.py` — Text normalization & helpers:
  - `normalize(text)`, `tokenize(text)`, `contains_any_phrase(text, phrases)`
- `interrupt_handler.py` — Core logic:
  - `ASRChunk(text, avg_confidence, is_final)`
  - Decisions: `IGNORE`, `ACCEPT`, `FORCE_STOP`
  - State methods: `on_tts_start()`, `on_tts_end()`, `classify(chunk)`
- `middleware.py` — Example wiring:
  - `LiveKitInterruptMiddleware(agent)` hooks transcript + TTS lifecycle and applies decisions

**Decision flow (while TTS is speaking):**
1. Priority phrase present → **FORCE_STOP**
2. Else `avg_confidence < asr_conf_min` → **IGNORE**
3. Else filler ratio ≥ `filler_ratio_min` → **IGNORE**
4. Else → **ACCEPT**

When the agent is **quiet** (after `on_tts_end()`), all detected speech is accepted.

---

## What Works
- Priority commands (e.g., “stop”, “wait”) interrupt immediately—even at low confidence
- Filler-only input while TTS is active is ignored
- Low-confidence murmurs are suppressed during TTS
- Real phrases (e.g., “ok got it”) correctly interrupt
- Proper state handling via `on_tts_start()` / `on_tts_end()`

**Unit tests (provider-free) cover:**
- Filler suppression during TTS  
- Accept when quiet (even fillers)  
- Priority overrides  
- Low-confidence suppression  
- Acceptance of real interruptions during TTS

---

## Known Issues / Edge Cases
- **ASR confidence scales differ** across providers → tune `ASR_CONF_MIN` per engine
- **Filler lists are language-specific** → extend `ignored_words` for your locale
- **Substring match** for priority phrases is simple by design; add an intent layer if you need semantics
- Highly granular partials may benefit from a tiny debounce (150–300 ms) before finalizing decisions
- Always pair `on_tts_start()` with `on_tts_end()` (including canceled playback) to keep state correct

---

## Steps to Test

### A) Fast, Offline Unit Test (no API keys)

**Layout:**
- livekit-agents/livekit/extension/
  - __init__.py
  - config.py
  - filters.py
  - interrupt_handler.py
  - middleware.py
- local_test/test_interrupt_handler.py #Free unit test which is easy to run
- demo_interrupt.py

The way i conducted the test was 

**Run:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel pytest
# run only the isolated test; bypass repo-level conftest
python -m pytest -q --confcutdir=local_tests local_tests/test_interrupt_handler.py
```
Further I conducted a sample interrupt case on the file `demo_interrup.py` which looked like the following 
```python
# demo_interrupt.py
from livekit.extension.interrupt_handler import InterruptHandler, ASRChunk, InterruptDecision

ih = InterruptHandler()  # default config
print("TTS start (agent speaking)")
ih.on_tts_start()

tests = [
    ("umm hmm", 0.9),
    ("wait one second", 0.3),
    ("ok got it", 0.95),
    ("stop please", 0.8),
    ("umm umm umm hmm umm umm hmm", 0.8)
]

for text, conf in tests:
    decision, meta = ih.classify(ASRChunk(text=text, avg_confidence=conf, is_final=True))
    print(f"{text!r:>18} -> {decision} [{meta['reason']}]")

print("TTS end (agent quiet)")
ih.on_tts_end()
text, conf = "umm", 0.2
decision, meta = ih.classify(ASRChunk(text=text, avg_confidence=conf, is_final=True))
print(f"{text!r:>18} -> {decision} [{meta['reason']}]")
```
The response to this demo was 
```bash
TTS start (agent speaking)
         'umm hmm' -> IGNORE [filler_dominated_while_speaking]
 'wait one second' -> FORCE_STOP [priority_word]
       'ok got it' -> ACCEPT [real_interruption_while_speaking]
     'stop please' -> FORCE_STOP [priority_word]
'umm umm umm hmm umm umm hmm' -> IGNORE [filler_dominated_while_speaking]
TTS end (agent quiet)
             'umm' -> ACCEPT [agent_quiet_accept]
```

## Wiring it into a Running agent
We can simple add the middleware after the agent is constructed according to user needs by doing the following: 
```python
from livekit.extension.middleware import LiveKitInterruptMiddleware

# after creating your Agent/session instance:
LiveKitInterruptMiddleware(agent)
```

We can also modify the environment knobs such as the ignored words and priority words which are currently: 
```bash
IGNORED_WORDS=uh,umm,um,hmm,haan,mm,uhh,erm,eh
PRIORITY_WORDS=stop,wait,hold on,pause,no,not that,one second
ASR_CONF_MIN=0.55
FILLER_RATIO_MIN=0.7
INTERRUPT_DEBUG=1
```
## Environment Details 
- **Python**: 3.12.1
- **Runtime Dependecies for `Extension`:** Standard Library only
- **Testing:** Pytest >= 8 and pytest-asyncio >= 0.23

## API Reference 

```python
# Data model for a transcript chunk
ASRChunk(text: str, avg_confidence: float, is_final: bool)

# Decisions
InterruptDecision.IGNORE
InterruptDecision.ACCEPT
InterruptDecision.FORCE_STOP

# Core handler
ih = InterruptHandler()           
ih.on_tts_start() ##Agent speaking = True                    
ih.on_tts_end() ##Agent Speaking = False                        
decision, meta = ih.classify(ASRChunk(...))
# meta: reason, tokens, avg_conf, filler_ratio, agent_speaking, etc.
```
## Tuning 
- If there are too many false stops we can increase `ASR_CONF_MIN` or `FILLER_RATIO_MIN`
- If legit commands are ignored: Add more `Priority words` and lower thresholds
- Mixed Language: Extend both `IGNORED_WORDS` and `PIRORITY_WORDS`
