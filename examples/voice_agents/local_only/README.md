# Fully local voice agent

A voice agent where every stage runs on your own hardware. No API keys, no cloud calls, no network dependency once models are downloaded.

| Stage | Component | How it connects |
|---|---|---|
| STT | [speaches](https://github.com/speaches-ai/speaches) (faster-whisper) | openai plugin + `base_url` |
| LLM | [Ollama](https://ollama.com) | `openai.LLM.with_ollama()` |
| TTS | [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) | openai plugin + `base_url` |
| VAD | silero | local ONNX (built in) |
| Turn detection | livekit turn-detector | local model (built in) |

The trick is that everything speaks the OpenAI API: the openai plugin's `base_url` parameter turns it into a universal adapter for any OpenAI-compatible local server. No new plugins needed.

## Setup

### 1. Ollama (LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

Any chat model works. Smaller models answer faster; see the latency table below before choosing.

### 2. speaches (STT)

```bash
docker run -d --gpus all -p 8000:8000 ghcr.io/speaches-ai/speaches:latest-cuda
```

CPU-only image also available (`latest-cpu`) — expect higher STT latency.

### 3. kokoro-fastapi (TTS)

```bash
docker run -d --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

### 4. Run the agent

```bash
python agent.py console   # terminal mode, local mic/speakers, no LiveKit server needed
python agent.py dev       # connect to a LiveKit server for real WebRTC sessions
```

Model and endpoint choices are environment-overridable — see the constants at the top of `agent.py`.

## Measured latency

Honest numbers, measured on real hardware with `bench.py` (5 runs per stage, warm path, medians). Reproduce them yourself: `python bench.py`.

**Test rig:** RTX 5080 16GB (Ollama LLM) + RTX 4060 Ti 16GB (kokoro container), STT on the 5080 via speaches CUDA container. Consumer parts, nothing exotic.

| Stage | Metric | Median | Spread (n=5) |
|---|---|---|---|
| TTS (kokoro) | first audio byte | **36ms** | 35–38ms |
| STT (faster-whisper-small) | full transcription, ~3s utterance | **359ms** | 356–366ms |
| LLM (llama3.2:3b) | time to first token | **127ms** | 115–130ms |
| LLM (llama3.2:3b) | generation throughput | **170 tok/s** | — |

Sum of stages after end of speech: **~520ms** before the agent starts speaking (plus turn-detection time). That's cloud-competitive, from hardware you own.

**Cold starts are real and separate:** first STT request loads the model (~20s), first LLM request a few seconds. They happen once per server lifetime — pre-pull the STT model with `curl -X POST http://localhost:8000/v1/models/Systran/faster-whisper-small` and send one warmup request at deploy time.

## Failure modes encountered

Recorded because the next person will hit them too:

1. **Thinking models are unusable for voice without care.** qwen3:4b streamed its first *reasoning* token at 142ms but its first *content* token at 18 seconds — the agent has the answer's ingredients almost instantly and says nothing while it deliberates. None of the obvious switches fixed it cleanly on Ollama's OpenAI-compatible endpoint: `/no_think` is ignored by newer qwen3 templates, `think: false` still stalled multiple seconds, and `reasoning_effort: "none"` moved the chain-of-thought *into* the content stream, which a voice agent would happily read aloud. Pick a non-thinking model for the voice path; that's why this example defaults to `llama3.2:3b`.
2. **New GPU architectures vs. prebuilt images.** The kokoro-fastapi GPU image's PyTorch build had no kernels for the RTX 5080 (Blackwell, sm_120) — `CUDA error: no kernel image is available`. Fix: pin the container to a supported GPU (`--gpus '"device=1"'` for the Ada card here) or use the CPU image.
3. **First-request latency is model load, not inference.** See cold starts above — don't benchmark (or alert on) the first request after a restart.

## Why run local?

- **Privacy** — audio never leaves the machine. Healthcare, legal, and anything under NDA becomes buildable.
- **Cost** — a busy agent's per-minute cloud bill becomes electricity.
- **The edge** — sites with no connectivity, or where connectivity is the failure mode.

The tradeoff is honesty about latency and quality: a local 4B model is not GPT-4.1, and this README's numbers table exists so you can decide with data instead of vibes.
