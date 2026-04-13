# Turn detector plugin for LiveKit Agents

This plugin introduces end-of-turn detection for LiveKit Agents using a custom open-weight model to determine when a user has finished speaking.

Traditional voice agents use VAD (voice activity detection) for end-of-turn detection. However, VAD models lack language understanding, often causing false positives where the agent interrupts the user before they finish speaking.

By leveraging a language model specifically trained for this task, this plugin offers a more accurate and robust method for detecting end-of-turns.

See [https://docs.livekit.io/agents/build/turns/turn-detector/](https://docs.livekit.io/agents/build/turns/turn-detector/) for more information.

## Installation

```bash
pip install livekit-plugins-turn-detector
```

## Usage

### Multilingual model

We've trained a multilingual model that supports the following languages: `English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Russian, Turkish, Hindi`

The multilingual model requires ~400MB of RAM and completes inferences in ~25ms.

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    ...
    turn_detection=MultilingualModel(),
)
```

### Usage with RealtimeModel

The turn detector can be used even with speech-to-speech models such as OpenAI's Realtime API. You'll need to provide a separate STT to ensure our model has access to the text content.

```python
session = AgentSession(
    ...
    stt=deepgram.STT(model="nova-3", language="multi"),
    llm=openai.realtime.RealtimeModel(),
    turn_detection=MultilingualModel(),
)
```

## Running your agent

This plugin requires model files. Before starting your agent for the first time, or when building Docker images for deployment, run the following command to download the model files:

```bash
python my_agent.py download-files
```

## Downloaded model files

Model files are downloaded to and loaded from the location specified by the `HF_HUB_CACHE` environment variable. If not set, this defaults to `$HF_HOME/hub` (typically `~/.cache/huggingface/hub`).

For offline deployment, download the model files first while connected to the internet, then copy the cache directory to your deployment environment.

## Model system requirements

The end-of-turn model is optimized to run on CPUs with modest system requirements. It is designed to run on the same server hosting your agents.

The model requires <500MB of RAM and runs within a shared inference server, supporting multiple concurrent sessions.

## License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).

## LLM-based turn detection (opt-in)

`LLMTurnDetector` is an alternative to the ONNX EOU model that uses any
`livekit.agents.llm.LLM` to classify turn completion. Useful when you already
pay for an LLM in the same conversation loop, want multilingual coverage
without a language table, or need semantic reasoning over the full context.

### Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import openai
from livekit.plugins.turn_detector import LLMTurnDetector

session = AgentSession(
    turn_detection=LLMTurnDetector(llm=openai.LLM(model="gpt-4o-mini")),
    # ... STT, TTS, LLM, etc.
)
```

### Tradeoffs vs the ONNX EOU model

| Dimension         | ONNX EOU                      | `LLMTurnDetector`             |
|-------------------|-------------------------------|-------------------------------|
| Cost per turn     | Free after download           | Paid LLM call                 |
| Latency           | ~10–50 ms CPU                 | 200–800 ms typical            |
| Setup             | Model download step           | Drop in any `llm.LLM`         |
| Language coverage | Fixed language table          | Whatever your LLM supports    |
| Reasoning quality | Classifier-only               | Full LLM semantics            |

The ONNX detector remains the default for most deployments. Reach for
`LLMTurnDetector` when its tradeoffs match your workload.

### Configuration

- `instructions`: override the default classification prompt (e.g., for
  non-English voice apps or domain-specific tuning).
- `unlikely_threshold` (default `0.5`): probability below which endpointing
  treats the turn as likely-incomplete and waits longer.
- `timeout` (default `1.5`): hard cap on the classifier call; on timeout the
  detector returns a neutral probability rather than blocking the agent.
- `max_history_turns` (default `6`): how many trailing chat messages are
  included in the classifier prompt.
