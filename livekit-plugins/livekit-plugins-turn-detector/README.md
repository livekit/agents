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

### English model

The English model is the smaller of the two models. It requires 200MB of RAM and completes inference in ~10ms

```python
from livekit.plugins.turn_detector.english import EnglishModel

session = AgentSession(
    ...
    turn_detection=EnglishModel(),
)
```

### Multilingual model

We've trained a separate multilingual model that supports the following languages: `English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Russian, Turkish`

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

## Model system requirements

The end-of-turn model is optimized to run on CPUs with modest system requirements. It is designed to run on the same server hosting your agents.

The model requires <500MB of RAM and runs within a shared inference server, supporting multiple concurrent sessions.

## License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).
