# Gandr plugin for LiveKit Agents

Support for [Gandr](https://gandr.ai) text to speech.

See [https://docs.livekit.io/agents/integrations/tts/](https://docs.livekit.io/agents/integrations/tts/) for more information.

## Installation

```bash
pip install livekit-plugins-gandr
```

## Pre-requisites

You will need a Gandr API key. It can be set as an environment variable: `GANDR_API_KEY`.

## Usage

```python
from livekit.plugins import gandr

session = AgentSession(
    tts=gandr.TTS(voice="gandr-mia"),
)
```
