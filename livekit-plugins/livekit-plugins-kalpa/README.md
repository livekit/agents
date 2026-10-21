# Kalpa plugin for LiveKit Agents

Support for text-to-speech with [Kalpa](https://kalpalabs.ai/) in LiveKit Agents.

See the [Kalpa docs](https://docs.kalpalabs.ai/text-to-speech) for the API, models and voices.

## Installation

```bash
pip install livekit-plugins-kalpa
```

## Pre-requisites

You'll need an API key from Kalpa. It can be set as an environment variable: `KALPA_API_KEY`. You can create one [here](https://kalpalabs.ai/dashboard/keys/).

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import kalpa

session = AgentSession(
    tts=kalpa.TTS(voice="18c26f49-9f9c-400c-bec9-7467f28e1db5"),  # Arjun
    # ... stt, llm, etc.
)
```

`kalpa-tts-1` speaks English and Hindi. List the voices with `GET https://api.kalpalabs.ai/v1/voices`.
