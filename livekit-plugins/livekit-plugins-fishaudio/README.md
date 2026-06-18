# Fish Audio plugin for LiveKit Agents

Support for voice synthesis with [Fish Audio](https://fish.audio/).

- Docs: `https://docs.fish.audio/`

## Installation

```bash
pip install livekit-plugins-fishaudio
```

## Prerequisites

Obtain an API key from Fish Audio.

Set the API key as an environment variable:

```
FISH_API_KEY=<your_api_key>
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

# Basic usage with env-based credentials
tts = fishaudio.TTS()

session = AgentSession(
    tts=tts,
    # ... stt, llm, etc.
)
```
