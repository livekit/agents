# Rumik AI plugin for LiveKit Agents

Support for speech synthesis (TTS) with the [Rumik AI](https://rumik.ai/) API.

## Installation

```bash
pip install livekit-plugins-rumik
```

## Pre-requisites

You'll need an API key from Rumik AI. It can be set as an environment variable: `RUMIK_API_KEY`

## Usage

### Text-to-Speech

```python
from livekit.plugins import rumik

# Standard tone-steered TTS (muga)
tts = rumik.TTS(model="muga")

# Rich instruct-based TTS (mulberry)
tts = rumik.TTS(
    model="mulberry",
    description="warm, upbeat narrator",
    speaker="speaker_2",
)
```
