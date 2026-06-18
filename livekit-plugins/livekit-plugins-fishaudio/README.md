# Fish Audio plugin for LiveKit Agents

Support for speech-to-text and voice synthesis with [Fish Audio](https://fish.audio/).

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

### Speech-to-text

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

stt = fishaudio.STT(language="en")

session = AgentSession(
    stt=stt,
    # ... llm, tts, etc.
)
```

To let Fish Audio auto-detect the spoken language, omit `language` or pass `language=None`.

### Text-to-speech

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

tts = fishaudio.TTS()

session = AgentSession(
    tts=tts,
    # ... stt, llm, etc.
)
```
