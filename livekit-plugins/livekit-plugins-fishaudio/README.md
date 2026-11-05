# Fish Audio plugin for LiveKit Agents

Support for voice synthesis and realtime speech-to-text with [Fish Audio](https://fish.audio/).

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

### Speech-to-text

`fishaudio.STT` streams audio to Fish Audio's realtime ASR. It works with LiveKit's turn detection:

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

session = AgentSession(
    stt=fishaudio.STT(),
    # ... llm, tts, vad, etc.
)
```

Fish Audio can also detect the end of each user turn itself:

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import fishaudio

session = AgentSession(
    stt=fishaudio.STT(),
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
    # ... llm, tts, etc.
)
```
