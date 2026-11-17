# Giggy plugin for LiveKit Agents

Support for [Giggy](https://giggy.ai/)'s text-to-speech and speech-to-text in LiveKit Agents.

LiveKit docs: [speech-to-text](https://docs.livekit.io/agents/models/stt/giggy/) and [text-to-speech](https://docs.livekit.io/agents/models/tts/giggy/) (live with the next release).

## Installation

```bash
pip install livekit-plugins-giggy
```

## Pre-requisites

You'll need an API key from Giggy. It can be set as an environment variable: `GIGGY_API_KEY`

Text-to-speech also needs a Giggy voice UUID, passed as `voice` or set as `GIGGY_VOICE_ID`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import giggy

session = AgentSession(
    stt=giggy.STT(language="en"),
    tts=giggy.TTS(voice="YOUR_GIGGY_VOICE_UUID"),
    # ... your LLM and other options
)
```

- `giggy.TTS` synthesizes each piece of text and streams back 24 kHz mono PCM audio. `speed` ranges from 0.25 to 4.
- `giggy.STT` streams live audio over a WebSocket and returns interim and final transcripts. Giggy detects the end of each utterance on the server.
- `language` defaults to automatic detection and accepts a language name or BCP-47 code, such as `"en-US"`.
