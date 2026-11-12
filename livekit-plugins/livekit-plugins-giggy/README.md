# Giggy plugin for LiveKit Agents

Support for [Giggy](https://giggy.ai/)'s text-to-speech and speech-to-text in LiveKit Agents.

More information is available in the [Giggy speech API docs](https://giggy.ai/docs/speech-api).

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
- `giggy.STT` streams live audio over a WebSocket and returns interim and final transcripts. It uses Silero VAD to detect the end of each utterance; pass `vad=` to use a different LiveKit VAD.
- `language` defaults to automatic detection and accepts a language name or BCP-47 code, such as `"en-US"`.

## Tests

The offline tests don't need Giggy credentials:

```bash
uv run --package livekit-plugins-giggy pytest -q livekit-plugins/livekit-plugins-giggy/tests/
```
