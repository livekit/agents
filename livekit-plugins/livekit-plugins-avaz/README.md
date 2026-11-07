# Avaz plugin for LiveKit Agents

Support for Turkish speech-to-text and text-to-speech with [Avaz](https://dashboard.mankyazilim.com.tr).

See the [Avaz API reference](https://dashboard.mankyazilim.com.tr/api/swagger/index.html) for more information.

## Installation

```bash
pip install livekit-plugins-avaz
```

## Pre-requisites

You'll need an API key from the [Avaz dashboard](https://dashboard.mankyazilim.com.tr). It can be set as an environment variable: `AVAZ_API_KEY`

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import avaz, silero

session = AgentSession(
    stt=avaz.STT(),
    tts=avaz.TTS(model="avaz3", speaker_id=5),
    vad=silero.VAD.load(),
    # ... llm
)
```

### STT

- Transcribes Turkish. It is not streaming: `AgentSession` uses its VAD to send each utterance once the user stops speaking, so a VAD is required.
- `base_url` (or `AVAZ_BASE_URL`): the API endpoint, for example of an on-premises deployment.

### TTS

- `model`: `avaz3` (48 kHz) or `avaz2` (24 kHz).
- `speaker_id`: a voice from the `GET /tts/voices` endpoint. Without it, Avaz uses the model's default voice. `tts.update_options(speaker_id=...)` changes it for the next streams.
- `base_url` (or `AVAZ_BASE_URL`): the API endpoint, for example of an on-premises deployment.
